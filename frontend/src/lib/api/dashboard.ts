import {
	asArray,
	asRecord,
	fetchApi,
	isNotFoundError,
} from './core';
import { getForvenStrategiesQuery } from './forven';
import type { ManualScannerRunResponse } from './forven';

// ============================================================================
// Dashboard
// ============================================================================

export interface DashboardKPIs {
	total_tested: number;
	best_sharpe: number;
	active_scans: number;
	signals_today: number;
	pipeline_count: number;
	data_coverage: number;
}

export interface DashboardOverview {
	kpis: DashboardKPIs;
	lifecycle_counts: Record<string, number>;
	blocked_count: number;
	last_ingestion_at: string | null;
	autopilot: {
		initialized?: boolean;
		running: boolean;
		paused: boolean;
		run_id: string | null;
		worker_concurrency: number;
		active_workers: number;
		queued_jobs: number;
		dead_letter_jobs: number;
		last_tick_error: string | null;
		health_ok: boolean | null;
		disabled_reason?: string | null;
	};
	timestamp: string;
}

export interface DashboardFunnelStage {
	state: string;
	count: number;
}

export interface DashboardExceptionItem {
	kind: string;
	id: string;
	strategy_id?: string | null;
	strategy_name?: string | null;
	job_type?: string;
	state?: string;
	message: string;
	ts: string;
	severity: 'high' | 'medium' | 'low' | string;
}

export interface DashboardActivityItem {
	type: 'lifecycle' | 'autopilot_job' | 'signal' | string;
	id: string;
	ts: string;
	title: string;
	detail: string;
	strategy_id?: string | null;
}

export interface DashboardActionItem {
	id: string;
	label: string;
	description: string;
	href: string;
	priority: number;
	kind: 'critical' | 'warning' | 'info' | 'ok' | string;
}

export interface LeaderboardEntry {
	id?: string;
	strategy_name: string;
	// Operator-editable friendly name; null = fall back to strategy_name.
	display_name?: string | null;
	symbol: string;
	timeframe: string;
	sharpe_ratio: number;
	total_return: number;
	monthly_return_pct?: number | null;
	annualized_return_pct?: number | null;
	max_drawdown: number;
	win_rate: number;
	total_trades: number;
	profit_factor: number;
	sortino_ratio: number;
	calmar_ratio: number;
	source: string;
	scan_id?: string;
	lifecycle_strategy_id?: string | null;
	tier?: QualityTier;
	mini_equity?: number[];
	deflated_sharpe?: number;
}

function toNumberOr(value: unknown, fallback = 0): number {
	const parsed = Number(value);
	return Number.isFinite(parsed) ? parsed : fallback;
}

function parseMaybeJsonValue<T = unknown>(value: unknown): T | unknown {
	if (typeof value !== 'string') return value;
	try {
		return JSON.parse(value) as T;
	} catch {
		return value;
	}
}

function asRecordOrEmpty(value: unknown): Record<string, unknown> {
	return asRecord(parseMaybeJsonValue(value)) ?? {};
}

function extractMetrics(value: unknown): Record<string, unknown> {
	return asRecordOrEmpty(value);
}

function strategyStageFromRow(row: Record<string, unknown>): string {
	const raw = String(row.stage ?? row.status ?? 'quick_screen').trim().toLowerCase().replace(/-/g, '_');
	if (!raw) return 'quick_screen';
	if (raw === 'researching' || raw === 'developing') return 'quick_screen';
	if (raw === 'backtesting') return 'gauntlet';
	if (raw === 'paper_trading' || raw === 'papertrading') return 'paper';
	if (raw === 'deployed' || raw === 'review' || raw === 'ceo_review' || raw === 'ceoreview') return 'live_graduated';
	if (raw === 'retired' || raw === 'killed' || raw === 'trash') return 'archived';
	return raw;
}

function tierFromSharpe(sharpe: number): 'elite' | 'strong' | 'marginal' | 'weak' {
	if (sharpe >= 2) return 'elite';
	if (sharpe >= 1) return 'strong';
	if (sharpe >= 0) return 'marginal';
	return 'weak';
}

function compareLeaderboardRows(a: LeaderboardEntry, b: LeaderboardEntry, sortBy: string): number {
	const key = (sortBy || 'sharpe_ratio').trim().toLowerCase();
	const aVal = (a as unknown as Record<string, unknown>)[key];
	const bVal = (b as unknown as Record<string, unknown>)[key];
	if (typeof aVal === 'string' || typeof bVal === 'string') {
		return String(bVal ?? '').localeCompare(String(aVal ?? ''));
	}
	return toNumberOr(bVal, 0) - toNumberOr(aVal, 0);
}

function coerceQualityTier(value: unknown, sharpeFallback: number): QualityTier {
	const normalized = String(value ?? '').trim().toLowerCase();
	if (normalized === 'elite' || normalized === 'strong' || normalized === 'marginal' || normalized === 'weak') {
		return normalized;
	}
	return tierFromSharpe(sharpeFallback);
}

function normalizeDashboardAction(raw: unknown, index: number): DashboardActionItem | null {
	const rec = asRecord(raw);
	if (!rec) return null;
	const id = String(rec.id ?? `action-${index}`).trim() || `action-${index}`;
	const label = String(rec.label ?? rec.title ?? rec.kind ?? `Action ${index + 1}`).trim() || `Action ${index + 1}`;
	const kind = String(rec.kind ?? 'info').trim() || 'info';
	const description = String(rec.description ?? rec.detail ?? label).trim() || label;
	const href = String(rec.href ?? '').trim()
		|| (kind === 'critical' || kind === 'warning' ? '/risk' : '/lab');
	const priority = toNumberOr(rec.priority, Math.max(0, 100 - index));
	return {
		id,
		label,
		description,
		href,
		priority,
		kind,
	};
}

function normalizeLeaderboardEntry(raw: unknown, index: number): LeaderboardEntry | null {
	const row = asRecord(raw);
	if (!row) return null;
	const metrics = extractMetrics(row.metrics);
	const strategyName = String(
		row.strategy_name ?? row.name ?? row.strategy ?? row.id ?? `Strategy ${index + 1}`
	).trim() || `Strategy ${index + 1}`;
	const symbol = String(row.symbol ?? row.asset ?? 'UNKNOWN').trim().toUpperCase() || 'UNKNOWN';
	const timeframe = String(row.timeframe ?? row.tf ?? '--').trim() || '--';
	const sharpe = toNumberOr(row.sharpe_ratio ?? row.sharpe ?? metrics.sharpe_ratio ?? metrics.sharpe, 0);
	const totalReturn = toNumberOr(
		row.total_return
		?? row.total_return_pct
		?? row.pnl_pct
		?? metrics.total_return
		?? metrics.total_return_pct
		?? metrics.pnl_pct,
		0,
	);
	const monthlyReturn = toNumberOr(
		row.monthly_return_pct ?? metrics.monthly_return_pct,
		totalReturn,
	);
	const annualizedReturn = toNumberOr(
		row.annualized_return_pct ?? metrics.annualized_return_pct,
		monthlyReturn * 12,
	);
	const winRateRaw = toNumberOr(row.win_rate ?? row.winRate ?? metrics.win_rate ?? metrics.winRate, 0);
	const winRate = winRateRaw > 1 ? winRateRaw : winRateRaw * 100;
	const totalTrades = toNumberOr(row.total_trades ?? row.trades ?? metrics.total_trades ?? metrics.trades, 0);
	const source = String(row.source ?? 'manual').trim() || 'manual';
	const scanId = String(row.scan_id ?? '').trim();
	const lifecycleStrategyId = String(row.lifecycle_strategy_id ?? row.strategy_id ?? '').trim();
	const miniEquity = asArray<number>(row.mini_equity ?? metrics.mini_equity).map((value) => toNumberOr(value, 0));
	const entryId = String(
		row.id ?? lifecycleStrategyId ?? scanId ?? `${strategyName}:${symbol}:${timeframe}:${index}`
	).trim();
	const displayName = String(row.display_name ?? '').trim() || null;
	return {
		id: entryId || `${strategyName}:${symbol}:${timeframe}:${index}`,
		strategy_name: strategyName,
		display_name: displayName,
		symbol,
		timeframe,
		sharpe_ratio: sharpe,
		total_return: totalReturn,
		monthly_return_pct: monthlyReturn,
		annualized_return_pct: annualizedReturn,
		max_drawdown: toNumberOr(row.max_drawdown ?? row.max_drawdown_pct ?? metrics.max_drawdown ?? metrics.max_drawdown_pct, 0),
		win_rate: winRate,
		total_trades: totalTrades,
		profit_factor: toNumberOr(row.profit_factor ?? row.pf ?? metrics.profit_factor ?? metrics.pf, 0),
		sortino_ratio: toNumberOr(row.sortino_ratio ?? metrics.sortino_ratio, 0),
		calmar_ratio: toNumberOr(row.calmar_ratio ?? metrics.calmar_ratio, 0),
		source,
		scan_id: scanId || undefined,
		lifecycle_strategy_id: lifecycleStrategyId || undefined,
		tier: coerceQualityTier(row.tier, sharpe),
		mini_equity: miniEquity,
		deflated_sharpe: toNumberOr(row.deflated_sharpe ?? metrics.deflated_sharpe, sharpe),
	};
}

async function getLegacyDashboardOverview(): Promise<DashboardOverview> {
	const [dashboardResult, strategiesResult] = await Promise.allSettled([
		fetchApi<Record<string, unknown>>('/dashboard'),
		getForvenStrategiesQuery(),
	]);
	const dashboard = dashboardResult.status === 'fulfilled' ? (dashboardResult.value ?? {}) : {};
	const strategyRows = strategiesResult.status === 'fulfilled' ? strategiesResult.value : [];

	const lifecycleCounts: Record<string, number> = {};
	let bestSharpe = Number.NEGATIVE_INFINITY;
	for (const raw of strategyRows) {
		const row = asRecord(raw);
		if (!row) continue;
		const stage = strategyStageFromRow(row);
		lifecycleCounts[stage] = (lifecycleCounts[stage] ?? 0) + 1;
		const metrics = extractMetrics(row.metrics);
		const sharpe = toNumberOr(metrics.sharpe_ratio ?? metrics.sharpe, Number.NaN);
		if (Number.isFinite(sharpe)) bestSharpe = Math.max(bestSharpe, sharpe);
	}

	const strategyCount = strategyRows.length;
	const daemonRunning = Boolean((dashboard as Record<string, unknown>).daemon_running);
	const tradingAllowed = Boolean((dashboard as Record<string, unknown>).trading_allowed);
	const tradingReason = (dashboard as Record<string, unknown>).trading_reason;

	return {
		kpis: {
			total_tested: strategyCount,
			best_sharpe: Number.isFinite(bestSharpe) ? bestSharpe : 0,
			active_scans: toNumberOr((dashboard as Record<string, unknown>).scan_count, 0),
			signals_today: 0,
			pipeline_count: strategyCount,
			data_coverage: 0,
		},
		lifecycle_counts: lifecycleCounts,
		blocked_count: 0,
		last_ingestion_at: ((dashboard as Record<string, unknown>).last_scan as string | null) ?? null,
		autopilot: {
			initialized: true,
			running: daemonRunning,
			paused: !tradingAllowed,
			run_id: null,
			worker_concurrency: 0,
			active_workers: 0,
			queued_jobs: 0,
			dead_letter_jobs: 0,
			last_tick_error: null,
			health_ok: daemonRunning,
			disabled_reason: tradingReason ? String(tradingReason) : null,
		},
		timestamp: new Date().toISOString(),
	};
}

async function getLegacyLeaderboardEntries(): Promise<LeaderboardEntry[]> {
	const rows = await getForvenStrategiesQuery();
	const mapped: LeaderboardEntry[] = [];
	for (const raw of rows) {
		const row = asRecord(raw);
		if (!row) continue;
		const metrics = extractMetrics(row.metrics);
		const sharpe = toNumberOr(metrics.sharpe_ratio ?? metrics.sharpe, 0);
		const totalReturn = toNumberOr(metrics.total_return ?? metrics.total_return_pct ?? metrics.pnl_pct, 0);
		const monthlyReturn = toNumberOr(metrics.monthly_return_pct ?? totalReturn, totalReturn);
		const annualizedReturn = toNumberOr(metrics.annualized_return_pct ?? (monthlyReturn * 12), monthlyReturn * 12);
		const maxDrawdown = toNumberOr(metrics.max_drawdown ?? metrics.max_drawdown_pct, 0);
		const winRateRaw = toNumberOr(metrics.win_rate ?? metrics.winRate, 0);
		const winRate = winRateRaw > 1 ? winRateRaw : winRateRaw * 100;

		mapped.push({
			id: String(row.id ?? ''),
			strategy_name: String(row.name ?? row.id ?? 'Unnamed Strategy'),
			display_name: String(row.display_name ?? '').trim() || null,
			symbol: String(row.symbol ?? 'BTC').toUpperCase(),
			timeframe: String(row.timeframe ?? '1h'),
			sharpe_ratio: sharpe,
			total_return: totalReturn,
			monthly_return_pct: monthlyReturn,
			annualized_return_pct: annualizedReturn,
			max_drawdown: maxDrawdown,
			win_rate: winRate,
			total_trades: toNumberOr(metrics.total_trades ?? metrics.trades, 0),
			profit_factor: toNumberOr(metrics.profit_factor ?? metrics.pf, 0),
			sortino_ratio: toNumberOr(metrics.sortino_ratio, 0),
			calmar_ratio: toNumberOr(metrics.calmar_ratio, 0),
			source: 'manual',
			lifecycle_strategy_id: row.id ? String(row.id) : null,
			tier: tierFromSharpe(sharpe),
			mini_equity: asArray<number>(metrics.mini_equity).map((v) => toNumberOr(v, 0)),
			deflated_sharpe: toNumberOr(metrics.deflated_sharpe, sharpe),
		});
	}
	return mapped;
}

export async function getDashboardKPIs(): Promise<DashboardKPIs> {
	try {
		return await fetchApi('/dashboard/kpis');
	} catch (error) {
		if (!isNotFoundError(error)) throw error;
		return (await getLegacyDashboardOverview()).kpis;
	}
}

export async function getDashboardOverview(): Promise<DashboardOverview> {
	try {
		return await fetchApi('/dashboard/overview');
	} catch (error) {
		if (!isNotFoundError(error)) throw error;
		return getLegacyDashboardOverview();
	}
}

export async function getDashboardFunnel(): Promise<DashboardFunnelStage[]> {
	return fetchApi('/dashboard/funnel');
}

export async function getDashboardExceptions(limit = 30): Promise<DashboardExceptionItem[]> {
	return fetchApi(`/dashboard/exceptions?limit=${limit}`);
}

function normalizeActivityTimestamp(raw: unknown): string {
	if (raw === null || raw === undefined) return '';
	if (typeof raw === 'number' && Number.isFinite(raw)) {
		return new Date(raw).toISOString();
	}
	const asString = String(raw).trim();
	if (!asString) return '';
	if (/^\d+$/.test(asString)) {
		const epoch = Number(asString);
		if (Number.isFinite(epoch)) return new Date(epoch).toISOString();
	}
	const parsed = new Date(asString);
	if (Number.isNaN(parsed.getTime())) return '';
	return asString;
}

function normalizeDashboardActivityItem(raw: unknown, index: number): DashboardActivityItem | null {
	const rec = asRecord(raw);
	if (!rec) return null;
	const source = String(rec.source ?? '').trim();
	const normalizedSource = source.toLowerCase();
	const rawType = String(rec.type ?? '').trim().toLowerCase();
	let activityType: DashboardActivityItem['type'];
	if (rawType) {
		activityType = rawType;
	} else if (normalizedSource.includes('signal') || normalizedSource.includes('scanner')) {
		activityType = 'signal';
	} else if (
		normalizedSource.includes('scheduler')
		|| normalizedSource.includes('autopilot')
		|| normalizedSource === 'brain'
	) {
		activityType = 'autopilot_job';
	} else {
		activityType = 'lifecycle';
	}

	const dataField = rec.data ?? rec.details;
	const dataRecord = asRecord(parseMaybeJsonValue(dataField));
	const detail = typeof dataField === 'string'
		? dataField
		: dataField
			? JSON.stringify(dataField)
			: '';

	const ts = normalizeActivityTimestamp(rec.ts ?? rec.timestamp ?? rec.created_at);
	const title = String(rec.title ?? rec.message ?? rec.msg ?? source ?? '').trim() || 'Activity';
	const id = String(rec.id ?? `${ts || 'activity'}-${index}`);
	const strategyId = dataRecord
		? String(dataRecord.strategy_id ?? dataRecord.strategy ?? '').trim() || undefined
		: undefined;

	return {
		type: activityType,
		id,
		ts,
		title,
		detail,
		strategy_id: strategyId,
	};
}

export async function getDashboardActivity(limit = 50): Promise<DashboardActivityItem[]> {
	try {
		const payload = await fetchApi<unknown>(`/dashboard/activity?limit=${limit}`);
		return asArray(payload)
			.map((row, index) => normalizeDashboardActivityItem(row, index))
			.filter((row): row is DashboardActivityItem => Boolean(row));
	} catch (error) {
		if (!isNotFoundError(error)) throw error;
		const rows = await fetchApi<Array<Record<string, unknown>>>(`/logs?limit=${limit}`);
		return rows
			.map((row, index) => normalizeDashboardActivityItem(row, index))
			.filter((row): row is DashboardActivityItem => Boolean(row));
	}
}

export async function getDashboardActions(): Promise<DashboardActionItem[]> {
	try {
		const payload = await fetchApi<unknown>('/dashboard/actions');
		const rows = asArray(payload);
		return rows
			.map((row, index) => normalizeDashboardAction(row, index))
			.filter((row): row is DashboardActionItem => Boolean(row));
	} catch (error) {
		if (!isNotFoundError(error)) throw error;
		const overview = await getDashboardOverview();
		const actions: DashboardActionItem[] = [];
		if (!overview.autopilot.running) {
			actions.push({
				id: 'legacy-start-daemon',
				label: 'Start Daemon',
				description: 'Daemon appears offline. Restart orchestration services.',
				href: '/lab?tab=247',
				priority: 100,
				kind: 'critical',
			});
		}
		if (overview.autopilot.disabled_reason || overview.autopilot.paused) {
			actions.push({
				id: 'legacy-review-risk',
				label: 'Review Risk',
				description: overview.autopilot.disabled_reason || 'Trading is paused. Review risk controls.',
				href: '/risk',
				priority: 90,
				kind: 'warning',
			});
		}
		actions.push({
			id: 'legacy-open-pipeline',
			label: 'Open Strategy Lab',
			description: 'Review strategy phases and pending handoffs in the Strategy Lab.',
			href: '/lab',
			priority: 70,
			kind: 'info',
		});
		return actions;
	}
}

export async function getDashboardLeaderboard(
	opts?: {
		sort_by?: string;
		limit?: number;
		min_sharpe?: number;
		symbol?: string;
		timeframe?: string;
		tier?: QualityTier | string;
	}
): Promise<LeaderboardEntry[]> {
	const p = new URLSearchParams();
	if (opts?.sort_by) p.set('sort_by', opts.sort_by);
	if (opts?.limit) p.set('limit', String(opts.limit));
	if (opts?.min_sharpe !== undefined) p.set('min_sharpe', String(opts.min_sharpe));
	if (opts?.symbol) p.set('symbol', opts.symbol);
	if (opts?.timeframe) p.set('timeframe', opts.timeframe);
	if (opts?.tier) p.set('tier', opts.tier);
	try {
		const payload = await fetchApi<unknown>(`/dashboard/leaderboard?${p}`);
		const normalized = asArray(payload)
			.map((row, index) => normalizeLeaderboardEntry(row, index))
			.filter((row): row is LeaderboardEntry => Boolean(row));
		const filtered = normalized.filter((row) => {
			if (opts?.symbol && row.symbol !== opts.symbol) return false;
			if (opts?.timeframe && row.timeframe !== opts.timeframe) return false;
			if (opts?.tier && row.tier !== opts.tier) return false;
			if (opts?.min_sharpe !== undefined && row.sharpe_ratio < opts.min_sharpe) return false;
			return true;
		});
		filtered.sort((a, b) => compareLeaderboardRows(a, b, opts?.sort_by ?? 'sharpe_ratio'));
		return filtered.slice(0, opts?.limit ?? filtered.length);
	} catch (error) {
		if (!isNotFoundError(error)) throw error;
		const fallbackRows = await getLegacyLeaderboardEntries();
		const filtered = fallbackRows.filter((row) => {
			if (opts?.symbol && row.symbol !== opts.symbol) return false;
			if (opts?.timeframe && row.timeframe !== opts.timeframe) return false;
			if (opts?.tier && row.tier !== opts.tier) return false;
			if (opts?.min_sharpe !== undefined && row.sharpe_ratio < opts.min_sharpe) return false;
			return true;
		});
		filtered.sort((a, b) => compareLeaderboardRows(a, b, opts?.sort_by ?? 'sharpe_ratio'));
		return filtered.slice(0, opts?.limit ?? 30);
	}
}

export async function getDashboardEquityCurves(scanId?: string, n = 5): Promise<unknown[]> {
	const p = new URLSearchParams();
	if (scanId) p.set('scan_id', scanId);
	p.set('n', String(n));
	try {
		return await fetchApi(`/dashboard/equity-curves?${p}`);
	} catch (error) {
		if (!isNotFoundError(error)) throw error;
		return [];
	}
}

export async function getDashboardCoverage(): Promise<{ coverage: Record<string, unknown> }> {
	try {
		return await fetchApi('/dashboard/coverage');
	} catch (error) {
		if (!isNotFoundError(error)) throw error;
		return { coverage: {} };
	}
}

export async function getDashboardSuggestions(): Promise<unknown[]> {
	return fetchApi('/dashboard/suggestions');
}

// ---- Quality Triage ----

export type QualityTier = 'elite' | 'strong' | 'marginal' | 'weak';

export interface TierDistribution {
	elite: number;
	strong: number;
	marginal: number;
	weak: number;
	tiers: Record<QualityTier, number>;
	total: number;
}

export interface PruningFunnelStage {
	stage: string;
	count: number;
}

export interface WinnerEntry {
	id: string;
	strategy_name: string;
	symbol: string;
	timeframe: string;
	deflated_sharpe: number;
	total_return: number;
	monthly_return_pct?: number | null;
	annualized_return_pct?: number | null;
	max_drawdown: number;
	total_trades: number;
	tier: QualityTier;
	created_at: string;
	scan_id: string;
}

function normalizeWinnerEntry(raw: unknown, index: number): WinnerEntry | null {
	const rec = asRecord(raw);
	if (!rec) return null;
	const strategyName = String(
		rec.strategy_name ?? rec.strategy ?? rec.name ?? rec.id ?? `Winner ${index + 1}`
	).trim() || `Winner ${index + 1}`;
	const symbol = String(rec.symbol ?? rec.asset ?? 'UNKNOWN').trim().toUpperCase() || 'UNKNOWN';
	const timeframe = String(rec.timeframe ?? rec.tf ?? '--').trim() || '--';
	const deflatedSharpe = toNumberOr(rec.deflated_sharpe ?? rec.sharpe_ratio ?? rec.sharpe, 0);
	const createdAtRaw = String(rec.created_at ?? rec.closed_at ?? rec.updated_at ?? '').trim();
	const totalReturn = toNumberOr(rec.total_return ?? rec.pnl_pct ?? rec.return_pct ?? rec.pnl_usd, 0);
	const monthly = rec.monthly_return_pct === null || rec.monthly_return_pct === undefined
		? null
		: toNumberOr(rec.monthly_return_pct, 0);
	const annualized = rec.annualized_return_pct === null || rec.annualized_return_pct === undefined
		? null
		: toNumberOr(rec.annualized_return_pct, 0);
	const scanId = String(rec.scan_id ?? rec.strategy_id ?? '').trim();
	return {
		id: String(rec.id ?? `${strategyName}:${symbol}:${timeframe}:${index}`).trim() || `${strategyName}:${symbol}:${timeframe}:${index}`,
		strategy_name: strategyName,
		symbol,
		timeframe,
		deflated_sharpe: deflatedSharpe,
		total_return: totalReturn,
		monthly_return_pct: monthly,
		annualized_return_pct: annualized,
		max_drawdown: toNumberOr(rec.max_drawdown ?? rec.max_drawdown_pct ?? rec.drawdown_pct, 0),
		total_trades: toNumberOr(rec.total_trades ?? rec.trades ?? rec.trade_count, 1),
		tier: coerceQualityTier(rec.tier, deflatedSharpe),
		created_at: createdAtRaw || new Date().toISOString(),
		scan_id: scanId,
	};
}

export async function getDashboardTierDistribution(scanId?: string): Promise<TierDistribution> {
	const p = new URLSearchParams();
	if (scanId) p.set('scan_id', scanId);
	const query = p.toString();
	let payload: Record<string, unknown> | null = null;
	try {
		payload = asRecord(await fetchApi<unknown>(`/dashboard/tier-distribution${query ? `?${query}` : ''}`));
	} catch (error) {
		if (!isNotFoundError(error)) throw error;
		const leaderboard = await getDashboardLeaderboard({ limit: 500 });
		const tiers = {
			elite: leaderboard.filter((entry) => (entry.tier ?? tierFromSharpe(entry.sharpe_ratio)) === 'elite').length,
			strong: leaderboard.filter((entry) => (entry.tier ?? tierFromSharpe(entry.sharpe_ratio)) === 'strong').length,
			marginal: leaderboard.filter((entry) => (entry.tier ?? tierFromSharpe(entry.sharpe_ratio)) === 'marginal').length,
			weak: leaderboard.filter((entry) => (entry.tier ?? tierFromSharpe(entry.sharpe_ratio)) === 'weak').length,
		};
		const total = tiers.elite + tiers.strong + tiers.marginal + tiers.weak;
		return {
			elite: tiers.elite,
			strong: tiers.strong,
			marginal: tiers.marginal,
			weak: tiers.weak,
			tiers,
			total,
		};
	}
	const payloadTiers = asRecord(payload?.tiers);
	const fallback = {
		elite: Number(payload?.elite ?? 0),
		strong: Number(payload?.strong ?? 0),
		marginal: Number(payload?.marginal ?? 0),
		weak: Number(payload?.weak ?? 0),
	};
	const tiers = {
		elite: Number(payloadTiers?.elite ?? fallback.elite),
		strong: Number(payloadTiers?.strong ?? fallback.strong),
		marginal: Number(payloadTiers?.marginal ?? fallback.marginal),
		weak: Number(payloadTiers?.weak ?? fallback.weak),
	};
	const total = Number(payload?.total ?? (tiers.elite + tiers.strong + tiers.marginal + tiers.weak));
	return {
		elite: tiers.elite,
		strong: tiers.strong,
		marginal: tiers.marginal,
		weak: tiers.weak,
		tiers,
		total,
	};
}

export async function getDashboardWinners(limit = 10): Promise<WinnerEntry[]> {
	try {
		const payload = await fetchApi<unknown>(`/dashboard/winners?limit=${limit}`);
		const normalized = asArray(payload)
			.map((row, index) => normalizeWinnerEntry(row, index))
			.filter((row): row is WinnerEntry => Boolean(row));
		normalized.sort((a, b) => {
			if (b.deflated_sharpe !== a.deflated_sharpe) return b.deflated_sharpe - a.deflated_sharpe;
			return b.total_return - a.total_return;
		});
		return normalized.slice(0, limit);
	} catch (error) {
		if (!isNotFoundError(error)) throw error;
		const leaderboard = await getDashboardLeaderboard({ sort_by: 'sharpe_ratio', limit: Math.max(limit, 30) });
		return leaderboard
			.filter((entry) => (entry.tier ?? tierFromSharpe(entry.sharpe_ratio)) !== 'weak')
			.slice(0, limit)
			.map((entry, idx) => ({
				id: String(entry.id ?? `${entry.strategy_name}-${idx}`),
				strategy_name: entry.strategy_name,
				symbol: entry.symbol,
				timeframe: entry.timeframe,
				deflated_sharpe: toNumberOr(entry.deflated_sharpe, entry.sharpe_ratio),
				total_return: toNumberOr(entry.total_return, 0),
				monthly_return_pct: entry.monthly_return_pct ?? null,
				annualized_return_pct: entry.annualized_return_pct ?? null,
				max_drawdown: toNumberOr(entry.max_drawdown, 0),
				total_trades: toNumberOr(entry.total_trades, 0),
				tier: entry.tier ?? tierFromSharpe(entry.sharpe_ratio),
				created_at: new Date().toISOString(),
				scan_id: String(entry.scan_id ?? ''),
			}));
	}
}

export async function getScanFunnel(scanId: string): Promise<PruningFunnelStage[]> {
	return fetchApi(`/scanner/scans/${scanId}/funnel`);
}

// ============================================================================
// Quant Factory Dashboard
// ============================================================================

export interface QuantFactoryAccount {
	account_value: number;
	net_exposure: number;
	daily_pnl_usd: number;
	daily_pnl_pct: number;
	execution_mode: string;
	trading_allowed: boolean;
	trading_reason: string;
	kill_switch_active: boolean;
	daemon_running: boolean;
	drawdown_pct: number;
	prices: Record<string, number>;
}

export interface QuantFactoryRadarEntry {
	id?: string;
	display_id?: string | null;
	strategy_name?: string | null;
	strategy: string;
	target: string;
	timeframe?: string;
	regime: string;
	stage: string;
	alpha: string;
	sharpe: number;
	trend: 'up' | 'down';
	model?: string;
}

export interface QuantFactoryAgent {
	id: string;
	name: string;
	role: string;
	enabled: boolean;
	model: string;
	status: 'active' | 'pending' | 'idle' | 'disabled';
	status_label: string;
}

export interface QuantFactoryLogEntry {
	id?: number;
	time: string;
	tag: string;
	layer: 'exec' | 'decay' | 'brain';
	level: string;
	msg: string;
}

export interface QuantFactoryArenaEntry {
	symbol: string;
	champion: { name: string; sharpe: number; return: number };
	challenger: { name: string; sharpe: number; return: number };
	edge_pct: number;
	threshold_pct: number;
}

export interface QuantFactoryValidation {
	is: { trades: number; sharpe: number; max_dd: number; win_rate: number };
	oos: { trades: number; sharpe: number; max_dd: number; win_rate: number };
	robustness: number;
	degradation_pct: number;
	strategy_name: string;
	status: string;
}

export interface QuantFactoryIntel {
	total_strategies: number;
	live: number;
	paper: number;
	backtesting: number;
	researching: number;
	total_trades: number;
	open_trades: number;
	avg_slippage_bps: number;
	total_backtests: number;
	agent_count: number;
}

export interface QuantFactoryData {
	account: QuantFactoryAccount;
	radar: QuantFactoryRadarEntry[];
	agents: QuantFactoryAgent[];
	logs: QuantFactoryLogEntry[];
	arena: QuantFactoryArenaEntry[];
	validation: QuantFactoryValidation;
	intel: QuantFactoryIntel;
}

export async function getQuantFactoryData(): Promise<QuantFactoryData> {
	return fetchApi('/quant-factory/');
}

// ============================================================================
// Signal Monitoring
// ============================================================================

export interface Signal {
	id: string;
	strategy_name: string;
	symbol: string;
	timeframe: string;
	signal_type: string;
	price: number;
	bar_timestamp: string;
	created_at: string;
	notified: boolean;
}

export interface MonitoredStrategy {
	id: string;
	strategy_name: string;
	definition_json: Record<string, unknown>;
	symbol: string;
	timeframe: string;
	active: boolean;
	created_at: string;
}

export async function getRecentSignals(limit = 50): Promise<Signal[]> {
	return fetchApi(`/signals/recent?limit=${limit}`);
}

export async function getMonitoredStrategies(): Promise<MonitoredStrategy[]> {
	return fetchApi('/signals/monitored');
}

export async function addMonitoredStrategy(
	strategyName: string, definitionJson: Record<string, unknown>, symbol: string, timeframe: string
): Promise<{ id: string }> {
	return fetchApi('/signals/monitored', {
		method: 'POST',
		body: JSON.stringify({
			strategy_name: strategyName,
			definition_json: definitionJson,
			symbol,
			timeframe,
		}),
	});
}

export async function removeMonitoredStrategy(id: string): Promise<{ status: string }> {
	return fetchApi(`/signals/monitored/${id}`, { method: 'DELETE' });
}

export async function checkSignalsNow(): Promise<ManualScannerRunResponse> {
	return fetchApi('/system/scanner/signal-run', { method: 'POST' });
}

// ============================================================================
// Pipeline funnel report (gate rejections)
// ============================================================================

export interface GateRejectionRow {
	gate: string;
	reason_code: string;
	count: number;
}

export interface PipelineFunnelReport {
	period_days: number;
	stage_counts: Record<string, number>;
	total_strategies: number;
	flows: Array<Record<string, unknown>>;
	gate_rejections: GateRejectionRow[];
	timeout_count: number;
	backtest_results_count: number;
	heartbeat_alert: boolean;
}

export async function getPipelineFunnelReport(days = 7): Promise<PipelineFunnelReport> {
	const payload = asRecord(await fetchApi<unknown>(`/pipeline/funnel-report?days=${days}`)) ?? {};
	const rejections = asArray(payload.gate_rejections)
		.map((row) => {
			const rec = asRecord(row);
			if (!rec) return null;
			return {
				gate: String(rec.gate ?? 'unknown').trim() || 'unknown',
				reason_code: String(rec.reason_code ?? 'unknown').trim() || 'unknown',
				count: toNumberOr(rec.count, 0),
			};
		})
		.filter((row): row is GateRejectionRow => Boolean(row));
	rejections.sort((a, b) => b.count - a.count);
	return {
		period_days: toNumberOr(payload.period_days, days),
		stage_counts: (asRecord(payload.stage_counts) ?? {}) as Record<string, number>,
		total_strategies: toNumberOr(payload.total_strategies, 0),
		flows: asArray(payload.flows).map((row) => asRecord(row) ?? {}),
		gate_rejections: rejections,
		timeout_count: toNumberOr(payload.timeout_count, 0),
		backtest_results_count: toNumberOr(payload.backtest_results_count, 0),
		heartbeat_alert: Boolean(payload.heartbeat_alert),
	};
}

// ============================================================================
// Task / queue health (from GET /api/health runtime summary)
// ============================================================================

export interface TaskHealthQueues {
	agent_pending: number;
	agent_running: number;
	agent_stale_pending: number;
	agent_stale_running: number;
	brain_pending: number;
	brain_running: number;
	brain_stale_pending: number;
	brain_stale_running: number;
}

export interface TaskHealth {
	status: string;
	issues: string[];
	queues: TaskHealthQueues;
	long_running_scheduler_jobs: number;
	overdue_due_scheduler_jobs: number;
}

export async function getTaskHealth(): Promise<TaskHealth> {
	const payload = asRecord(await fetchApi<unknown>('/health')) ?? {};
	const details = asRecord(payload.details) ?? {};
	const queues = asRecord(details.queues) ?? {};
	return {
		status: String(payload.status ?? 'unknown').trim() || 'unknown',
		issues: asArray(payload.issues).map((issue) => String(issue ?? '')).filter(Boolean),
		queues: {
			agent_pending: toNumberOr(queues.agent_pending, 0),
			agent_running: toNumberOr(queues.agent_running, 0),
			agent_stale_pending: toNumberOr(queues.agent_stale_pending, 0),
			agent_stale_running: toNumberOr(queues.agent_stale_running, 0),
			brain_pending: toNumberOr(queues.brain_pending, 0),
			brain_running: toNumberOr(queues.brain_running, 0),
			brain_stale_pending: toNumberOr(queues.brain_stale_pending, 0),
			brain_stale_running: toNumberOr(queues.brain_stale_running, 0),
		},
		long_running_scheduler_jobs: toNumberOr(details.long_running_scheduler_jobs, 0),
		overdue_due_scheduler_jobs: toNumberOr(details.overdue_due_scheduler_jobs, 0),
	};
}

// ============================================================================
// System pulse (full GET /api/health runtime detail for the ops dashboard)
// ============================================================================

export interface SystemPulseWorkerLoop {
	name: string;
	fresh: boolean;
	ageSeconds: number | null;
}

export interface SystemPulse {
	status: string;
	issues: string[];
	schedulerAgeSeconds: number | null;
	workerLoops: SystemPulseWorkerLoop[];
	queues: TaskHealthQueues;
	overdueJobs: number;
	overdueJobIds: string[];
	longRunningJobs: number;
	runtimeOwner: string;
}

export async function getSystemPulse(): Promise<SystemPulse> {
	const payload = asRecord(await fetchApi<unknown>('/health')) ?? {};
	const details = asRecord(payload.details) ?? {};
	const queues = asRecord(details.queues) ?? {};
	const worker = asRecord(details.api_task_worker) ?? {};
	const loops = asRecord(worker.loops) ?? {};

	const workerLoops: SystemPulseWorkerLoop[] = Object.entries(loops)
		.map(([name, raw]) => {
			const rec = asRecord(raw) ?? {};
			const age = Number(rec.age_seconds);
			return {
				name,
				fresh: Boolean(rec.fresh),
				ageSeconds: Number.isFinite(age) ? age : null,
			};
		})
		.sort((a, b) => a.name.localeCompare(b.name));

	const schedulerAge = Number(details.scheduler_age_seconds);

	return {
		status: String(payload.status ?? 'unknown').trim() || 'unknown',
		issues: asArray(payload.issues).map((issue) => String(issue ?? '')).filter(Boolean),
		schedulerAgeSeconds: Number.isFinite(schedulerAge) ? schedulerAge : null,
		workerLoops,
		queues: {
			agent_pending: toNumberOr(queues.agent_pending, 0),
			agent_running: toNumberOr(queues.agent_running, 0),
			agent_stale_pending: toNumberOr(queues.agent_stale_pending, 0),
			agent_stale_running: toNumberOr(queues.agent_stale_running, 0),
			brain_pending: toNumberOr(queues.brain_pending, 0),
			brain_running: toNumberOr(queues.brain_running, 0),
			brain_stale_pending: toNumberOr(queues.brain_stale_pending, 0),
			brain_stale_running: toNumberOr(queues.brain_stale_running, 0),
		},
		overdueJobs: toNumberOr(details.overdue_due_scheduler_jobs, 0),
		overdueJobIds: asArray(details.overdue_due_scheduler_job_ids)
			.map((id) => String(id ?? ''))
			.filter(Boolean),
		longRunningJobs: toNumberOr(details.long_running_scheduler_jobs, 0),
		runtimeOwner: String(details.runtime_owner ?? '').trim(),
	};
}

// ============================================================================
// System alerts feed (GET /api/logs, filtered to error/warning levels)
// ============================================================================

export interface SystemAlertEntry {
	id: string;
	level: 'error' | 'warning' | string;
	source: string;
	message: string;
	createdAt: string;
	/** Structured context the emitter attached (parsed from the data column). */
	details: Record<string, unknown> | null;
}

const ALERT_LEVELS = new Set(['error', 'warning', 'critical']);

export async function getRecentSystemAlerts(limit = 60): Promise<SystemAlertEntry[]> {
	// /api/logs has no level filter; over-fetch and filter client-side so
	// heartbeats/info rows don't crowd out the alerts the operator must see.
	const rows = asArray(await fetchApi<unknown>('/logs?limit=500'));
	const alerts: SystemAlertEntry[] = [];
	for (const row of rows) {
		const rec = asRecord(row);
		if (!rec) continue;
		const level = String(rec.level ?? '').trim().toLowerCase();
		if (!ALERT_LEVELS.has(level)) continue;
		alerts.push({
			id: String(rec.id ?? ''),
			level,
			source: String(rec.source ?? '').trim(),
			message: String(rec.message ?? '').trim(),
			createdAt: String(rec.created_at ?? '').trim(),
			details: parseAlertDetails(rec.data),
		});
		if (alerts.length >= limit) break;
	}
	return alerts;
}

function parseAlertDetails(raw: unknown): Record<string, unknown> | null {
	if (raw && typeof raw === 'object' && !Array.isArray(raw)) return raw as Record<string, unknown>;
	if (typeof raw !== 'string' || !raw.trim()) return null;
	try {
		const parsed = JSON.parse(raw);
		return parsed && typeof parsed === 'object' && !Array.isArray(parsed)
			? (parsed as Record<string, unknown>)
			: null;
	} catch {
		return null;
	}
}

// ============================================================================
// Lifecycle transitions feed (GET /api/lifecycle/events)
// ============================================================================

export interface LifecycleEventSummary {
	id: string;
	strategyId: string;
	fromState: string;
	toState: string;
	actor: string;
	reason: string;
	createdAt: string;
}

export async function getRecentLifecycleEvents(limit = 30): Promise<LifecycleEventSummary[]> {
	const rows = asArray(await fetchApi<unknown>(`/lifecycle/events?limit=${limit}`));
	const events: LifecycleEventSummary[] = [];
	for (const row of rows) {
		const rec = asRecord(row);
		if (!rec) continue;
		events.push({
			id: String(rec.id ?? ''),
			strategyId: String(rec.strategy_id ?? '').trim(),
			fromState: String(rec.from_state ?? '').trim(),
			toState: String(rec.to_state ?? '').trim(),
			actor: String(rec.actor ?? '').trim(),
			reason: String(rec.reason ?? '').trim(),
			createdAt: String(rec.created_at ?? '').trim(),
		});
	}
	return events;
}

// ============================================================================
// Scheduler watch (GET /api/scheduler)
// ============================================================================

export interface SchedulerJobSummary {
	id: string;
	name: string;
	enabled: boolean;
	lastRunAt: string | null;
	nextRunAt: string | null;
	runningSince: string | null;
	lastStatus: string;
	lastError: string | null;
}

export async function getSchedulerJobs(): Promise<SchedulerJobSummary[]> {
	const rows = asArray(await fetchApi<unknown>('/scheduler'));
	const jobs: SchedulerJobSummary[] = [];
	for (const row of rows) {
		const rec = asRecord(row);
		if (!rec) continue;
		jobs.push({
			id: String(rec.id ?? ''),
			name: String(rec.name ?? rec.id ?? '').trim(),
			enabled: Boolean(Number(rec.enabled ?? 0)),
			lastRunAt: rec.last_run_at ? String(rec.last_run_at) : null,
			nextRunAt: rec.next_run_at ? String(rec.next_run_at) : null,
			runningSince: rec.running_since ? String(rec.running_since) : null,
			lastStatus: String(rec.last_status ?? '').trim().toLowerCase(),
			lastError: rec.last_error ? String(rec.last_error) : null,
		});
	}
	return jobs;
}

// ============================================================================
// Critical health alerts (GET /api/health/alerts)
// ============================================================================

export interface HealthAlertItem {
	severity: string;
	component: string;
	message: string;
	timestamp: string;
	action_taken: string;
}

export interface HealthAlertsResponse {
	alerts: HealthAlertItem[];
	count: number;
}

export async function getCriticalHealthAlerts(limit = 25): Promise<HealthAlertsResponse> {
	const payload = asRecord(
		await fetchApi<unknown>(`/health/alerts?severity=critical&limit=${limit}`)
	) ?? {};
	const alerts = asArray(payload.alerts)
		.map((row) => {
			const rec = asRecord(row);
			if (!rec) return null;
			return {
				severity: String(rec.severity ?? 'critical').trim() || 'critical',
				component: String(rec.component ?? '').trim(),
				message: String(rec.message ?? '').trim(),
				timestamp: String(rec.timestamp ?? '').trim(),
				action_taken: String(rec.action_taken ?? '').trim(),
			};
		})
		.filter((row): row is HealthAlertItem => Boolean(row));
	return { alerts, count: toNumberOr(payload.count, alerts.length) };
}

// ============================================================================
// Paper session PnL rollup (GET /api/paper/summary)
// ============================================================================

export interface PaperSessionSummaryRow {
	session_id: string;
	strategy_id: string;
	strategy_name: string;
	symbol: string;
	timeframe: string;
	status: string;
	closed_count: number;
	open_count: number;
	realized_pnl_usd: number;
	win_rate_pct: number | null;
	close_reasons: Record<string, number>;
}

export interface PaperSummaryTotals {
	session_count: number;
	closed_count: number;
	open_count: number;
	realized_pnl_usd: number;
	win_rate_pct: number | null;
	close_reasons: Record<string, number>;
}

export interface PaperSummary {
	sessions: PaperSessionSummaryRow[];
	totals: PaperSummaryTotals;
	include_deployed: boolean;
	timestamp: string;
}

function normalizeCloseReasons(value: unknown): Record<string, number> {
	const rec = asRecord(value) ?? {};
	const out: Record<string, number> = {};
	for (const [reason, count] of Object.entries(rec)) {
		const key = String(reason ?? '').trim();
		if (!key) continue;
		out[key] = toNumberOr(count, 0);
	}
	return out;
}

function normalizeWinRate(value: unknown): number | null {
	if (value === null || value === undefined) return null;
	const parsed = Number(value);
	return Number.isFinite(parsed) ? parsed : null;
}

export async function getPaperSummary(includeDeployed = false): Promise<PaperSummary> {
	const payload = asRecord(
		await fetchApi<unknown>(`/paper/summary?include_deployed=${includeDeployed}`)
	) ?? {};
	const totalsRec = asRecord(payload.totals) ?? {};
	const sessions = asArray(payload.sessions)
		.map((row) => {
			const rec = asRecord(row);
			if (!rec) return null;
			return {
				session_id: String(rec.session_id ?? '').trim(),
				strategy_id: String(rec.strategy_id ?? '').trim(),
				strategy_name: String(rec.strategy_name ?? '').trim(),
				symbol: String(rec.symbol ?? '').trim(),
				timeframe: String(rec.timeframe ?? '').trim(),
				status: String(rec.status ?? '').trim(),
				closed_count: toNumberOr(rec.closed_count, 0),
				open_count: toNumberOr(rec.open_count, 0),
				realized_pnl_usd: toNumberOr(rec.realized_pnl_usd, 0),
				win_rate_pct: normalizeWinRate(rec.win_rate_pct),
				close_reasons: normalizeCloseReasons(rec.close_reasons),
			};
		})
		.filter((row): row is PaperSessionSummaryRow => Boolean(row));
	return {
		sessions,
		totals: {
			session_count: toNumberOr(totalsRec.session_count, sessions.length),
			closed_count: toNumberOr(totalsRec.closed_count, 0),
			open_count: toNumberOr(totalsRec.open_count, 0),
			realized_pnl_usd: toNumberOr(totalsRec.realized_pnl_usd, 0),
			win_rate_pct: normalizeWinRate(totalsRec.win_rate_pct),
			close_reasons: normalizeCloseReasons(totalsRec.close_reasons),
		},
		include_deployed: Boolean(payload.include_deployed ?? includeDeployed),
		timestamp: String(payload.timestamp ?? '').trim(),
	};
}
