<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import { promoteForvenStrategy } from '$lib/api/forven';
	import { lifecycleStageLabel } from '$lib/utils/lifecyclePresentation';
	import { addToast } from '$lib/stores/processTracker';

	export let strategyId: string;
	/** Normalized current lifecycle stage (quick_screen | gauntlet | paper | live_graduated | archived | rejected | ...). */
	export let currentStage: string = '';
	/** Ordered pipeline stages; targets beyond these (archive/revive) are added automatically. */
	export let pipelineStages: ReadonlyArray<{ key: string; label: string }> = [];

	const dispatch = createEventDispatcher<{ changed: { toStage: string } }>();

	let open = false;
	let targetStage: string | null = null;
	let reason = '';
	let submitting = false;
	// Populated when the promotion gate rejects the transition; enables the
	// informed operator override path (mirrors the old Configuration-tab flow).
	let blockReason = '';

	const TERMINAL = new Set(['archived', 'rejected']);

	type StageOption = { key: string; label: string; kind: 'forward' | 'backward' | 'terminal' | 'revive' };

	$: currentIndex = pipelineStages.findIndex((stage) => stage.key === currentStage);
	$: stageOptions = buildOptions(currentStage, currentIndex, pipelineStages);
	$: targetOption = stageOptions.find((option) => option.key === targetStage) ?? null;
	$: isCapitalTarget = targetStage === 'paper' || targetStage === 'live_graduated';

	function buildOptions(
		stage: string,
		index: number,
		stages: ReadonlyArray<{ key: string; label: string }>,
	): StageOption[] {
		const options: StageOption[] = [];
		if (TERMINAL.has(stage)) {
			options.push({ key: 'quick_screen', label: 'Quick Screen (revive)', kind: 'revive' });
			return options;
		}
		stages.forEach((candidate, candidateIndex) => {
			if (candidate.key === stage) return;
			options.push({
				key: candidate.key,
				label: candidate.label,
				kind: candidateIndex > index ? 'forward' : 'backward',
			});
		});
		options.push({ key: 'archived', label: 'Archived', kind: 'terminal' });
		return options;
	}

	function stageTone(stage: string): string {
		switch (stage) {
			case 'live_graduated':
				return 'border-emerald-600/60 bg-emerald-950/40 text-emerald-200';
			case 'paper':
				return 'border-cyan-600/60 bg-cyan-950/40 text-cyan-200';
			case 'gauntlet':
				return 'border-violet-600/60 bg-violet-950/40 text-violet-200';
			case 'archived':
			case 'rejected':
				return 'border-red-900/60 bg-red-950/30 text-red-300';
			default:
				return 'border-[#2b2b2b] bg-black text-gray-300';
		}
	}

	function optionTone(kind: StageOption['kind']): string {
		switch (kind) {
			case 'forward':
				return 'text-cyan-200 hover:bg-cyan-950/40';
			case 'backward':
				return 'text-amber-200 hover:bg-amber-950/40';
			case 'terminal':
				return 'text-red-300 hover:bg-red-950/40';
			case 'revive':
				return 'text-emerald-200 hover:bg-emerald-950/40';
		}
	}

	function toggleOpen(): void {
		open = !open;
		if (!open) resetSelection();
	}

	function resetSelection(): void {
		targetStage = null;
		reason = '';
		blockReason = '';
	}

	function selectTarget(key: string): void {
		targetStage = key;
		reason = '';
		blockReason = '';
	}

	/** Open the panel pre-set to a target stage (used by "Run check" on the gauntlet card). */
	export function openFor(stage: string): void {
		open = true;
		selectTarget(stage);
	}

	async function confirm(override = false): Promise<void> {
		if (!targetStage || submitting) return;
		const target = targetStage;
		submitting = true;
		try {
			await promoteForvenStrategy(strategyId, target, {
				fromStatus: currentStage || undefined,
				reason:
					reason.trim() ||
					(override ? 'Operator gate override' : `Manual stage change from strategy container`),
				force: true,
				override,
			});
			addToast(`${strategyId} → ${lifecycleStageLabel(target)}`, 'success');
			open = false;
			resetSelection();
			dispatch('changed', { toStage: target });
		} catch (err) {
			const message = err instanceof Error ? err.message : 'Stage change failed';
			if (!override && isCapitalTarget) {
				// Promotion gate rejected a capital-stage move: surface the reason and
				// let the operator explicitly override instead of silently failing.
				blockReason = message;
			} else {
				addToast(message, 'error');
			}
		} finally {
			submitting = false;
		}
	}
</script>

<div class="relative flex items-center gap-1.5" data-testid="stage-control">
	<button
		type="button"
		data-testid="stage-control-toggle"
		class={`flex items-center gap-1.5 rounded border px-2 py-0.5 text-[11px] font-semibold uppercase tracking-[0.14em] transition hover:brightness-125 ${stageTone(currentStage)}`}
		title="Change lifecycle stage"
		aria-expanded={open}
		on:click={toggleOpen}
	>
		{lifecycleStageLabel(currentStage)}
		<svg class="h-3 w-3 opacity-70" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
			<path fill-rule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 10.94l3.71-3.71a.75.75 0 111.06 1.06l-4.24 4.24a.75.75 0 01-1.06 0L5.21 8.29a.75.75 0 01.02-1.08z" clip-rule="evenodd" />
		</svg>
	</button>

	{#if open}
		<div
			data-testid="stage-control-panel"
			class="absolute left-0 top-full z-30 mt-1.5 w-72 rounded-lg border border-[#2b2b2b] bg-[#0b0b0b] p-2 shadow-[0_18px_40px_rgba(0,0,0,0.6)]"
		>
			{#if !targetStage}
				<div class="px-1 pb-1.5 text-[10px] uppercase tracking-[0.18em] text-gray-500">Move to stage</div>
				<div class="space-y-0.5">
					{#each stageOptions as option (option.key)}
						<button
							type="button"
							data-testid={`stage-option-${option.key}`}
							class={`flex w-full items-center justify-between rounded px-2 py-1.5 text-left text-xs transition ${optionTone(option.kind)}`}
							on:click={() => selectTarget(option.key)}
						>
							<span>{option.label}</span>
							<span class="text-[9px] uppercase tracking-[0.14em] opacity-60">
								{option.kind === 'forward' ? 'Promote' : option.kind === 'backward' ? 'Demote' : option.kind === 'revive' ? 'Revive' : 'Archive'}
							</span>
						</button>
					{/each}
				</div>
			{:else}
				<div class="space-y-2 p-1">
					<div class="text-xs text-gray-200">
						Move <span class="font-mono text-cyan-300">{strategyId}</span> to
						<span class="font-semibold">{targetOption?.label ?? lifecycleStageLabel(targetStage)}</span>?
					</div>
					{#if targetStage === 'archived'}
						<div class="rounded border border-red-900/40 bg-red-950/15 px-2 py-1.5 text-[11px] text-red-200">
							Archiving removes the strategy from the active pipeline and stops scanning it.
						</div>
					{/if}
					<textarea
						bind:value={reason}
						data-testid="stage-control-reason"
						placeholder="Reason (optional)"
						rows="2"
						class="w-full rounded border border-[#2b2b2b] bg-black px-2 py-1 text-xs text-gray-300 placeholder:text-gray-600 focus:border-cyan-700 focus:outline-none"
					></textarea>
					{#if blockReason}
						<div class="rounded border border-amber-700/50 bg-amber-950/30 p-2 text-[11px]" data-testid="stage-control-block-reason">
							<div class="font-semibold text-amber-200">Promotion gate blocked this:</div>
							<div class="mt-0.5 text-amber-100/90">{blockReason}</div>
							<div class="mt-1 text-amber-300/70">Overriding promotes anyway (logged). The mainnet hard-gate is separate and unaffected.</div>
						</div>
					{/if}
					<div class="flex gap-1.5">
						{#if blockReason}
							<button
								type="button"
								data-testid="stage-control-override"
								disabled={submitting}
								class="rounded bg-amber-600 px-3 py-1 text-xs text-white transition hover:bg-amber-500 disabled:opacity-50"
								on:click={() => void confirm(true)}
							>{submitting ? 'Overriding…' : 'Override gate & promote'}</button>
						{:else}
							<button
								type="button"
								data-testid="stage-control-confirm"
								disabled={submitting}
								class="rounded bg-cyan-600 px-3 py-1 text-xs text-white transition hover:bg-cyan-500 disabled:opacity-50"
								on:click={() => void confirm(false)}
							>{submitting ? 'Moving…' : 'Confirm'}</button>
						{/if}
						<button
							type="button"
							class="rounded border border-[#2b2b2b] bg-black px-3 py-1 text-xs text-gray-400 transition hover:text-gray-200"
							on:click={resetSelection}
						>Back</button>
						<button
							type="button"
							class="ml-auto rounded border border-[#2b2b2b] bg-black px-3 py-1 text-xs text-gray-400 transition hover:text-gray-200"
							on:click={toggleOpen}
						>Cancel</button>
					</div>
				</div>
			{/if}
		</div>
	{/if}
</div>
