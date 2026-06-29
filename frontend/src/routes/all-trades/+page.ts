import type { PageLoad } from './$types';
import { getForvenAllTrades, getForvenTradesStats } from '$lib/api';
import type { ForvenTradesPage, ForvenTradesStats } from '$lib/api';

export const ssr = false;

export const load: PageLoad = async () => {
	const [pageResult, statsResult] = await Promise.allSettled([
		getForvenAllTrades({ limit: 100, sort: 'opened_at', sort_dir: 'desc' }),
		getForvenTradesStats()
	]);
	return {
		initialPage: pageResult.status === 'fulfilled' ? pageResult.value : null,
		initialStats: statsResult.status === 'fulfilled' ? statsResult.value : null
	} satisfies {
		initialPage: ForvenTradesPage | null;
		initialStats: ForvenTradesStats | null;
	};
};
