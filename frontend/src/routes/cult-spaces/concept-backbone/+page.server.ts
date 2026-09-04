import type { SharedSpaceStats } from '$lib/types';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ fetch }) => {
	const stats: SharedSpaceStats = await (await fetch('/data/shared-space-stats.json')).json();
	return { stats };
};
