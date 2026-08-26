import { sanity } from '$lib/server/sanity';
import { literatureListQuery } from '$lib/server/queries';
import type { LiteratureItem } from '$lib/types';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async () => {
	const items = await sanity.fetch<LiteratureItem[]>(literatureListQuery);
	return { items };
};
