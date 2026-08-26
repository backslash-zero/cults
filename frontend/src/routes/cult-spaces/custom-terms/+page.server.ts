import { sanity } from '$lib/server/sanity';
import { customTermListQuery } from '$lib/server/queries';
import type { CustomTerm } from '$lib/types';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async () => {
	const items = await sanity.fetch<CustomTerm[]>(customTermListQuery);
	return { items };
};
