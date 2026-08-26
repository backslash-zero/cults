import { sanity } from '$lib/server/sanity';
import { dictionaryListQuery } from '$lib/server/queries';
import type { DictionaryEntry } from '$lib/types';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async () => {
	const items = await sanity.fetch<DictionaryEntry[]>(dictionaryListQuery);
	return { items };
};
