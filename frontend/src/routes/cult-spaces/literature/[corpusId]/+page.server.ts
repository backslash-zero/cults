import { error } from '@sveltejs/kit';
import { sanity } from '$lib/server/sanity';
import { literatureDetailQuery } from '$lib/server/queries';
import type { LiteratureItem } from '$lib/types';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ params }) => {
	const item = await sanity.fetch<LiteratureItem | null>(literatureDetailQuery, {
		corpusId: params.corpusId
	});

	if (!item) error(404, 'Literature item not found');

	return { item };
};
