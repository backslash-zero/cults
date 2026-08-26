import { error } from '@sveltejs/kit';
import { sanity } from '$lib/server/sanity';
import { interviewDetailQuery } from '$lib/server/queries';
import type { Interview } from '$lib/types';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ params }) => {
	const interview = await sanity.fetch<Interview | null>(interviewDetailQuery, {
		corpusId: params.corpusId
	});

	if (!interview) error(404, 'Interview not found');

	return { interview };
};
