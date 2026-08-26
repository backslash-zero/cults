import { sanity } from '$lib/server/sanity';
import { interviewListQuery } from '$lib/server/queries';
import type { InterviewListItem } from '$lib/types';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async () => {
	const interviews = await sanity.fetch<InterviewListItem[]>(interviewListQuery);
	return { interviews };
};
