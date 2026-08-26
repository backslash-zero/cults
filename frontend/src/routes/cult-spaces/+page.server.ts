import { sanity } from '$lib/server/sanity';
import { overviewQuery } from '$lib/server/queries';
import { aggregateInterviews } from '$lib/server/stats';
import type { CorpusOverview, InterviewSummary } from '$lib/types';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async () => {
	const raw = await sanity.fetch<{
		interviewCount: number;
		literatureCount: number;
		dictionaryCount: number;
		customTermCount: number;
		interviews: InterviewSummary[];
	}>(overviewQuery);

	const overview: CorpusOverview = {
		interviewCount: raw.interviewCount,
		literatureCount: raw.literatureCount,
		dictionaryCount: raw.dictionaryCount,
		customTermCount: raw.customTermCount,
		interviewStats: aggregateInterviews(raw.interviews)
	};

	return { overview };
};
