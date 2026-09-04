import { sanity } from '$lib/server/sanity';
import { overviewQuery } from '$lib/server/queries';
import { aggregateInterviews } from '$lib/server/stats';
import type { CorpusOverview, InterviewSummary, MiviludesDocument, SharedSpaceStats } from '$lib/types';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ fetch }) => {
	const raw = await sanity.fetch<{
		interviewCount: number;
		literatureCount: number;
		miviludesCriteriaCount: number;
		interviews: InterviewSummary[];
	}>(overviewQuery);

	const overview: CorpusOverview = {
		interviewCount: raw.interviewCount,
		literatureCount: raw.literatureCount,
		miviludesCriteriaCount: raw.miviludesCriteriaCount,
		interviewStats: aggregateInterviews(raw.interviews)
	};

	const stats: SharedSpaceStats = await (await fetch('/data/shared-space-stats.json')).json();
	const miviludesDocuments: MiviludesDocument[] = await (await fetch('/data/miviludes-documents.json')).json();

	return { overview, stats, miviludesDocumentCount: miviludesDocuments.length };
};
