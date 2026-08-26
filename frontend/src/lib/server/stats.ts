import type { InterviewAggregates, InterviewSummary } from '$lib/types';

export function aggregateInterviews(rows: InterviewSummary[]): InterviewAggregates {
	const totalWordCount = rows.reduce((sum, r) => sum + (r.totalWordCount ?? 0), 0);

	const countBy = (field: 'batch' | 'language' | 'method') =>
		rows.reduce<Record<string, number>>((acc, r) => {
			const key = r[field] ?? 'Unknown';
			acc[key] = (acc[key] ?? 0) + 1;
			return acc;
		}, {});

	return {
		totalWordCount,
		avgWordCount: rows.length ? Math.round(totalWordCount / rows.length) : 0,
		byBatch: countBy('batch'),
		byLanguage: countBy('language'),
		byMethod: countBy('method'),
		translatedCount: rows.filter((r) => r.translated).length,
		untranslatedCount: rows.filter((r) => !r.translated).length
	};
}
