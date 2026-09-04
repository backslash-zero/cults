import type { SourceDataset } from './types';

// Fixed categorical hue order (never cycled) from the validated default
// palette. 6 unordered categories exceed the all-pairs CVD-safe cap of 3
// documented for scatter/point-cloud use, so identity is deliberately never
// carried by hue alone here: the legend always pairs each swatch with its
// category name, the detail panel always shows the selected point's
// source_dataset as text, and only 2 categories (literature,
// concept_backbone) are visible by default -- the rest are opt-in via the
// legend, keeping simultaneous on-screen comparisons small in practice.
export const SOURCE_DATASET_COLORS: Record<SourceDataset, { light: string; dark: string }> = {
	literature: { light: '#2a78d6', dark: '#3987e5' },
	concept_backbone: { light: '#eb6834', dark: '#d95926' },
	interviews: { light: '#1baf7a', dark: '#199e70' },
	miviludes: { light: '#eda100', dark: '#c98500' },
	miviludes_criteria_fr: { light: '#e87ba4', dark: '#d55181' },
	miviludes_criteria_en: { light: '#008300', dark: '#008300' }
};

export const SOURCE_DATASET_LABELS: Record<SourceDataset, string> = {
	literature: 'Literature',
	concept_backbone: 'Concept backbone',
	interviews: 'Interviews',
	miviludes: 'MIVILUDES',
	miviludes_criteria_fr: 'MIVILUDES criteria (FR)',
	miviludes_criteria_en: 'MIVILUDES criteria (EN)'
};

export const DEFAULT_VISIBLE_DATASETS: SourceDataset[] = ['literature', 'concept_backbone'];

export const ALL_SOURCE_DATASETS = Object.keys(SOURCE_DATASET_COLORS) as SourceDataset[];

export function prefersDarkMode(): boolean {
	return typeof window !== 'undefined' && window.matchMedia('(prefers-color-scheme: dark)').matches;
}

export function colorForDataset(dataset: SourceDataset, dark: boolean): string {
	return dark ? SOURCE_DATASET_COLORS[dataset].dark : SOURCE_DATASET_COLORS[dataset].light;
}
