import type { PointRole, SourceDataset } from './types';

// Fixed categorical hue order (never cycled) from the validated default
// palette -- slots 1-6 (blue/orange/aqua/yellow/magenta/green), assigned in
// the order these datasets were introduced. 6 unordered categories exceed
// the all-pairs CVD-safe cap of 3 documented for scatter/point-cloud use,
// so identity is deliberately never carried by hue alone here: the legend
// always pairs each swatch with its category name, the detail panel always
// shows the selected point's source_dataset as text, and only 2 categories
// (literature, concept_backbone) are visible by default -- the rest are
// opt-in via the legend, keeping simultaneous on-screen comparisons small
// in practice.
export const SOURCE_DATASET_COLORS: Record<SourceDataset, { light: string; dark: string }> = {
	literature: { light: '#2a78d6', dark: '#3987e5' },
	// point_role: 'reference' -- an external, corpus-independent vocabulary
	// (WordNet), not derived from anything in the corpora.
	concept_backbone: { light: '#eb6834', dark: '#d95926' },
	interviews: { light: '#1baf7a', dark: '#199e70' },
	miviludes: { light: '#eda100', dark: '#c98500' },
	miviludes_criteria: { light: '#e87ba4', dark: '#d55181' },
	// point_role: 'reference', like concept_backbone above -- but
	// corpus-derived rather than topic-neutral (extracted from the
	// corpora's own expression text, not WordNet). See
	// SOURCE_DATASET_ROLES below for the shared point_role grouping.
	structural_concepts: { light: '#4a3aa7', dark: '#9085e9' },
	// point_role: 'emergent' -- named entities/concepts mentioned BY the
	// corpora themselves, as distinct from both reference subsets above
	// (all render at a fixed size/opacity per point_role in
	// EmbeddingExplorer.svelte, not by source_dataset).
	emergent_entities: { light: '#008300', dark: '#008300' }
};

export const SOURCE_DATASET_LABELS: Record<SourceDataset, string> = {
	literature: 'Literature',
	concept_backbone: 'Concept backbone',
	interviews: 'Interviews',
	miviludes: 'MIVILUDES',
	miviludes_criteria: 'MIVILUDES criteria',
	structural_concepts: 'Structural concepts',
	emergent_entities: 'Emergent entities'
};

// The static SourceDataset -> PointRole mapping the legend groups by (the
// per-point point_role field says the same thing per-point; this is the
// same fact at the category level, for grouping datasets that have no
// individual points to inspect yet).
export const SOURCE_DATASET_ROLES: Record<SourceDataset, PointRole> = {
	literature: 'expression',
	miviludes: 'expression',
	interviews: 'expression',
	miviludes_criteria: 'expression',
	concept_backbone: 'reference',
	structural_concepts: 'reference',
	emergent_entities: 'emergent'
};

export const DEFAULT_VISIBLE_DATASETS: SourceDataset[] = ['literature', 'concept_backbone'];

export const ALL_SOURCE_DATASETS = Object.keys(SOURCE_DATASET_COLORS) as SourceDataset[];

export function prefersDarkMode(): boolean {
	return typeof window !== 'undefined' && window.matchMedia('(prefers-color-scheme: dark)').matches;
}

export function colorForDataset(dataset: SourceDataset, dark: boolean): string {
	return dark ? SOURCE_DATASET_COLORS[dataset].dark : SOURCE_DATASET_COLORS[dataset].light;
}
