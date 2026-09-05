export interface Interview {
	corpusId: string;
	batch?: string;
	dateTime?: string;
	dateTimePrecision?: string;
	method?: string;
	language?: string;
	translated?: boolean;
	translationLanguage?: string;
	location?: string;
	age?: string;
	gender?: string;
	nationality?: string;
	mainLanguage?: string;
	languageSpoken?: string;
	nQuestions?: number;
	nAnswers?: number;
	totalWordCount?: number;
	text?: string;
	translationText?: string;
}

export type InterviewListItem = Pick<
	Interview,
	'corpusId' | 'batch' | 'dateTime' | 'language' | 'method' | 'translated' | 'location' | 'totalWordCount'
>;

export type InterviewSummary = Pick<Interview, 'batch' | 'language' | 'method' | 'translated' | 'totalWordCount'>;

export interface InterviewAggregates {
	totalWordCount: number;
	avgWordCount: number;
	byBatch: Record<string, number>;
	byLanguage: Record<string, number>;
	byMethod: Record<string, number>;
	translatedCount: number;
	untranslatedCount: number;
}

export interface CorpusOverview {
	interviewCount: number;
	literatureCount: number;
	miviludesCriteriaCount: number;
	interviewStats: InterviewAggregates;
}

export interface LiteratureItem {
	corpusId: string;
	title: string;
	authors?: string[];
	year?: string;
	type?: 'article' | 'book_chapter' | 'book' | 'report' | 'other';
	source?: string;
	doi?: string;
	isbn?: string;
	language?: string;
	tags?: string[];
	rawFile?: string;
	dateAdded?: string;
	abstract?: string;
	notes?: string;
}

export interface DictionaryEntry {
	corpusId: string;
	term: string;
	source?: string;
	language?: string;
	citation?: string;
}

export interface CustomTerm {
	corpusId: string;
	term: string;
	definition?: string;
	relatedTerms?: string[];
}

export interface MiviludesCriterion {
	corpusId: string;
	criterionFr: string;
	criterionEn: string;
	order?: number;
	source?: string;
	citation?: string;
}

export interface MiviludesDocument {
	documentId: string;
	title: string;
	url: string;
	citation: string;
	itemsEmbedded: number;
}

export type SourceDataset =
	| 'literature'
	| 'miviludes'
	| 'interviews'
	| 'miviludes_criteria'
	| 'concept_backbone'
	| 'structural_concepts'
	| 'emergent_entities';

// Cuts across SourceDataset to group the six datasets into three kinds of
// point: 'expression' (literature/miviludes/interviews/miviludes_criteria,
// a claim a source makes), 'reference' (concept_backbone, an external,
// corpus-independent vocabulary), 'emergent' (emergent_entities, a named
// entity/concept mentioned by the corpora themselves).
export type PointRole = 'expression' | 'reference' | 'emergent';

export type ProjectionMethod = 'pca' | 'umap' | 'tsne';

export interface PointMeta {
	source_dataset: SourceDataset;
	point_role?: PointRole;
	key: string;
	label: string;
	label_en?: string;
	label_fr?: string;
	attribution?: string;
	claim_mode?: string;
	epistemic_status?: string;
	response_rank?: number;
}

export interface SharedSpaceStats {
	totalPoints: number;
	countsBySourceDataset: Record<string, number>;
	registry: {
		totalDocuments: number;
		byCorpus: Record<
			string,
			{
				documents: number;
				itemsEmbedded: number;
				byStage2Status: Record<string, number>;
			}
		>;
	};
	sharedSpace: {
		chosenDimensions: number;
		varianceAtK: number;
		varianceThreshold: number;
	};
}
