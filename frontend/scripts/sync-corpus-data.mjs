#!/usr/bin/env node
// This script reshapes the corpus pipeline's shared-space output for
// efficient browser consumption. It is run manually whenever the pipeline
// output changes; its output is committed to git. No Sanity token or live
// API dependency is required.
//
// Reads (all git-tracked, from ../../thesis/corpus/processed relative to
// this file):
//   registry.csv
//   shared_space/variance_curve.json
//   shared_space/visualization_{pca,umap,tsne}_3d.jsonl
//
// Writes, under frontend/static/data/:
//   points-meta.json       -- [{source_dataset, point_role, key, label, label_en, attribution,
//                               claim_mode, epistemic_status, response_rank}, ...]
//                              (46,648 entries, written once). point_role groups the six
//                              source_dataset values into three kinds of point: 'expression'
//                              (literature/miviludes/interviews/miviludes_criteria), 'reference'
//                              (concept_backbone), 'emergent' (emergent_entities -- named entities
//                              the corpora themselves surface, formerly called entity_anchors).
//   positions-pca.json     -- [[x, y, z], ...] order-aligned to points-meta.json
//   positions-umap.json
//   positions-tsne.json
//   shared-space-stats.json    -- counts + registry + variance summary for the Overview page
//   miviludes-documents.json   -- the 2 MIVILUDES source documents (title/url/citation from
//                                 their .bib files, item counts live from registry.csv)
//
// The three visualization_*_3d.jsonl files are structurally row-aligned:
// visualize_3d.py builds one in-memory points list and writes it three
// times with different coordinate arrays, never re-sorting or re-filtering
// in between -- so splitting metadata (shared once) from positions (three
// small arrays) is safe and roughly halves the total payload versus
// shipping all three raw files verbatim.
//
// Usage (from frontend/): node scripts/sync-corpus-data.mjs

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const FRONTEND_DIR = path.resolve(SCRIPT_DIR, '..');
const CORPUS_PROCESSED_DIR = path.resolve(FRONTEND_DIR, '../thesis/corpus/processed');
const SHARED_SPACE_DIR = path.join(CORPUS_PROCESSED_DIR, 'shared_space');

const REGISTRY_PATH = path.join(CORPUS_PROCESSED_DIR, 'registry.csv');
const VARIANCE_CURVE_PATH = path.join(SHARED_SPACE_DIR, 'variance_curve.json');
const METHODS = ['pca', 'umap', 'tsne'];

const OUTPUT_DIR = path.join(FRONTEND_DIR, 'static', 'data');

function parseRegistry(csvText) {
	const lines = csvText.trim().split('\n');
	const header = lines[0].split(',');
	return lines.slice(1).map((line) => {
		const fields = line.split(',');
		return Object.fromEntries(header.map((key, i) => [key, fields[i]]));
	});
}

// The 2 MIVILUDES source documents don't warrant a Sanity type of their
// own -- this hardcoded map (title/url/citation from their .bib files:
// thesis/corpus/MIVILUDES/Latest Report/Latest Report/Latest Report.bib and
// thesis/corpus/MIVILUDES/Sectarian Drifts/CommentIdentifierDerive.bib) is
// joined below with registry.csv's live item_count, so chunk counts stay
// accurate if the pipeline reprocesses without needing a .bib parser for
// just 2 records.
const MIVILUDES_DOCUMENTS = {
	'mission-interministe-rielle-de-vigilance-et-de-lutte-contre-les-de-rives-sectair': {
		title: "Rapport d'activité de la Miviludes 2022-2024",
		url: 'https://www.miviludes.interieur.gouv.fr/nos-ressources/nos-publications/rapports-dactivite-de-miviludes',
		citation: 'missioninterministerielledevigilanceetdeluttecontrelesderivessectairesmiviludesRapportDactiviteMiviludes2024',
	},
	'raw-transcript': {
		title: 'Comment identifier une dérive sectaire ? | Miviludes',
		url: 'https://www.miviludes.interieur.gouv.fr/comprendre-prevenir-et-lutter/comment-identifier-derive-sectaire',
		citation: 'CommentIdentifierDerive',
	},
};

function buildMiviludesDocuments(registryRows) {
	return registryRows
		.filter((row) => row.corpus === 'miviludes')
		.map((row) => {
			const known = MIVILUDES_DOCUMENTS[row.document_id];
			if (!known) {
				throw new Error(`No known title/url/citation for MIVILUDES document_id "${row.document_id}" -- update MIVILUDES_DOCUMENTS`);
			}
			return {
				documentId: row.document_id,
				title: known.title,
				url: known.url,
				citation: known.citation,
				itemsEmbedded: Number(row.item_count) || 0,
			};
		});
}

function summarizeRegistry(rows) {
	const byCorpus = {};
	for (const row of rows) {
		const corpus = byCorpus[row.corpus] ?? (byCorpus[row.corpus] = {
			documents: 0,
			itemsEmbedded: 0,
			byStage2Status: {},
		});
		corpus.documents += 1;
		corpus.itemsEmbedded += Number(row.item_count) || 0;
		corpus.byStage2Status[row.stage2_status] = (corpus.byStage2Status[row.stage2_status] ?? 0) + 1;
	}
	return { totalDocuments: rows.length, byCorpus };
}

function loadJsonlLines(filePath) {
	const text = readFileSync(filePath, 'utf-8');
	return text
		.split('\n')
		.map((line) => line.trim())
		.filter((line) => line.length > 0)
		.map((line) => JSON.parse(line));
}

function main() {
	console.log(`Reading registry: ${REGISTRY_PATH}`);
	const registryRows = parseRegistry(readFileSync(REGISTRY_PATH, 'utf-8'));
	const registrySummary = summarizeRegistry(registryRows);

	console.log(`Reading variance curve: ${VARIANCE_CURVE_PATH}`);
	const varianceCurve = JSON.parse(readFileSync(VARIANCE_CURVE_PATH, 'utf-8'));

	mkdirSync(OUTPUT_DIR, { recursive: true });

	let pointsMeta = null;
	const countsBySourceDataset = {};

	for (const method of METHODS) {
		const inputPath = path.join(SHARED_SPACE_DIR, `visualization_${method}_3d.jsonl`);
		console.log(`Reading ${inputPath} ...`);
		const rows = loadJsonlLines(inputPath);

		if (pointsMeta === null) {
			pointsMeta = rows.map((r) => ({
				source_dataset: r.source_dataset,
				point_role: r.point_role ?? undefined,
				key: r.key,
				label: r.label,
				label_en: r.label_en ?? undefined,
				attribution: r.attribution ?? undefined,
				claim_mode: r.claim_mode ?? undefined,
				epistemic_status: r.epistemic_status ?? undefined,
				response_rank: r.response_rank ?? undefined
			}));
			for (const r of pointsMeta) {
				countsBySourceDataset[r.source_dataset] = (countsBySourceDataset[r.source_dataset] ?? 0) + 1;
			}
		} else if (rows.length !== pointsMeta.length) {
			throw new Error(
				`${method} has ${rows.length} rows but a prior method had ${pointsMeta.length} -- files are not row-aligned`,
			);
		}

		const vectorField = `${method}_3d_vector`;
		const positions = rows.map((r) => r[vectorField]);
		const positionsPath = path.join(OUTPUT_DIR, `positions-${method}.json`);
		writeFileSync(positionsPath, JSON.stringify(positions));
		console.log(`Wrote ${positionsPath} (${positions.length} points)`);
	}

	const pointsMetaPath = path.join(OUTPUT_DIR, 'points-meta.json');
	writeFileSync(pointsMetaPath, JSON.stringify(pointsMeta));
	console.log(`Wrote ${pointsMetaPath} (${pointsMeta.length} points)`);

	const stats = {
		totalPoints: pointsMeta.length,
		countsBySourceDataset,
		registry: registrySummary,
		sharedSpace: {
			chosenDimensions: varianceCurve.chosen_k,
			varianceAtK: varianceCurve.variance_at_k,
			varianceThreshold: varianceCurve.threshold,
		},
	};
	const statsPath = path.join(OUTPUT_DIR, 'shared-space-stats.json');
	writeFileSync(statsPath, JSON.stringify(stats, null, 2));
	console.log(`Wrote ${statsPath}`);

	const miviludesDocuments = buildMiviludesDocuments(registryRows);
	const miviludesDocumentsPath = path.join(OUTPUT_DIR, 'miviludes-documents.json');
	writeFileSync(miviludesDocumentsPath, JSON.stringify(miviludesDocuments, null, 2));
	console.log(`Wrote ${miviludesDocumentsPath} (${miviludesDocuments.length} documents)`);

	console.log(`\nDone. ${pointsMeta.length} points across ${METHODS.length} projection methods.`);
}

main();
