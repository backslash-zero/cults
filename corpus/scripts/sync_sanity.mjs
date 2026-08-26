#!/usr/bin/env node
/**
 * Sync the file-based corpus (corpus/metadata/*.json) into Sanity.
 *
 * The file-based JSON databases are the source of truth. This script upserts
 * each record into Sanity by looking it up on its `corpusId` field — it never
 * sets a document's `_id` directly, per Sanity's own guidance (document IDs
 * are an implementation detail; `corpusId` is the stable identity that maps
 * back to corpus/*_source.yaml and the thesis LaTeX \label{}s).
 *
 * Requires: @sanity/client (see corpus/package.json — run `pnpm install` in
 * corpus/ first) and a SANITY_API_TOKEN with write access.
 *
 * Usage: node corpus/scripts/sync_sanity.mjs
 */
import {createClient} from '@sanity/client'
import {readFileSync} from 'node:fs'
import {fileURLToPath} from 'node:url'
import path from 'node:path'

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url))
const CORPUS_DIR = path.resolve(SCRIPT_DIR, '..')
const METADATA_DIR = path.join(CORPUS_DIR, 'metadata')

const PROJECT_ID = process.env.SANITY_PROJECT_ID || 'dm4p8gdv'
const DATASET = process.env.SANITY_DATASET || 'production'
const API_VERSION = process.env.SANITY_API_VERSION || '2026-01-01'
const TOKEN = process.env.SANITY_API_TOKEN

if (!TOKEN) {
  console.error(
    'SANITY_API_TOKEN is not set. Create a write-access token at ' +
      'manage.sanity.io -> your project -> API -> Tokens, then ' +
      '`export SANITY_API_TOKEN=...` before running this script. See ' +
      'corpus/README.md for details. Aborting — nothing was synced.',
  )
  process.exit(1)
}

const client = createClient({
  projectId: PROJECT_ID,
  dataset: DATASET,
  apiVersion: API_VERSION,
  token: TOKEN,
  useCdn: false,
})

function readJson(file, key) {
  const full = path.join(METADATA_DIR, file)
  try {
    const data = JSON.parse(readFileSync(full, 'utf-8'))
    return data[key] || []
  } catch (err) {
    console.warn(`Could not read ${full}: ${err.message}`)
    return []
  }
}

async function upsert(sanityType, corpusId, fields) {
  const existing = await client.fetch(
    `*[_type == $type && corpusId == $corpusId][0]{_id}`,
    {type: sanityType, corpusId},
  )
  if (existing?._id) {
    await client.patch(existing._id).set(fields).commit()
    return 'updated'
  }
  await client.create({_type: sanityType, corpusId, ...fields})
  return 'created'
}

function mapInterview(r) {
  const iv = r.interviewee || {}
  return {
    batch: r.batch,
    dateTime: r.date_time,
    dateTimePrecision: r.date_time_precision || '',
    method: r.method,
    language: r.language,
    translated: !!r.translated,
    translationLanguage: r.translation_language || '',
    location: r.location || '',
    age: iv.age != null ? String(iv.age) : '',
    gender: iv.gender || '',
    nationality: iv.nationality || '',
    mainLanguage: iv.main_language || '',
    languageSpoken: iv.language_spoken || '',
    nQuestions: r.n_questions,
    nAnswers: r.n_answers,
    totalWordCount: r.total_word_count,
    text: r.text,
    translationText: r.translation_text || '',
  }
}

function mapLiterature(r) {
  return {
    title: r.title,
    authors: r.authors || [],
    year: r.year != null ? String(r.year) : '',
    type: r.type || '',
    source: r.source || '',
    language: r.language || '',
    tags: r.tags || [],
    rawFile: r.raw_file || '',
    dateAdded: r.date_added || undefined,
    notes: r.notes || '',
  }
}

function mapDictionary(r) {
  return {
    source: r.source || '',
    term: r.term,
    language: r.language || '',
    definition: r.definition || '',
    citation: r.citation || '',
    dateAdded: r.date_added || undefined,
  }
}

function mapCustomTerm(r) {
  return {
    term: r.term,
    definition: r.definition || '',
    relatedTerms: r.related_terms || [],
    dateAdded: r.date_added || undefined,
    notes: r.notes || '',
  }
}

const SOURCES = [
  // interviews' database.json lives under corpus/interviews/metadata/, not
  // the shared corpus/metadata/ used by the three newer types.
  {path: path.join(CORPUS_DIR, 'interviews', 'metadata', 'database.json'),
   key: 'interviews', sanityType: 'interview', map: mapInterview},
  {path: path.join(METADATA_DIR, 'literature.json'),
   key: 'literature', sanityType: 'literatureItem', map: mapLiterature},
  {path: path.join(METADATA_DIR, 'dictionaries.json'),
   key: 'dictionaries', sanityType: 'dictionaryEntry', map: mapDictionary},
  {path: path.join(METADATA_DIR, 'custom_terms.json'),
   key: 'custom_terms', sanityType: 'customTerm', map: mapCustomTerm},
]

async function main() {
  for (const {path: full, key, sanityType, map} of SOURCES) {
    let records
    try {
      records = JSON.parse(readFileSync(full, 'utf-8'))[key] || []
    } catch (err) {
      console.warn(`Skipping ${sanityType}: could not read ${full} (${err.message})`)
      continue
    }
    if (records.length === 0) {
      console.log(`${sanityType}: 0 records, nothing to sync.`)
      continue
    }
    let created = 0
    let updated = 0
    for (const r of records) {
      const result = await upsert(sanityType, r.id, map(r))
      if (result === 'created') created++
      else updated++
    }
    console.log(`${sanityType}: ${created} created, ${updated} updated (of ${records.length}).`)
  }
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
