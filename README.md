# Info

Célestin Meunier\\
Design \& Computation, UdK/TU Berlin\\
Student ID: 4060246\\
c.meunier@udk-berlin.de or meunier@campus.tu-berlin.de

supervisors:
Prof. Dr. Dr. Daniel D Hromada
Prof. Dr. des. Marc Pfaff

## Main info

This repository contains the code and data for my master thesis in which I compare different epistemologies with conceptual spaces.

# Frontend

`./frontend` (SvelteKit) documents the progress of my thesis. The `/cult-spaces`
section queries `./studio-cults` live (a Sanity client + GROQ) to show research
progress and stats per corpus type, with content browsing for interviews.
Ideally it should also show the latest available version of the thesis as
typed in `./thesis`.

# Studio

`./studio-cults` is the Sanity Studio project that backs the frontend
(project `dm4p8gdv`, dataset `production`). Its four document types mirror the
four corpus types in `./thesis/corpus` — see below.

# Corpus

`./thesis/corpus` contains the unified research corpus: interviews,
literature, dictionary/thesaurus entries, and custom terms — file-based
JSON/YAML (plus, for literature, Zotero-exported `.bib` files) as the source
of truth, generated into both the thesis LaTeX appendices and (via
`thesis/corpus/scripts/sync_sanity.mjs`) the Sanity project above. It lives
inside `./thesis` since it exists to feed that document's appendices.
Naming conventions, the pipeline, and the update procedure are documented in
`./thesis/corpus/README.md`.

# Meditations

`./Meditations` contains transcripts of voice memos I recorded while thinking
through the project out loud — informal, unstructured reflections, sometimes
annotated with follow-up notes on how a point could feed into the thesis. Raw
thinking material, not corpus content or thesis text.

# Thesis-template

`./thesis-template` contains a template that needs to be used in order to write the thesis. It is based on the TU Berlin template for theses.

# Thesis

`./thesis` contains the thesis itself. It is based on the template in `./thesis-template`. The thesis is written in LaTeX and compiled with `latexmk`. The main file is `./thesis/thesis.tex`. The thesis is compiled into a PDF file called `thesis.pdf` in the same directory.

The raw interview batch material lives in `./thesis/Interviews/`; the processed
interview corpus (cleaned/translated text, metadata database, generated LaTeX
appendix, standalone analysis) is now part of the unified corpus at
`./thesis/corpus/interviews/` — see `./thesis/corpus/README.md`.

# Obsidian Vault

Contains my current note, and the latest plan for the thesis. The file is located at:
/Users/celestinmeunier/Documents/Obsidian Vault/Thesis 2026.md
