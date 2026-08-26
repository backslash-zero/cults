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

`./frontend` contains the frontend code documenting the progress of my thesis.
Ideally it should show the latest available version of the thesis as typed in `./thesis`

# Studio

`./studio-cults` is the Sanity Studio project that should back the frontend
(project `dm4p8gdv`, dataset `production`). Schema types are being built out
alongside `./corpus` — see below.

# Corpus

`./corpus` contains the unified research corpus: interviews, literature,
dictionary/thesaurus entries, and custom terms — file-based JSON/YAML as the
source of truth, generated into both the thesis LaTeX appendices and (via
`corpus/scripts/sync_sanity.mjs`) the Sanity project above. Naming conventions,
the pipeline, and the update procedure are documented in `./corpus/README.md`.

# Papers

`./papers` contains pre-papers I made to help me conceptually understand what's going on.

# Thesis-template

`./thesis-template` contains a template that needs to be used in order to write the thesis. It is based on the TU Berlin template for theses.

# Thesis

`./thesis` contains the thesis itself. It is based on the template in `./thesis-template`. The thesis is written in LaTeX and compiled with `latexmk`. The main file is `./thesis/main.tex`. The thesis is compiled into a PDF file called `thesis.pdf` in the same directory.

The raw interview batch material lives in `./thesis/Interviews/`; the processed
interview corpus (cleaned/translated text, metadata database, generated LaTeX
appendix, standalone analysis) is now part of the unified corpus at
`./corpus/interviews/` — see `./corpus/README.md`.

# Obsidian Vault

Contains my current note, and the latest plan for the thesis. The file is located at:
/Users/celestinmeunier/Documents/Obsidian Vault/Thesis 2026.md
