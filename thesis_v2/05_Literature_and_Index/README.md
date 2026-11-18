# Bibliography.bib

`Bibliography.bib` is a **live Zotero Better BibTeX auto-export**, not a
hand-maintained file. Zotero rewrites it automatically whenever the
underlying Zotero library changes, with no explicit export step and no
warning. This is the file `thesis.tex` actually cites from
(`\addbibresource{05_Literature_and_Index/Bibliography.bib}`) — it is
distinct from the `.bib` files under `thesis/corpus/literature/raw/`, which
feed the separate corpus/appendix pipeline documented in
`thesis/corpus/README.md` and are not live-synced.

**Any manual edit made directly in this file will eventually be silently
overwritten** the next time Zotero re-exports (e.g. after any edit to any
item in the library, not just this one). This includes edits made by an
assistant/agent working on the thesis, not just by hand.

If a citation needs a metadata fix (missing `author`/`organization` field,
wrong ISBN, a `shorttitle`, etc.), make the fix **in Zotero itself**, on the
corresponding item — not in this file — so it survives the next
auto-export. Treat this file as a generated/derived artefact: fine to read,
grep, or diff against, not safe to edit and rely on.
