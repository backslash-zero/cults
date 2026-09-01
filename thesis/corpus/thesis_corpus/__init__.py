"""PDF-to-clean-text extraction for the thesis corpus.

Scope is deliberately narrow (see clean_text.py's module docstring): finding
PDFs, checking they open, extracting text (native, falling back to OCR),
preserving page provenance, and logging outcomes. No chapter selection, no
Zotero/metadata linking, no NLP.
"""
