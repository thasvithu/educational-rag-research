# Step 2: Clean the educational PDF corpus

This pipeline creates one consistent text corpus for the five chunking experiments.
It processes every member's PDFs, preserves educational evidence, and records what
happened to every page. It does not chunk documents, create embeddings, or call an LLM.

## Run

From the project root, using the existing Python 3.11 virtual environment:

```bash
uv pip install -r requirements.txt
.venv/bin/python 02_data_cleaning/clean_pdfs.py
.venv/bin/python 02_data_cleaning/prepare_texts.py
```

On Windows, use `.venv\Scripts\python.exe` instead of `.venv/bin/python`.
The cleaner also requires **Poppler's `pdftotext` executable** on `PATH`. It is already
available in the research lead's environment. On Ubuntu/Debian it is provided by
`poppler-utils`; on macOS by Homebrew's `poppler` package. Windows users need a Poppler
installation with its executable directory on `PATH`. Check with `pdftotext -v`.

The Python PDF extractors are pinned in `requirements.txt`. The exact Python,
PyMuPDF, PyMuPDF4LLM, Poppler and pipeline versions are recorded in each manifest.
The current `pyproject.toml` has an empty dependency list; use the requirements file
to install dependencies. If using `uv run`, use `uv run --no-sync` so that this
requirements-managed environment is not synchronized against that empty list.

```bash
# Run all members with three independent document workers.
.venv/bin/python 02_data_cleaning/clean_pdfs.py --workers 3

# Run one member into a separate output directory.
.venv/bin/python 02_data_cleaning/clean_pdfs.py --members vithusan --output-dir final_data_vithusan

# Check the cleaning rules and an actual generated PDF end to end.
.venv/bin/python -m unittest discover -s 02_data_cleaning/tests -v

# Check the generated corpus, source hashes and page offsets.
.venv/bin/python 02_data_cleaning/validate_corpus.py
```

By default, input is `data/` and output is **`final_data/`**. Paths are resolved
relative to the script's project root, so the default command also works from another
working directory. Immediate subdirectories of `data/` are treated as members,
excluding `outputs` and hidden directories. PDFs are found recursively, including
files with a PDF header but no `.pdf` extension. Symlinked input files are skipped.

Use separate output directories for subset runs: each run replaces the manifest and
corpus for that selected input. Individual output files are written atomically.
Older per-document artifacts are retained, but only the current manifest/corpus
defines the active dataset. A run with any failed PDF exits with status 1 and lists
the failures; they are never counted as successfully cleaned.

## What gets produced

```text
final_data/
├── README.md
├── manifest.json
├── corpus.jsonl
├── review_queue.csv
├── aysha/
│   ├── handbook__<source-hash>.txt
│   └── handbook__<source-hash>.json
├── haleema/
└── vithusan/
```

| File | Purpose |
|---|---|
| Per-PDF `.txt` | Prepared UTF-8 plain text after `prepare_texts.py`; table separators and fixed-width layout spacing are retained. |
| Per-PDF `.json` | Original extracted text, physical lines and bounding boxes, inferred heading candidates, removed margin lines, page labels, clean text, character spans, conversion attempts and quality flags. |
| `corpus.jsonl` | One **complete document** per line, with `text`, source identity, page spans and a `requires_review` signal. This is the main experiment input. |
| `manifest.json` | Input/output hashes, configuration, tool versions, document status, counts and duplicate decisions. |
| `review_queue.csv` | Page-level review signals, including image-only pages and difficult tables. |
| `preparation_report.json` | Per-document formatting changes, excluded corrupt pages and original-output backup location. |

Generated data is excluded from Git, matching the repository's existing treatment
of source PDFs and exploration reports. Python code, documentation and tests can be
shared. Hash suffixes avoid overwriting similarly named source versions.

### Loading the same text for all chunkers

```python
import json
from pathlib import Path

with Path("final_data/corpus.jsonl").open(encoding="utf-8") as stream:
    for line in stream:
        document = json.loads(line)
        text = document["text"]
        # Pass this SAME complete text to each chunking strategy.
        # Keep document_id and source with every resulting chunk.
        # Use document["pages"] to map chunk character spans to PDF pages.
```

Page records are **provenance, not precomputed chunks**. Do not force fixed-size,
semantic or late chunking to restart at every PDF page. A page boundary can occur
inside a paragraph. `char_start` is inclusive and `char_end` is exclusive, measured
in Python Unicode characters. PDF page numbers are one-based physical pages; a
printed page label is separately retained when the PDF provides it.

There are no artificial `Page 1`/filename headings injected into the indexed text.
Heading levels generated from font sizes are heuristics; `heading_candidates` in
the detailed JSON explicitly marks them as inferred. The PDF's real bookmark TOC
is retained separately. For structure-aware chunking, inspect this metadata alongside
the extracted headings, including on pages that needed layout fallback.

## Cleaning policy

### Final formatting pass

After extraction, run `prepare_texts.py` to read every TXT and companion JSON,
remove artificial page fences, Markdown emphasis and heading prefixes, convert
HTML breaks, remove table alignment markup, and remove remaining page numbers
only when bottom-margin geometry and a repeated physical-to-printed page offset
support the decision. Genuine footnote stars, code literals, quotes, mathematical
symbols, repeated credits, table pipes and column spacing remain.
Pages whose entire extracted content is a single bottom-margin number are also
cleared; their page record and raw extraction remain available in metadata.

This pass backs up the entire output in `data/outputs/`, prepares a separate staging
directory, and validates it before replacing the generated files. Unexpected manual
text changes stop the run; the existing UCSC wrapper-only edit is reconciled safely.
Repeating the unchanged script validates the prepared output without processing it twice.
After changing preparation rules, start from the extraction backup or rerun extraction.

TXT, detailed JSON and JSONL contain identical prepared text, and all page offsets,
hashes, review previews and corpus counts are rebuilt. Original extraction provenance
and raw text are preserved. The old `preservation` score describes extraction before
preparation; it is **not an accuracy score for the final text**.

For structure-aware chunking, use `pages[].headings` (page-local character offsets;
inferred levels may be null), the original `heading_candidates`, and PDF bookmarks.
Add a page's `char_start` to a heading's local offset to obtain its document offset.
Plain TXT no longer supplies generated `#` markers to a Markdown-header splitter.

The current corpus has a source-hash-specific exclusion for 44 corrupt-font pages
of `aysha/UOC_FMF_HB_2018_2022.pdf`: physical pages 9–56 except 18, 19, 25 and 26.
Their extraction is unreadable encoded text. Their `clean_text` is empty and
`include_in_chunking` is false; `excluded_text` and `raw_text` retain the evidence
for later re-extraction. Readable portions of that document remain. This is a
documented corpus exclusion, not an OCR repair or an English-language heuristic.
Do not use excluded content as answer evidence in the evaluation benchmark.

### PDF extraction stage

1. **Read every PDF and retain evidence.** Capture the source SHA-256 hash, original
   extracted text, physical lines, page geometry, font sizes, metadata and bookmarks.
2. **Detect margin noise using position and repetition.** A repeated line must be
   wholly within the top/bottom 9%, have the same wording and similar vertical
   position, and occur on at least three pages and 20% of the document's pages.
   Preserve its first occurrence. Isolated page-number patterns are removed only
   within the outermost 6%. Repeated body text is preserved.
3. **Remove nominated margin text in memory.** Text-only PDF redactions affect the
   temporary document, never the source file or images. If token counts show that
   an overlapping font box removed additional text, restore that page and flag it.
   Pages with existing redaction annotations are not edited by this step.
4. **Extract structure.** PyMuPDF4LLM attempts reading-order Markdown with font-based
   headings and ruled-table detection. No images or generated descriptions are
   inserted. Extraction is local and does not use a generative model.
5. **Check preservation.** Compare source and output token multisets. Require 99%
   token coverage and retention of numeric tokens, mathematical symbols, and
   selected rule words such as `not`, `unless`, `minimum`, `and`, and `or`.
   If that fails, use Poppler's layout extraction. Course-code-heavy pages without
   detected tables also use this fallback to preserve row alignment.
6. **Preserve difficult text.** If the layout fallback fails the same checks, retain
   the post-margin-removal original text and flag its reading order for review.
   Record failed conversion attempts rather than hiding them behind the fallback's
   successful text-coverage score.
7. **Normalize conservatively.** Use NFC Unicode normalization, expand known Latin
   ligatures, normalize nonbreaking spaces and known leading bullet glyphs, remove
   control artifacts and soft hyphens, trim trailing whitespace and excessive blank
   lines. Preserve horizontal table/code spacing, Tamil, Sinhala, formulas,
   superscripts, subscripts, case, stopwords and repeated educational content.
8. **Keep provenance and deduplicate exactly.** Retain a text and audit file for
   every successfully processed PDF. Exclude byte-identical or cleaned-text-identical
   duplicate documents from `corpus.jsonl`, recording `duplicate_of`. Do not deduplicate
   individual paragraphs or silently collapse similar handbooks from different years.

`clean_text(text)` is independently reusable; `clean_pdf(path, output_dir,
source_root, config)` handles one complete PDF. The numerical thresholds live in
the immutable `CleaningConfig` dataclass and are serialized with the outputs.

## Review signals and practical limits

**This is an automatically cleaned baseline, not a manually verified gold corpus.**
PDFs do not encode reading order or table semantics consistently. A high token
coverage score cannot establish that a table row, formula, or column was interpreted
correctly. Layout conversion may reflow text or join hyphenated words even though
the final normalization function does not guess at hard-hyphen repairs. No engine
here reconstructs images, chart relationships or mathematical layout.

| Signal | What to inspect |
|---|---|
| `no_extractable_text`, `ocr_candidate` | Blank, cover, scanned or image-only pages. OCR is not run automatically. Source pages and metadata are retained; empty documents are excluded from indexing. |
| `substantial_image_content` | A raster image occupies at least 20% of the page; text extraction does not describe its content. A decorative background can also trigger this flag. |
| `layout_fallback_review` | Markdown could not satisfy text preservation, or a course table lacked a reliable structured extraction. Inspect column and row order. |
| `reading_order_review_required` | The final fallback preserves original extracted text but cannot promise correct layout. |
| `table_structure_review` | An inferred Markdown table needs a visual check for cell alignment and headers. |
| `unmapped_private_use_characters`, `replacement_characters` | Font/encoding issues. Unknown glyphs are preserved rather than deleted or guessed. |
| `fragmented_text_or_formula` | Many isolated letters; this may be an extraction issue or valid mathematics. |
| `contains_tamil_or_sinhala` | Language coverage needs consideration for the planned English embedding model; the text is retained. |
| `margin_removal_skipped_to_preserve_text` | A proposed margin edit overlapped other text, so the original page text was preserved. |

Flags are **not automatic rejection decisions**. Review the source pages supporting
your benchmark questions, and document which document types and languages are
included in the study. OCR/image interpretation and correction of unusual tables
remain separate review tasks. Raw audit text is itself a PDF text extraction, not
a substitute for visually inspecting the PDF.

For a fair experiment, freeze the cleaned dataset and manifest before comparison,
use the same accepted evidence across all five strategies, and distinguish PDF
extraction/cleaning failures from chunking, retrieval and generation failures.
Avoid placing near-identical annual handbooks in different evaluation splits without
checking for leakage. Exact deduplication alone does not detect near duplicates.

## References behind the extraction approach

- [pypdf: why text extraction is hard](https://pypdf.readthedocs.io/en/4.3.1/user/extract-text.html#why-text-extraction-is-hard): PDF text order, tables, headers and OCR limitations.
- [PyMuPDF text extraction](https://pymupdf.readthedocs.io/en/latest/recipes-text.html): geometry and reading-order considerations.
- [PyMuPDF4LLM source at the pinned version](https://github.com/pymupdf/pymupdf4llm/tree/v0.0.27): structure-aware Markdown extraction. The installed version's API is used; current documentation may describe newer optional OCR/layout features.
- [Unicode normalization](https://www.unicode.org/reports/tr15/): canonical versus compatibility normalization.

The choices and thresholds above are an explicit project policy, not a claim that
these settings have already been experimentally validated as optimal.
