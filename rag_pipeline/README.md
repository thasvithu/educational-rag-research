# RAG pipeline — Phase 1

Phase 1 supplies the common foundation for the five chunking methods: a verified
corpus loader, exact source references, eligible text regions, a chunk record
factory, configuration validation, and inspection reports.

Chunking algorithms and embedding models belong to later phases. No model or API
is loaded by this phase. The roadmap is in the project's `PLAN.md` when available.

## Run with the existing environment

From the project root:

```bash
.venv/bin/python -m rag_pipeline inspect --run-id phase1_my_review
```

On Windows, use `.venv\Scripts\python.exe` in place of `.venv/bin/python`.
Use a new run ID each time. Existing output directories are never overwritten;
Phase 1 intentionally has no resume or overwrite flag.

The command checks the prepared corpus, then validates original PDF hashes and
page counts with the existing cleaning validator. It saves all results under
`data/outputs/rag/<run-id>/`. It does not update `final_data/validation_report.json`
or any other input file.

## Reproduce the Phase 1 environment

Use Python 3.11 and the requirements-file workflow. In a fresh checkout:

```bash
uv venv --python 3.11
uv pip install --python .venv/bin/python -r requirements-phase1.txt
.venv/bin/python -m rag_pipeline inspect --run-id phase1_my_review
```

For Windows, use `.venv\Scripts\python.exe` in the second and third commands.
An existing environment does not need to be recreated. Install the missing Phase 1
requirements into it if necessary.

`requirements-phase1.txt` pins the complete four-package runtime dependency set
for this phase and the cleaning tools: PyMuPDF, PyMuPDF4LLM, markdown-it-py, and
mdurl. It is included by the full `requirements.txt`, which also contains the
later RAG dependencies. Do not install all later dependencies merely to inspect
the corpus. A fresh temporary environment with only these four packages was
tested on the lead's Linux/Python 3.11 setup; other operating systems still need
their own installation check.

Git must be available because output ignore rules and code revision are checked.
Poppler's `pdftotext` is required to re-extract PDFs and run the cleaning integration
tests. It is not required for prepared-text loading or PDF hash/page-count checks.
The environment report records its availability/version.

The current `pyproject.toml` has an empty dependency list. For this project, use
the requirements files with `uv pip`; do not use `uv sync` to manage this
requirements-based environment. If using `uv run`, add `--no-sync`.

Teammates also need a copy of the original `data/{member}/` PDFs and matching
`final_data/`; those ignored files are not distributed by Git.

## Configuration

The default is `rag_pipeline/config.json`. Paths inside it are relative to the
project root, regardless of the current working directory. An explicitly supplied
`--config` filename is resolved relative to the shell's working directory.

```bash
.venv/bin/python -m rag_pipeline inspect --config rag_pipeline/config.json --run-id phase1_second_review
```

Expected document/page/member counts describe the current corpus and are checked,
not silently inferred. A deliberately different corpus needs its own reviewed
configuration. Unknown configuration fields, invalid counts, unsafe paths,
unsupported gap policies, tracked output files, and non-ignored output locations
are rejected. Source and prepared-corpus directories must remain separate.

`seed` is recorded for future reproducibility. Current inspection samples use
deterministic member/source ordering, so they do not consume randomness.

## Files produced by an inspection

| File | Purpose |
|---|---|
| `inspection_report.md` | Read this first: counts, one sample per member, a cross-page example, and excluded-gap boundaries. |
| `inspection_examples.json` | Exact sample text, source spans, converted headings, and region boundaries. |
| `document_summary.json` | All documents with page counts, blank/excluded pages, regions, and source quality flags. |
| `validation_report.json` | Existing validator's PDF/source and corpus integrity results. |
| `corpus_snapshot.json` | Hashes of input manifest/JSONL/TXT/detail files and the effective corpus identity. |
| `run_config.json` | The complete effective configuration. |
| `environment.json` | Python, installed package versions, platform, Git revision/dirty state, source-code hashes, and Poppler version. |
| `stage_manifest.json` | Running/complete/failed status, timestamps, corpus/config hashes, artifact hashes, and errors. |

If validation fails, the CLI exits with status 1 and retains the failed run's
manifest for diagnosis. Correct the cause and use a new run ID. It will not repair
or regenerate the prepared corpus automatically. Early configuration/output-path
errors happen before an output directory is created.

## Shared contracts for future chunkers

### Load exact text

```python
from rag_pipeline.common.config import ROOT, load_config
from rag_pipeline.common.load_corpus import load_corpus

config = load_config(ROOT / "rag_pipeline/config.json")
corpus = load_corpus(ROOT / config.corpus_dir, config)
document = corpus.documents[0]

for region in document.regions:
    eligible_text = document.text[region.char_start:region.char_end]
    # Future chunkers choose spans inside this region.
```

The loader checks every TXT against JSONL and detailed page records, verifies
hashes and exact offsets, and retains only prepared text and typed metadata.
`raw_text` and `excluded_text` are not part of its document interface. The loader
alone checks internal corpus consistency; use `inspect` for original-PDF checks too.

### Respect gaps and source references

- Offsets use Python Unicode characters, not UTF-8 bytes or model tokens.
- Ends are exclusive: `[char_start, char_end)`.
- Original page headings in JSON are page-local. Loaded `Heading` objects use
  document-level offsets; the original files are not edited.
- Ordinary adjacent nonblank pages can share a region and eventually a chunk.
- Explicit exclusions **and blank/whitespace-only prepared pages** break regions.
  This conservative policy avoids joining text across potentially missing material.
- A page containing text plus images stays eligible and retains its quality flags;
  the loader cannot infer what the images contain.
- Separator-only spans, empty spans, and spans crossing gaps are rejected.
- Page citations include only intersecting nonempty page text; an empty page does
  not acquire a citation just because its zero-length offset falls inside a range.

`pages_for_span()` supplies page references. `load_structure_metadata()` provides
explicit, hash-checked lazy access to outline and layout candidates for eligible
pages; these candidates remain inferred and are not automatically trusted headings.

### Create a validated chunk record

`make_chunk()` in `common/chunk_schema.py` constructs the shared record from a
document, exact span, method, corpus/config hashes, and run metadata. It validates
the region, derives source pages/flags, and obtains text by direct slicing.
Never locate a span by searching for the first occurrence of its text.

IDs depend on corpus, document, method, method configuration, and source span.
They remain stable across execution run IDs. A changed corpus or configuration
produces new IDs. Pass the hash of the full effective chunking configuration.

Token count and tokenizer identity may both be null for Phase 1 inspection because
no tokenizer has been selected. Actual chunkers must supply both in later phases;
no character/word count is presented as a model token count. Configuration version
and model/tokenizer revisions must be included when those components are selected.

The inspected regions are **not chunks**. Their size is unrestricted; token-limited
chunk boundaries will be chosen by the later algorithms.

## Relevant checks

```bash
.venv/bin/python -m unittest discover -s rag_pipeline/tests -v
.venv/bin/python -m unittest discover -s 02_data_cleaning/tests -v
```

Synthetic fixtures are built in temporary directories during tests. Tests cover
Unicode, duplicate passages, offsets, headings, gap handling, deterministic IDs,
tampered input, configuration safety, output protection, failure reporting, and
the validator's external-report option.

## Current result and next phase

Phase 1 accounts for 60 documents and 6,215 pages. It identifies 44 explicitly
excluded pages, 172 other blank prepared pages, and 169 eligible text regions.
The latter counts reflect the selected gap policy, not a new deletion of input.

These checks establish provenance and internal integrity, not semantic accuracy
of tables, formulas, or image content. The next phase is the experiment protocol,
benchmark, and embedding-feasibility decision. No chunking or embedding has run.
