# RAG pipeline — corpus preparation through grounded answers

Phase 1 supplies the common foundation for the five chunking methods: a verified
corpus loader, exact source references, eligible text regions, a chunk record
factory, configuration validation, and inspection reports.

Phase 1 inspection loads no model or API. Phase 2 adds an explicit offline model
probe and benchmark validation. Phase 3 adds fixed-size chunking across the corpus.
Phase 4 adds sliding windows using the same tokenizer and source-reference rules.
Phase 5 adds structure-aware sections, paragraph/table/list packing and recorded fallbacks.
Phase 6 adds semantic boundaries using cached local Jina embeddings and recorded scores.
Phase 7 embeds the four completed chunk corpora using a shared batched encoder.
Phase 8 creates contextual vectors at the exact fixed-size boundaries.
Phase 9 indexes all five methods and saves pilot retrieval and budgeted contexts.
Phase 10 turns those contexts into traceable, cited answers through Groq.
The roadmap is in the project's `PLAN.md` when available.

## Phase 10: grounded answer generation

LangChain builds the same prompt for every method and calls Groq
`qwen/qwen3.8-27b`. The runner prepares all 100 pilot requests, then selects
three questions across all five methods for the first live sample. It saves
exact prompts/evidence, answers, citations, provider usage, retries and errors.
A persistent cache and quota ledger support later all-question runs and resume.
See the [Phase 10 handover](docs/phase10_completion.md) for commands and results.

```bash
.venv/bin/python -m rag_pipeline generate-answers --run-id phase10_prepare_new --dry-run
.venv/bin/python -m rag_pipeline generate-answers --run-id phase10_pilot_new
```

The second command can call the configured Groq account when matching answers
are not cached. Set `GROQ_API_KEY` in the ignored `.env` or environment. Use
`--cache-only` with a new run ID for a replay without provider calls.

## Phase 9: indexing and retrieval

Five separate LangChain/FAISS indexes hold **53,977 vectors**. The shared pilot
produced **100 retrieval results and contexts** from 20 query vectors. Raw top-10
results, exact source pieces and budget decisions are saved separately. All
indexes, rankings and contexts match the cached repeat. See the
[Phase 9 handover](docs/phase9_completion.md) for commands and review examples.
Pilot annotation review remains pending; these are development diagnostics.

## Phase 8: late chunking

The same Jina model reads longer passages before pooling each fixed-size chunk.
All 60 documents produced **8,627 matched contextual vectors** across **1,804
encoder windows**, with exact source spans, one owner per chunk and saved
window-placement diagnostics. Read the [Phase 8 handover](docs/phase8_completion.md)
for the simple explanation, commands, audit, resource measurements and limitations.

## Phase 7: shared chunk embeddings

The common Jina encoder produces **45,350 vectors of 512 dimensions** across the
four methods, using CUDA FP32 and a tested batch size of eight. Each method has
its own matrix, exact chunk copy and vector-row mapping. A separate cache and
explicit resume command preserve work after interruption. See the
[Phase 7 handover](docs/phase7_completion.md) for commands, validation, source
examples and time/storage measurements. Phase 8 uses its fixed-size vectors as
the matched early baseline.

## Phase 6: semantic chunking

The semantic method produced 11,927 chunks across all 60 documents. It compares
adjacent text units, uses the region's 90th-percentile distance as a starting rule,
and retains the 256-content-token cap. Read the [Phase 6 handover](docs/phase6_completion.md)
for GPU/cache commands, scores, validation and limitations. Boundary vectors are
separate from the final retrieval embeddings created in Phase 7.

## Phase 5: structure-aware chunking

The structure-aware method produced 13,377 chunks across all 60 documents under
the same 256-content-token cap. Read the [Phase 5 handover](docs/phase5_completion.md)
for commands, heading decisions, fallback counts and review examples.

## Phase 4: sliding-window chunking

Sliding windows use 256 content tokens with 64-token overlap. All 60 documents
produced 11,419 chunks. Read the [Phase 4 handover](docs/phase4_completion.md) for
commands, overlap examples and the comparison with the fixed-size baseline.

## Phase 3: fixed-size chunking

The baseline uses a LangChain `TextSplitter` with the pinned Jina tokenizer:
256 content tokens, no overlap, exact source spans and retained tails. All 60
documents produced 8,627 chunks. Read the [Phase 3 handover](docs/phase3_completion.md)
for results, boundary examples, output files and runnable commands.

## Phase 2: start here

- [Start here: LangChain guide and runnable examples](docs/langchain_guide.md)
- [Status and next steps](docs/phase2_status.md)
- [Experiment protocol](docs/experiment_protocol.md)
- [Model feasibility results and reproduction](docs/model_feasibility.md)
- [Annotation and review guide](docs/annotation_guide.md)

The hardware decision is **this computer only**; the LLM budget is **free only**.
CPU and CUDA embedding probes passed. LangChain loader, embedding and Groq
interfaces are ready. Twenty pilot drafts are under
`data/benchmark/pilot_review.md`; human review and development/test authoring are due before final evaluation.
They do not block Phase 3 development with the pilot.

```bash
.venv/bin/python -m rag_pipeline.evaluation.validate_benchmark
```

This validates exact reference text and reports unfinished review work. It does
not generate chunks or scores. Add `--require-ready` to require all freeze gates;
it currently fails as intended. See the model report for the optional probe command.

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
of tables, formulas, or image content. Phase 2's narrow model probe has run;
all five chunking/vector paths and shared retrieval are implemented. Phase 10
adds a small live generation pilot. Benchmark review/freezing and controlled
evaluation remain for Phase 11; development examples are not final findings.
