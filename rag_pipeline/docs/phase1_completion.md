# Phase 1 completion — 19 September 2026

## Result

Implemented the shared foundation and inspection command. No chunking methods,
tokenizer selection, embeddings, retrieval, generation, or benchmark work has run.

| Check | Result |
|---|---|
| Default corpus | 60 documents: Aysha 12, Haleema 24, Vithusan 24 |
| Physical page records | 6,215 |
| Explicitly excluded pages | 44 |
| Other blank prepared pages | 172 |
| Eligible continuous text regions | 169 |
| Foundation tests | 17 passed |
| Existing cleaning tests, including external-report regression | 17 passed |
| Isolated runtime | Full inspection and both test suites passed with only the four pinned Phase 1 packages |
| Original PDFs and prepared files | All 187 checked file hashes unchanged |
| Output integrity | Completed-run artifact hashes verified |
| Repeated corpus identity | Identical in development and isolated environments |
| Git handling | Outputs ignored; no commit or push performed |

The existing development environment also passed `uv pip check` (133 packages).
The isolated run used Python 3.11.16 on Linux with the existing system Git and
Poppler executable. The development environment was not replaced.

## Review these outputs

The verified run is `data/outputs/rag/phase1_clean_environment_20260919/`:

- [Readable report](../../data/outputs/rag/phase1_clean_environment_20260919/inspection_report.md)
- [Detailed examples](../../data/outputs/rag/phase1_clean_environment_20260919/inspection_examples.json)
- [PDF/source validation](../../data/outputs/rag/phase1_clean_environment_20260919/validation_report.json)

Paths above are relative to this documentation file's folder; the run artifacts
remain local and are not distributed through Git.

The examples include UCSC text spanning physical pages 1–2, a sample from every
member, and the UOC 2018–2022 handbook's separate eligible regions around its
excluded pages. A region is a boundary constraint for later chunkers, not a chunk.

## Run your own inspection

From the repository root, choose an unused run ID:

```bash
.venv/bin/python -m rag_pipeline inspect --run-id phase1_my_review
```

## Files and responsibilities

- `common/load_corpus.py`: consistency checks, exact text, safe metadata loading.
- `common/provenance.py`: document/page/heading/region types and span mapping.
- `common/chunk_schema.py`: deterministic IDs and validated source-slice records.
- `common/config.py`: strict configuration and path rules.
- `common/run_manifest.py`: environment, source hashes, run status, output hashes.
- `cli.py`: complete inspection workflow and readable reports.
- `02_data_cleaning/validate_corpus.py`: optional external report path to support
  read-only validation; its original default behavior remains supported.
- `requirements-phase1.txt`: four pinned runtime dependencies, included by the
  full requirements file.

## Limits and next action

These checks verify integrity and provenance. They do not reconstruct missing
images or prove that extracted tables/formulas have correct semantic relationships.
All later methods must apply the same recorded blank/excluded-page policy.
Token counts are not guessed; the chunk record permits null token fields until
the model/tokenizer decision is made.

Phase 2 will establish the experiment protocol, benchmark, and model feasibility.
`PLAN.md` was updated locally; its existing ignore rule was preserved.
