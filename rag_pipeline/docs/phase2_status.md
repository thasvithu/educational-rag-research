# Phase 2 handover — 20 September 2026

**Status: ready for Phase 3 development. LangChain and GPU preparation are complete; final benchmark review/freezing remains deferred until before Phase 11.**

**Workflow update from Vithusan:** use these 20 questions for development through
answer generation (Phases 3–10). The team will supply the larger reviewed benchmark
later, before final evaluation in Phase 11. Its absence does not block development.
Draft pilot scores are debugging observations, not final research findings.

## Ready now

| Item | Result |
|---|---|
| Resource decision | Local embeddings; CPU tested, RTX 3050 4 GB CUDA access confirmed outside sandbox; Groq free tier preferred for generation. |
| Embedding choice | Jina v2 small English selected. LangChain adapter passed CUDA token-output, pooling, padding, query consistency and overflow checks at 2,048 total tokens. |
| Comparison design | Same model and content-only pooling for five methods; late reuses fixed-size spans. Proposed sizes: 128/256/512. |
| Protocol | Draft metrics, evidence coverage, tuning limits, context budget, split rules and failure labels documented. |
| Pilot | 20 drafts: 18 answerable, one unanswerable candidate, one ambiguous. Seven source PDFs and 13 visually inspected reference pages. |
| Benchmark integrity | Exact spans, eligible regions, page references and hashes checked. Freeze requires independent human reviews and completed splits. |
| Prepared corpus | Unchanged. All methods will use the Phase 1 corpus identity. |

The 2,048-token model test used about 1.1 seconds per synthetic forward and
approximately 1.37 GiB peak process RAM. These numbers do not predict the full
experiment runtime or demonstrate retrieval quality. The model is provisional
until a development retrieval pilot. No corpus chunks or embeddings were produced.

## Open these files together

Start with [the short LangChain guide](langchain_guide.md). It explains the new code and commands.

1. `data/benchmark/pilot_review.md` — readable questions, draft answers, rubrics,
   exact evidence and links to rendered PDF pages. Start with the first ten.
2. [Annotation guide](annotation_guide.md) — how Aysha, Haleema and Vithusan review.
3. `data/benchmark/annotation_reviews.csv` — pending review worksheet.
4. [Experiment protocol](experiment_protocol.md) — proposed comparison rules.
5. [Model feasibility](model_feasibility.md) — pinned revisions, measured results,
   limitations and commands to reproduce the CPU probe.

## Commands and outputs

Verification completed: **39 pipeline tests and 17 cleaning tests passed (56
distinct tests)**. All 20 draft records passed integrity validation; readiness and
freeze commands correctly refused the unfinished benchmark. The 122 prepared
input files in the Phase 1 snapshot have unchanged hashes. Git ignore checks cover
the benchmark, source corpus and downloaded weights.

```bash
.venv/bin/python -m rag_pipeline.evaluation.validate_benchmark
.venv/bin/python -m unittest discover -s rag_pipeline/tests -v
```

The benchmark command succeeds for structurally valid drafts and reports review
blockers. `--require-ready` and `--freeze` must fail while the current draft gates
remain unmet. A validation success is not a human quality certification.

Local artifacts:

- `data/benchmark/pilot.jsonl`, `pilot_review.md`, `benchmark_manifest.json`,
  `annotation_reviews.csv`, and `pilot_source_pages/`.
- `data/outputs/rag/model_probe_20260920/` — pinned weights/tokenizer/custom source bundle.
- `data/outputs/rag/phase2_probe_20260920/` — capability measurements and environment.
- `data/outputs/rag/phase2_preparation_20260920/` — final validation/check reports.

Code and documentation can be reviewed in Git. All PDFs, model weights, question
records, review sheets and generated reports stay under ignored local directories.
No commit, push or merge was performed.

## Pre-Phase 3 changes completed

- Added `PreparedCorpusLoader`, a LangChain `BaseLoader` returning 169 `Document`
  regions from all 60 sources. Exact text, offsets, pages and exclusions are preserved.
- Added `JinaEmbeddings`, implementing LangChain `embed_documents()` and
  `embed_query()` plus a token-output helper for the later late-chunking phase.
- Tested CUDA FP32 on the RTX 3050, including a 2,048-token input. Peak PyTorch
  allocation was about 667 MiB, with 798 MiB reserved. Batch size 1 is the default.
- Added a `ChatGroq` factory and compatible pinned dependencies. No API requests
  were made. The exact free model remains unset until generation setup.
- Added model configuration, an offline teaching example and a readable guide.

Commands:

```bash
.venv/bin/python -m rag_pipeline.examples.inspect_langchain
.venv/bin/python -m rag_pipeline inspect-langchain --run-id my_langchain_review
```

New verification artifacts: `data/outputs/rag/phase2_langchain_cuda_20260920/`,
`phase2_langchain_cpu_20260920/`, `phase2_langchain_loader_20260920/`, and
`phase2_langchain_checks_20260920/` under the
same output root. Original CPU probe results remain as historical measurements.

## Before final evaluation

1. Cross-review and correct the pilot. **Human approvals currently: zero.**
2. Vithusan reviews the protocol, proposed runtime limits, benchmark workload,
   primary outcome and parameter grid. Free-only spending and local-only hardware
   are already decided; they do not need to be reconfirmed.
3. Complete source/version family grouping and author the development/test set.
   The provisional target remains 40 development + 80 test; **both currently have
   zero records**. Reviewed pilot questions may move to development, never test.
4. Reconcile reviews, validate and freeze the final benchmark before scored test
   runs. Record future generator decisions before answer experiments.

Phase 3 has not started. Its development prerequisites are ready under the revised
workflow. The larger benchmark and human approvals remain mandatory for final
scored research claims.
