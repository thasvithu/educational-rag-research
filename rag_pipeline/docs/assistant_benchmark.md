# Assistant-authored benchmark — 6 October 2026

Vithusan requested that the assistant prepare questions directly from the corpus
so provisional experiments can continue without waiting for manual authoring.

## What is ready

A separate **100-question answerable set** is saved under
`data/benchmark/assistant_100_20261006/`. It includes 60 handbook questions and
40 lecture questions, with 10 questions from each of 10 source PDFs.

- [Spreadsheet](../../data/benchmark/assistant_100_20261006/questions.csv)
- [Readable question book and PDF page images](../../data/benchmark/assistant_100_20261006/question_book.md)
- [Scope, selection notes and reproduction instructions](../../data/benchmark/assistant_100_20261006/README.md)
- [Validation report](../../data/benchmark/assistant_100_20261006/validation_report.json)

Each question has a reference answer, rubric, source scope, category, exact
supporting passages, physical PDF pages, character offsets and source hashes.
The assistant visually checked 50 original PDF pages. The 138 evidence units
have 144 exact-span alternatives in total. All 100 records pass schema,
source-span, page and source-hash validation. The original team CSV, original
benchmark manifest and all 122 prepared corpus files remain unchanged.

## How to interpret it

The author is **Codex (AI assistant)**. Assistant checks are recorded separately
from the pending independent human review. No human approvals are inferred.
The same assistant authored and checked the questions, so these checks are not
an independent review of the labels.

All questions are assigned to `pilot`, for provisional development comparisons.
There are no held-out test questions or unanswerable questions in this set.
Earlier pilots and team drafts overlap with some source material. Do not call
these unseen test results or use them to assess abstention performance.

This is a deliberate sample of 10 documents, with repeated topics within source
families. The equal 10-per-document allocation is a practical sampling choice,
not a statistical-power justification. Source groups are kept together for
paired uncertainty estimates. Alternative evidence locations are not exhaustive.

The existing final benchmark review/freeze checks remain in place. Provisional
work can continue now. Final thesis reporting needs an explicit account of AI
authorship, independent validation actually performed, sampling limitations and
whether an independent held-out evaluation was completed.

## Continue the experiment

The local `retrieval_config.json` points to the new benchmark and the same five
full-corpus indexes/vectors. It preserves the existing retrieval settings. Only
question text enters the query encoder; reference answers and evidence labels
do not restrict retrieval.

The local `generation_config.json` points to this new retrieval run. Prepare
requests with `--dry-run --all-pilot` before making hosted calls. This writes
empty, checksum-recorded answer files so offline evaluation reports missing
answers explicitly. A dry run makes no provider calls, including when old
responses exist in the shared cache. The configured model and free-tier limits
must be checked again before live generation.

The authored spec, page-check log and `build_records.py` are preserved with the
ignored benchmark. Rebuild only before experiments; create a new version for
later corrections so input checksums continue to identify the labels used.

## Completed provisional run

All 100 questions were searched across all 60 documents for each of the five
methods, producing **500 retrieval/context records**. The independent retrieval
audit checked reloaded indexes, rankings against float64 dot products, source
spans, pages, context budgets and evidence coverage. All checks passed. All
122 prepared inputs and 26 retrieval input artifacts were unchanged.

The generator dry run prepared **500 requests and zero answers**, with no API
calls. Its independent audit checked all 500 prompts and their source evidence.
The offline evaluator now reports retrieval metrics for all 500 records and
keeps all answer-quality fields unscored.

Artifacts:

- Retrieval: `data/outputs/rag/phase11_ai100_retrieval_20261006_gpu/`
- Prepared prompts: `data/outputs/rag/phase11_ai100_prompts_20261006/`
- [Provisional evaluation report](../../data/outputs/rag/phase11_ai100_provisional_20261006/report.md)
- [Per-method metrics](../../data/outputs/rag/phase11_ai100_provisional_20261006/metrics_summary.csv)

The earlier attempt without GPU access is preserved as a failed run at
`phase11_ai100_retrieval_20261006`; the successful run has the `_gpu` suffix.

The prompt preparation estimates 1,199,701 input tokens for the 500 requests.
With the configured answer allowance and counting margin, it reserves 1,583,701
tokens. Live generation will need staged execution under the free-tier quota;
this is a local estimate, not provider usage or confirmation of today's limits.
No answer correctness, faithfulness or abstention conclusion is available yet.

The generation change and existing evaluation checks passed **32 focused tests**.
The dry-run check verifies that no provider client is created and that each
method has an empty answer export. This makes missingness auditable without
silently generating or reusing answers.

The offline evaluation was repeated with identical inputs. The independent
character-position audit recomputed all **500 question–method records** and
matched every retrieval metric; repeat artifacts were identical (apart from
environment metadata). Evidence integrity and repeatability do not establish
that the AI-authored labels are independently correct. The audit is saved as
`data/outputs/rag/phase11_ai100_provisional_20261006_audit.json`.
