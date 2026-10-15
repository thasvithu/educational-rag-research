# Phase 11 automated failure taxonomy (7 October 2026)

**What this is.** An automated, evidence-based first pass at the PLAN failure
taxonomy (`experiment_protocol.md`, "Generation, repeats, and failure
analysis": extraction, boundary fragmentation, missing reference context,
table/list separation, overflow, retrieval miss, wrong source/version,
redundant overlap, context loss, generation error, benchmark/evaluator
error), computed directly from saved retrieval/context/generation traces with
no new model or API calls.

**What this is not.** A substitute for human qualitative review. Several PLAN
categories (extraction quality, table/list separation, true "context loss"
root-causing) need a person reading the actual chunk text; this pass only
automates the categories that are mechanically derivable from the retrieval
and context-assembly diagnostics that are already recorded for every
question/method pair. Treat the labels below as a structured starting point
for human review, not a final verdict.

**Source data.** `data/outputs/rag/phase11_benchmark120_full_openai_20261007`
(600/600 successful generations) and its provisional evaluation
`data/outputs/rag/phase11_benchmark120_full_openai_eval` (all 120 pilot
questions × 5 methods). Raw outputs saved at
`data/outputs/rag/phase11_failure_taxonomy_20261007/`.

## 1. Retrieval/context failures (answerable questions)

For each of the 500 answerable question×method pairs (100 answerable
questions × 5 methods), every evidence unit's `covered_at_k`,
`first_single_chunk_rank`, and `covered_in_context` fields (already computed
by `score_retrieval`) were used to classify every unit that failed to make it
into the final assembled context:

- **`retrieval_miss`**: the evidence never appeared in the top-10 retrieved
  chunks at all (not even jointly across several chunks).
- **`boundary_fragmentation`**: the evidence was jointly coverable across
  multiple retrieved chunks, but no single chunk covered it completely on its
  own (and the context-assembly step didn't end up including a complete
  union either).
- **`overflow_budget`**: a single chunk that fully covers the evidence *was*
  retrieved at a usable rank, but the 1,536-token evidence budget was
  exhausted before that chunk's turn (`decisions[...].status ==
  "skipped_budget"`).
- **`redundant_overlap_dedup`**: the covering chunk was dropped because its
  content was already fully covered by an earlier higher-ranked chunk
  (`status == "skipped_no_new_nonwhitespace_text"`). **0 cases observed.**

| | Pairs | Fully successful (`context_complete=1`) |
|---|---|---|
| Answerable question × method pairs | 500 | 266 (53.2%) |

Unit-level failure counts (one question can have >1 evidence unit, so counts
exceed the number of failing pairs):

| Category | Count |
|---|---|
| `retrieval_miss` | 285 |
| `overflow_budget` | 19 |
| `boundary_fragmentation` | 1 |
| `redundant_overlap_dedup` | 0 |

By method:

| Method | `retrieval_miss` | `overflow_budget` | `boundary_fragmentation` |
|---|---|---|---|
| fixed_size | 68 | 3 | 0 |
| sliding_window | 56 | 9 | 0 |
| structure_aware | 48 | 2 | 0 |
| semantic | 54 | 3 | 0 |
| late | 59 | 2 | 1 |

**Reading this honestly**: `retrieval_miss` dominates every method by a wide
margin — the single biggest bottleneck in this pipeline is not context-budget
overflow or chunk-boundary fragmentation, it's the embedding/search step
simply not surfacing the right passage in the top 10 at all. `structure_aware`
has the fewest misses (48) and `fixed_size` the most (68), consistent with the
RAGAS/retrieval-metric ranking reported earlier. `sliding_window` stands out
with the most `overflow_budget` cases (9 vs 2-3 for everyone else) — its
25%-overlap windows cost more of the shared 1,536-token evidence budget per
chunk, so it runs out of budget more often even when it does retrieve the
right chunk.

## 2. Wrong-source and out-of-corpus cases (unanswerable questions)

For each unanswerable question×method pair (18 unanswerable questions × 5
methods = 90 pairs), checked whether the assembled context's source documents
belonged to a different `document_groups` family than the question's own
named institution/handbook (`group_id`), and whether the model's answer text
contained a hedging phrase (pattern match on "not mentioned", "does not
state", "unable to determine", etc. — a crude heuristic, not a judgment of
correctness).

`out_of_corpus_probe` questions (deliberately about institutions/documents
**not in the corpus at all**) are reported separately from genuine
wrong-source cases, since there is no "correct" document for them to begin
with.

| Category | Count | Meaning |
|---|---|---|
| `wrong_source_hedged` | 21 | Retrieved a different institution's handbook, but the answer hedged/flagged uncertainty |
| `wrong_source_unflagged` | 5 | Retrieved a different institution's handbook and answered confidently with no caveat |
| `out_of_corpus_hedged` | 26 | Correctly had nothing in-corpus to cite; answer hedged |
| `out_of_corpus_unflagged` | 4 | Retrieved unrelated real documents for an out-of-corpus question and answered confidently anyway |

By method (counts of the two "unflagged"/confident-wrong categories):

| Method | `wrong_source_unflagged` | `out_of_corpus_unflagged` |
|---|---|---|
| fixed_size | 2 | 1 |
| sliding_window | 1 | 1 |
| structure_aware | 0 | 1 |
| semantic | 0 | 1 |
| late | 2 | 0 |

**9 of 90 unanswerable-question/method pairs (10%)** produced a confident
answer sourced from the wrong institution's document with no caveat. This
revises the earlier informal "14/600" estimate from a looser ad hoc check
during an interim session; this version is methodologically cleaner (separates
genuine wrong-source from out-of-corpus-probe cases, and is reproducible from
`wrong_source_and_ooc_cases.json`). Every method produces at least one such
case — this is a shared retrieval/generation limitation (same-template
handbook boilerplate text across institutions reads as highly similar under
any embedding), not specific to one chunking method.

## 3. Categories not covered by this automated pass

- **`extraction`** (faithful extraction of facts from retrieved text) and
  **true `context_loss`** (semantic degradation even when text is present) —
  need a human reading the answer against the evidence; this is exactly what
  the pending answer-review pass (`answer_review_template.jsonl`) is for.
- **`table_list_separation`** — would need per-chunk structural metadata
  cross-referenced with evidence spans that fall inside table/list rows; not
  attempted here.
- **`generation_error`** — **0 cases**. All 600/600 generations in
  `phase11_benchmark120_full_openai_20261007` completed successfully
  (confirmed via `stage_manifest.json` status and per-method `answers.jsonl`
  status fields).
- **`benchmark_evaluator_error`** — **0 cases**. `validate_benchmark` reports
  `valid: true` with zero errors throughout; no `validate_trace` failures were
  raised while building the 500 provisional scoring rows used here.

## 4. Raw artifacts

- `data/outputs/rag/phase11_failure_taxonomy_20261007/retrieval_context_failures.json`
  — full per-category/per-method counts and three example question IDs per category.
- `data/outputs/rag/phase11_failure_taxonomy_20261007/wrong_source_and_ooc_cases.json`
  — every unanswerable question×method pair's classification, including the
  retrieved document groups and hedge-detection result.
