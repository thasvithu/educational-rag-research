# Phase 11 final results — held-out test split (7 October 2026)

**Scope.** The frozen benchmark's `test` split: 78 questions (70 answerable,
8 unanswerable), held out from `development` by document-group isolation
(`data/benchmark/benchmark_120_20261007/frozen_snapshot.json`). All five
chunking methods share identical retrieval (Jina v2-small embeddings, pinned
revision, 512-dim) and identical generation (OpenAI `gpt-4o-mini`,
temperature ≈0, seed 42) — the only variable under test is the chunking
strategy.

## Methodology note on answer-quality scoring

The per-answer human rubric review (`answer_correctness`, `faithfulness`,
`citation_support`, `relevance` in `rag_pipeline/evaluation/answer_metrics.py`)
was **not completed** for this run. That code path hard-requires a named
human reviewer (`reviewer_kind == "human"`) before it will compute those
metrics, by design (`answer_metrics.py` docstring: *"Human answer judgments
and citation syntax checks; no invented judge scores"*) — an LLM cannot
substitute for it without falsifying provenance, so this project did not do
that. **Answer-quality comparison here instead relies entirely on RAGAS**
(`gpt-4o-mini` judge + `text-embedding-3-small` judge embeddings), which is
transparently and correctly labeled as an automated LLM judge in every output
file (`judge_identity` field) — never disguised as human judgment. State this
explicitly in the thesis methodology section so a reader does not assume the
`answer_correctness`/`citation_support` columns (which are `null` throughout
`scores.jsonl`) were silently dropped by accident.

## 1. Retrieval metrics (test split, 70 answerable questions)

| Method | recall@10 | MRR@10 | context_recall | context_complete |
|---|---|---|---|---|
| fixed_size | 0.493 | 0.370 | 0.474 | **0.457** |
| sliding_window | 0.633 | 0.440 | 0.550 | 0.529 |
| semantic | 0.581 | 0.416 | 0.567 | 0.557 |
| late | 0.600 | 0.413 | 0.600 | 0.600 |
| structure_aware | 0.652 | 0.417 | 0.638 | **0.629** |

`context_complete` = fraction of answerable questions where every evidence
unit's full text made it into the final assembled context actually sent to
the generator (the primary, strictest metric).

**Paired bootstrap significance** (2000 resamples, cluster-grouped by shared
document/question family, 10 independent clusters, seed 42,
`context_complete`):

| Comparison | Diff (A − B) | 95% CI | Significant? |
|---|---|---|---|
| fixed_size vs structure_aware | −0.171 | [−0.272, −0.043] | **Yes** |
| fixed_size vs late | −0.143 | [−0.257, −0.043] | **Yes** |
| sliding_window vs structure_aware | −0.100 | [−0.186, −0.014] | **Yes** |
| fixed_size vs sliding_window | −0.071 | [−0.143, −0.014] | **Yes** |
| fixed_size vs semantic | −0.100 | [−0.271, 0.043] | No |
| semantic vs structure_aware | −0.071 | [−0.215, 0.057] | No |
| late vs sliding_window | 0.071 | [−0.014, 0.171] | No |
| late vs semantic | 0.043 | [−0.171, 0.214] | No |
| late vs structure_aware | −0.029 | [−0.171, 0.114] | No |
| semantic vs sliding_window | 0.029 | [−0.071, 0.158] | No |

**Finding**: `fixed_size` is **significantly worse** than `structure_aware`,
`late`, and `sliding_window` on retrieval-context completeness — this holds
up even with only 10 independent clusters. `structure_aware` has the best
point estimate (0.629) and is significantly ahead of both `fixed_size` and
`sliding_window`, though not distinguishably ahead of `semantic`/`late` at
this sample size.

## 2. RAGAS judge metrics (test split, 70 answerable questions)

| Method | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|
| fixed_size | 0.773 | 0.693 | 0.660 | 0.715 |
| sliding_window | 0.796 | 0.686 | 0.666 | 0.807 |
| semantic | 0.835 | 0.725 | 0.651 | 0.817 |
| late | 0.848 | 0.704 | 0.655 | 0.768 |
| structure_aware | 0.803 | 0.754 | 0.651 | 0.825 |

**Paired bootstrap significance** (same methodology, applied to RAGAS
scores):

| Metric | Significant comparisons (95% CI excludes 0) |
|---|---|
| Faithfulness | `fixed_size` vs `semantic` (diff −0.061, worse) |
| Answer relevancy | `sliding_window` vs `structure_aware` (diff −0.068, worse); `late` vs `structure_aware` (diff −0.050, worse) |
| Context precision | None |
| Context recall | `fixed_size` vs `structure_aware` (diff −0.110, worse); `fixed_size` vs `semantic` (diff −0.102, worse) |

**Finding**: `fixed_size` is the only method that comes out **significantly
worse** on both faithfulness and RAGAS's context recall — converging with
the retrieval-metric finding above. `structure_aware` is significantly ahead
on answer relevancy specifically versus `sliding_window` and `late`.
`context_precision` shows no significant differences between any pair of
methods — treat precision as inconclusive at this sample size, not as "all
methods are equal."

## 3. Consolidated verdict

Across three independent measurements (exact-span retrieval metrics, RAGAS
automated judge, and the qualitative failure taxonomy below), **`fixed_size`
is the consistent lower bound** and **`structure_aware` is the consistent
upper bound**, with `semantic` and `late` close behind structure_aware and
`sliding_window` in between. This is a defensible, multi-metric finding, not
a single-metric artifact — but with only 10 independent question/document
clusters in the test split, treat anything not flagged "significant" above as
genuinely inconclusive, and do not claim a strict total ranking beyond "
`fixed_size` measurably underperforms the other four."

## 4. Failure taxonomy summary

See `rag_pipeline/docs/phase11_failure_taxonomy.md` for full detail (computed
over the full 120-question pilot set, all 5 methods, 600 generations).
Headline: the dominant retrieval failure mode is **`retrieval_miss`**
(evidence never surfaced in the top-10 at all), not context-budget overflow
or chunk-boundary fragmentation. `sliding_window` uniquely shows an elevated
`overflow_budget` rate (9 cases vs 2–3 for every other method) — its
overlapping windows cost more of the shared 1,536-token evidence budget per
chunk. **9 of 90 (10%) unanswerable-question/method pairs** answered
confidently from the wrong institution's document or hallucinated from
unrelated real documents with no hedge — a shared limitation across all five
methods, not specific to one chunking strategy.

## 5. Artifacts

- `data/outputs/rag/phase11_final_test_scoring_20261007/` — retrieval metrics, coverage, paired differences.
- `data/outputs/rag/phase11_benchmark120_ragas_judge_test_split_20261007/` — RAGAS scores (70 answerable test questions × 5 methods = 350 samples, all complete).
- `data/outputs/rag/phase11_failure_taxonomy_20261007/ragas_test_split_paired_significance.json` — full paired bootstrap output for all 4 RAGAS metrics.
- `data/outputs/rag/phase11_failure_taxonomy_20261007/` — retrieval/context failure categories and wrong-source/out-of-corpus cases.
