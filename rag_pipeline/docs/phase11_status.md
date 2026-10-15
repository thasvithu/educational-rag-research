# Phase 11 — provisional evaluation implementation

**6 October 2026. Status: in progress.** The offline evaluation code is working.
Final benchmark evaluation and thesis findings are still pending.

## New benchmark preparation — 6 October 2026

The user subsequently requested assistant-led authoring instead of waiting for
the team correction queue. A separate [100-question AI-authored benchmark](assistant_benchmark.md)
now passes source and format validation. It has 138 evidence units from 10 PDFs;
50 physical PDF pages were visually checked by the assistant. Independent human
review and final freezing remain pending. The older 20-question run described
below is preserved; its counts should not be confused with the new set. The new
run completed 500 retrieval evaluations and prepared 500 generation prompts,
with zero generated answers and no API calls. See the [new provisional report](../../data/outputs/rag/phase11_ai100_provisional_20261006/report.md).
Retrieval and prompt/source audits passed; 32 focused tests passed.

## What this step does, in simple English

For each question and each chunking method, we ask:

1. Did the retrieved chunks contain the required supporting passage?
2. How early did a useful chunk appear in the ranked results?
3. Did that evidence survive the token limit and reach the answer generator?
4. Was an answer generated, and has someone checked its correctness and support?

We now save these checks separately. Retrieving the right evidence does not
automatically mean the answer is correct. A citation such as `[S1]` does not
automatically mean the cited passage supports the claim.

## Scope and saved results

The run uses the **existing 20 draft pilot questions**, not the 153 imported
team questions. There are 18 answerable, one unanswerable and one ambiguous
question. Each of five methods contributes a record, giving **100 records**.
Only **15 answers** were generated in Phase 10; **85 remain unexecuted**.
The reference annotations and answer judgments still need human review.

Main output: `data/outputs/rag/phase11_provisional_20261006_v2/`.
Repeat: `data/outputs/rag/phase11_provisional_20261006_v2_repeat/`.
The earlier `phase11_provisional_20261006/` development run is preserved.

No model weights are loaded, no API is contacted, and no benchmark labels,
source documents or earlier experiment outputs are edited by the command.
It loads the local pinned tokenizer to verify context packing.

## Commands

Use a new run ID for each execution:

```bash
.venv/bin/python -m rag_pipeline evaluate --run-id phase11_check_new --provisional
.venv/bin/python -m rag_pipeline evaluate --run-id phase11_check_repeat --provisional
.venv/bin/python -m rag_pipeline.evaluation.audit \
  --run-id phase11_check_new --repeat-id phase11_check_repeat
```

`--retrieval-run` and `--generation-run` accept other compatible saved pilot
runs. They must have complete, checksum-verified manifests and matching
corpus, questions, evidence and generation settings. This initial command
does not accept final test splits or intake-schema drafts. It requires
`--provisional`; omitting that flag fails before creating output.

Dependencies are recorded in `requirements-phase11.txt`, extending the
existing Phase 10 environment. RAGAS 0.2.6 is optional for offline metric
computation; its installed prompt definitions are recorded when available.
The new tests include its schema and stubbed adapter behavior.

## Files to inspect first

| File inside the evaluation run | What it tells you |
|---|---|
| `report.md` | Short explanation of the run and limitations. |
| `coverage.csv` | Completed/missing answers, review coverage, saved generation usage and context tokens. |
| `metrics_summary.csv` | Scores by method and answerability, with scored and eligible question counts. |
| `category_metrics.csv` | The same measurements by question category. Small groups are descriptive only. |
| `scores.jsonl` | Every question's evidence-unit decisions and individual metrics. |
| `paired_differences.json` | Paired differences in complete evidence after context packing. |
| `answer_review_template.jsonl` | Fifteen saved answers, their rubrics and exact supplied evidence, with blank judgment fields. |
| `ragas_status.json` | Installed metric requirements and prompt definitions. Judge status is `not_run`. |
| `benchmark_validation.json` | Existing benchmark validation and outstanding freeze blockers. |

Method-specific copies of `scores.jsonl` appear under each method's
`variant_001/` directory. JSON `null` and blank CSV means mean **not scored or
not applicable**, never an automatic zero. No method winner is claimed.

## Metric rules implemented

- **Evidence Recall@K:** fraction of required evidence units fully covered by
  the top K source spans. K is 1, 3, 5 and 10. Several chunks can jointly cover
  an evidence unit. Overlap counts once.
- **Complete evidence@K:** all required evidence units are covered.
- **MRR@K:** reciprocal rank of the first single chunk that covers at least one
  complete evidence unit. Joint coverage alone does not qualify.
- **Context recall / complete evidence:** apply coverage to source pieces
  actually supplied to generation. Complete evidence here is the proposed
  primary outcome.
- **Alternative evidence:** at least one entire alternative must be covered.
  Parts of different alternatives cannot be combined. Whitespace gaps are
  allowed; missing non-whitespace characters are not.
- **Answer correctness:** reviewer-marked rubric facts satisfied divided by
  facts required. Reviewers must check numbers, units, conditions and exceptions
  explicitly. Fully correct also requires no contradiction. Answerable-question
  abstention is incorrect.
- **Faithfulness / citation support:** reviewer-counted supported claims over
  checkable claims / cited claims respectively. A zero denominator is N/A.
- **Relevance:** human rating 0, 1 or 2, with counts and mean.
- **Unanswerable behavior:** reviewed abstention without an unsupported answer,
  and unsupported-answer rate, reported separately from retrieval metrics.

Missing retrieval or context is a conservative zero for answerable questions.
Missing answers and failed, absent or stale evaluator judgments remain missing
with coverage shown. Ambiguous and extraction-diagnostic questions are excluded
from primary retrieval scores. The runner currently imports completed source
stages; support for importing failed-stage manifests remains future work.

Paired comparisons resample whole connected source/question families, retaining
all questions in each sampled family. The code uses 2,000 bootstrap samples,
seed 42, and percentile 95% intervals. It reports no interval for fewer than two
independent groups. These draft-pilot intervals do not demonstrate statistical
power, significance, or generalization to the final benchmark.

## Human answer review

Copy `answer_review_template.jsonl` outside the completed run before editing.
For each answer, read the rubric and actual supplied evidence, then fill:

- `facts_satisfied`: one `true` or `false` for each rubric item.
- `contradiction`, `abstention`, `unsupported_answer`: true/false judgments.
- `checkable_claims`, `supported_claims`, `cited_claims`,
  `supported_cited_claims`, `uncited_claims`: nonnegative claim counts.
  Each checkable claim belongs to either the cited or uncited group.
- `relevance`: 0 (does not address the question), 1 (partly addresses it),
  or 2 (directly addresses it).
- `reviewer`, `notes`, and `status: "approved"` when the human review is done.

Keep `review_identity`, question/method/trial IDs and supplied trace fields
unchanged. The identity binds the review to the exact annotation, answer and
context. A changed answer or stale review yields missing scores with an error.
Pending entries may remain in the file.

```bash
.venv/bin/python -m rag_pipeline evaluate --run-id phase11_reviewed_new \
  --provisional --answer-reviews data/benchmark/pilot_answer_reviews.jsonl
```

This still produces **provisional** scores: reviewing generated answers does not
approve the underlying benchmark annotations.

## Code map

| Module under `rag_pipeline/evaluation/` | Responsibility |
|---|---|
| `inputs.py` | Verify experiment checksums, questions, exact source spans, context assembly and prompt lineage. |
| `retrieval_metrics.py` | Evidence-unit coverage and ranking metrics. |
| `answer_metrics.py` | Validate human judgments and parse individual/grouped citation labels. |
| `compare_runs.py` | Aggregate with coverage and compute paired grouped bootstrap differences. |
| `ragas_adapter.py` | Optional explicit-client RAGAS interface; stub-tested, not live-validated. |
| `run.py` | Run the offline checks and write reviewable outputs. |
| `audit.py` | Independently recompute metrics using character positions and verify repeat outputs. |

The RAGAS adapter requires caller-supplied judge and embedding clients and their
configuration identity. It uses the text actually supplied to generation and
records invalid/failed outputs as missing. The offline CLI does not invoke it.
Before live use, select the judge, enforce the zero-paid-budget/rate policy,
capture provider usage, inspect a small sample and add that run integration.
Its API/schema behavior follows the installed pinned version and the
[RAGAS 0.2.6 evaluation reference](https://docs.ragas.io/en/v0.2.6/references/evaluate/).

## Verification and remaining work

**Earlier team handoff (now optional for continued provisional work):** use the [author correction sheets](../../../data/benchmark/reviews/team_corrections_20261006/README.md)
to resolve the imported benchmark. The queue has 73 evidence-confirmation cases,
25 answerability decisions and 29 wording-only cases; 26 other annotations have
no automated intake flag. These are workflow groups, not 127 proven wrong answers.
Each correction row contains a concrete action and preserves the original question
and answer. Suggested wording remains separate until a teammate accepts it.

All **154 pipeline tests passed**, including **19 new evaluation tests** for
coverage, alternatives, Unicode, missingness, review identity, source/context
validation, grouped comparisons and stubbed RAGAS execution.

The independent audit recomputed all **90 answerable question–method records**
using non-whitespace character positions. It matched every saved retrieval
metric, confirmed identical repeat artifacts (excluding environment metadata),
and checked **122 unchanged prepared files**. The audit is saved beside the run
as `phase11_provisional_20261006_v2_audit.json`. This verifies implementation and
repeatability; it does not validate the draft labels or an automated judge.

Phase 11 remains in progress. Before final scoring, we still need:

1. Final benchmark selection, resolved source/evidence records and independent review; disclose any AI-authored annotations. The imported team questions and assistant-authored pilot are separate options, not automatically combined.
2. Protected development/test groups and an approved, frozen protocol.
3. Method settings chosen on development questions, then frozen.
4. Comparable retrieval/generation across the selected benchmark.
5. Human answer review and a validated, budgeted automated-judge workflow if used.
6. Final scored outputs and uncertainty reporting, followed by Phases 12–13.
