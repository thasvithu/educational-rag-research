# Phase 10 — Grounded answer generation

Implemented and pilot-validated on **27 September 2026**.

## What the code does

For each question, the model reads the evidence saved by Phase 9 and writes an
answer. It receives the same instructions and limits for all five methods. Each
answer is linked to its complete prompt, exact source passages, source files,
physical PDF pages and provider usage. Reference answers never enter the prompt.

```text
Saved question + each method's saved evidence
    → LangChain ChatPromptTemplate
    → count the complete prompt with the generator tokenizer
    → ChatGroq → answer and provider metadata
    → save response, citation links, usage and errors
```

This stage tests answer generation on development drafts. It does not establish
which chunking method is best. Team review and the larger frozen benchmark are
still required before Phase 11 scored evaluation.

## Shared generator settings

| Setting | Value |
|---|---|
| Provider / interface | GroqCloud through LangChain `ChatGroq` |
| Model | `qwen/qwen3.8-27b` |
| Reasoning mode | `none`; return the answer without a reasoning trace |
| Temperature / top-p / seed | `1e-8` / `1.0` / `42` |
| Maximum answer | 512 generator tokens |
| Total budget | 8,192 tokens, including a 256-token estimation margin and the answer reserve |
| Retrieved evidence | Unchanged Phase 9 contexts: at most 1,536 Jina content tokens, before wrappers |
| Budget policy | Free account, zero paid budget, no model/provider fallback |
| First live sample | Questions `pilot_001`, `pilot_004`, `pilot_019`, each across all five methods |

The account's model list did not include the earlier proposed Llama model when
checked on 27 September 2026. Qwen was available. Groq documents its `none`
reasoning mode and publishes free-plan limits of 30 requests/minute, 1,000/day,
8,000 tokens/minute and 200,000/day for this model.
[Model documentation](https://console.groq.com/docs/model/qwen/qwen3.8-27b),
[rate limits](https://console.groq.com/docs/rate-limits).

The pipeline uses lower local limits: 24 requests/minute, 900/day, 7,000 tokens/minute
and 180,000/day. A shared ledger reserves the estimated prompt, maximum answer and
margin before every attempt, including failed attempts. Other use of the same
Groq account is outside this ledger; provider limits remain authoritative.
This intentionally conservative scheduling can leave some free quota unused.

Groq does not expose billing-tier verification through this client. The free-tier
setting records the user's account/budget choice; it is not a technical guarantee
that an arbitrary paid account cannot be billed.

## Tokens and reproducibility

Jina tokens measure retrieval evidence. Qwen tokens measure generation. They are
different tokenizers and their counts cannot be substituted for each other.
The complete serialized chat is counted with Qwen's pinned tokenizer/template;
the actual prompt/answer usage returned by Groq is also saved.
Evidence and wrapper token counts are logged separately for inspection.

Tokenizer: `Qwen/Qwen3.8-27B`, revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. File checksums are pinned in
`configs/generation.json`. Only tokenizer files are downloaded. No local generator
weights or GPU are needed. Hosted weights cannot be pinned with this API, so every
response saves the returned model, request ID, timestamp and system fingerprint
when available. Seed and low temperature do not guarantee identical fresh answers.

## Commands

Run from the repository root. The existing environment already has these versions.

```bash
uv pip install --python .venv/bin/python -r requirements-phase10.txt
.venv/bin/python -m rag_pipeline.download_generator_tokenizer
```

Put `GROQ_API_KEY` in the local environment or ignored `.env`. The runner reads
only that key; credentials are not written into artifacts. Local tests, preparation
and cache replay require no key or API calls.

Prepare all 100 prompts and token estimates under a new run ID:

```bash
.venv/bin/python -m rag_pipeline generate-answers --run-id phase10_prepare_new --dry-run
```

Generate the balanced 15-answer sample:

```bash
.venv/bin/python -m rag_pipeline generate-answers --run-id phase10_pilot_new
```

Replay saved answers into a new run without calling Groq:

```bash
.venv/bin/python -m rag_pipeline generate-answers --run-id phase10_replay_new --cache-only
```

The default `trial_001` reuses matching answers. For a genuinely new hosted trial,
add `--trial-id trial_002` to a new run; this causes new requests.

To generate all 20 questions across all five methods, use `--all-pilot` with a
new run. The prepared prompts exceed one day's published free allowance. The
runner stops with explicit quota failures when its daily budget is exhausted.
On a later UTC day, repeat the same command and run ID with `--resume`:

```bash
.venv/bin/python -m rag_pipeline generate-answers --run-id phase10_all_new --all-pilot
.venv/bin/python -m rag_pipeline generate-answers --run-id phase10_all_new --all-pilot --resume
```

Resume requires unchanged inputs, code, settings, selection and trial ID.
Completed runs cannot be overwritten. A cache hit avoids sending a completed
request again. There is a narrow unavoidable failure window if a process dies
after Groq processes a request but before the response is saved locally; its
started attempt remains visible and a resumed request may then be repeated.

## Files and error handling

| File | Purpose |
|---|---|
| `generation/answer.py` | LangChain messages, full-prompt counting, request identity and citation diagnostics. |
| `generation/client.py` | Shared ChatGroq transport and sanitized provider metadata. |
| `generation/state.py` | Persistent answer cache, quota ledger, explicit retries and writer lock. |
| `generation/run.py` | Input checks, all-prompt preparation, selected pilot, resume and artifact export. |
| `generation/audit.py` | Prompt/source/token/response integrity and saved replay comparison. |
| `generation/prompts/grounded_answer.txt` | Versioned grounding and abstention instructions. |
| `configs/generation.json` | Exact model, token limits, pilot selection and tokenizer pins. |
| `tests/test_generation.py` | Offline request, error, quota and replay tests. |

Each method's `variant_001/` contains `requests.jsonl`, an exact copy of
`generation_contexts.jsonl`, and selected `answers.jsonl`. The run root contains
configuration, environment, corpus snapshot, `preparation.json`, `usage_report.json`,
`attempts.jsonl` and a stage manifest with artifact checksums.

The answer cache and quota ledger live in
`data/outputs/rag/generation_answer_cache_v1/`. Keep them with local research
backups. The quota ledger is shared across runs and trials.

429 and transient server/connection failures have bounded retries; each attempt
is logged. Authentication errors, empty answers, model mismatches and missing
cache entries remain failures. A daily quota stops requests instead of silently
switching models. Rate-limit headers and `retry-after` are preserved where available.
Provider error bodies and exception strings are excluded to avoid exposing secrets.

Citation checks verify that `[S1]` refers to a supplied source span. They do not
prove that a claim is supported. Unknown citations, answers without citations,
the abstention prefix and output-limit truncation remain explicit diagnostics.
Do not interpret HTTP success or a syntactically valid citation as correctness.

## Development review

The first sample deliberately includes a source/version question, a context-packing
case and a personal-mark question unsupported by this teaching corpus. All five
methods receive the same questions. These observations are assistant inspection
of draft development questions, not human-approved benchmark judgments.

## Measured results

| Check | Result |
|---|---|
| Prepared requests | **100**: all 20 pilot questions across all five methods |
| Live pilot | **15/15 completed**: 3 shared questions × 5 methods; no API errors or retries |
| Remaining live requests | **85 not executed**; use `--all-pilot` across later quota windows |
| Actual usage | **35,888 input + 1,266 output = 37,154 tokens** |
| Full prepared input | **242,365 estimated input tokens**; 319,165 including output reservations and margins |
| Largest prepared prompt | **3,020 Qwen tokens**; all 100 fit without truncation |
| Prompt-count comparison | Provider and local counts match exactly for all 15 live requests |
| Answer limits | No answer reached the 512-token output limit |
| Cached repeat | **15/15 cache hits**, identical prompts, evidence and raw responses; zero new API calls |
| Tests | **129 pipeline + 17 cleaning = 146 passed** |
| Inputs | **122 prepared files** and the complete Phase 9 artifact inventory remain unchanged |
| Timings | **428.5 seconds** for the recorded live run, largely quota pacing; **13.46 seconds** summed client request latency |
| Human review | Pilot annotations remain drafts; no human approval is claimed |

Read these first:

- [Answer review with exact evidence and examples](../../data/outputs/rag/phase10_generation_review_20260927.md).
- [Independent audit and saved replay check](../../data/outputs/rag/phase10_generation_pilot_20260927_audit.json).
- [Provider usage report](../../data/outputs/rag/phase10_generation_pilot_20260927/usage_report.json).

Main artifacts: `data/outputs/rag/phase10_generation_pilot_20260927/`.
Cached repeat: `data/outputs/rag/phase10_generation_repeat_20260927/`.
The earlier `phase10_prepare_20260927/` is a no-call preparation snapshot made
before final transport checks; use the main run's 100 requests as the final version.

To audit a newly named run and its repeat:

```bash
.venv/bin/python -m rag_pipeline.generation.audit --run-id phase10_pilot_new --repeat-id phase10_replay_new
```

The audit writes a new adjacent `_audit.json`; it refuses to overwrite a previous
audit. It checks exact source slices/pages, unchanged Phase 9 inputs, complete
messages and shared decoding parameters, request identities, actual token usage,
response/cache linkage and identical replay. It does not score answer correctness.

## What we learned from the sample

- **Structure-aware / pilot_001:** The supplied UCSC passage supports the cited
  answer: 15 lecture hours or 30 practical hours, or an equivalent combination.
- **Fixed / pilot_004:** Phase 9 found the gold passage at rank 10 but omitted it
  from the packed context. Generation abstained. Structure-aware retained the
  passage and answered the two thresholds: 40% continuous assessment and 80%
  attendance. This traces a missing answer to context selection.
- **Late / pilot_019:** The supplied teaching material does not establish the
  user's personal mark; the model declines to invent one.
- **Fixed / pilot_001:** The abstention includes unsupported university-name
  expansions. An abstention can still contain ungrounded claims.

No partially answered multi-part response was observed in these three questions;
the review shows the actual missing-evidence abstention instead of inventing a
partial-answer example. Four responses gave direct answers and eleven began with
an abstention. These counts describe behavior, not final accuracy or faithfulness.

### Remaining limits

The first citation diagnostic recognizes individual `[S1]` labels. Grouped forms
such as `[S1, S2]` remain in the raw answer and full source map, but are not fully
parsed by that diagnostic. Phase 11 citation scoring must handle grouped forms
and review semantic support, not reuse the simple label count as a metric.

Groq returned three different system fingerprints during the pilot. They are
preserved per response; the API does not expose enough information to establish
what changed behind them. The saved repeat proves replay integrity, not stable
hosted model weights or deterministic fresh generation.

The 1,536-token packing limit is still measured with Jina. Qwen prompt counts
include all evidence, wrappers and instructions under a separate common limit.
Finalize the primary evaluation budget convention before benchmark scoring.

**Next: Phase 11 — controlled evaluation, once the team has supplied and reviewed
the larger benchmark and frozen the evaluation protocol.** Finish the remaining
pilot questions over available free-tier days when needed for development.

