# Experiment protocol — Phase 2 draft

**Status: needs team review; no held-out benchmark has been frozen.**
Prepared 20 September 2026. These rules must be agreed before final test scoring.
No results have been used to choose these settings.

**Workflow amendment:** Vithusan chose to use the 20 pilot drafts while building
Phases 3–10, and supply the larger reviewed benchmark later. Final benchmark
authoring/freezing is a gate for Phase 11 evaluation, not for starting Phase 3.
The pilot stays development-only. Define the question categories/rubrics now;
write final questions from source evidence without selecting cases to favor a method.

## Research and scope

Compare fixed-size, sliding-window, structure-aware, semantic, and late chunking
on the same 60-document educational corpus. Measure retrieval, grounded answers,
failure causes, and resource cost. Member folders describe collection ownership;
all five methods search the combined corpus.

The Phase 1 corpus identity is
`dc3eeded7225fc31c96a3d27a2231b241783a377d98cca3443dc3eb0ec7260cf`.
Keep text unchanged. Explicitly excluded and blank pages are barriers; ordinary
nonblank page boundaries may be crossed. Missing images and broken extraction
belong in extraction diagnostics, not primary chunking scores.

Primary questions are English. Preserve other-language source text but make no
multilingual performance claim. Qualify university/year/course where relevant.

## Resources and spending

- Vithusan chose **this computer for embeddings**: 16 logical CPUs, about 14.93 GiB
  RAM and an NVIDIA RTX 3050 Laptop GPU with 4,096 MiB physical VRAM. The sandbox
  hides CUDA access; outside it, PyTorch recognizes the GPU and a tiny tensor
  calculation and a narrow 2,048-token embedding probe passed. Full-corpus GPU
  memory/runtime is not yet measured.
- Vithusan chose **free LLMs**: the paid generation/evaluation budget is **zero**.
- Embeddings run locally. No paid service or credit-card-backed fallback is enabled.
- Vithusan selected **GroqCloud's free tier** as the preferred generation provider.
  The exact model remains open: current Groq documentation retires Llama 3.1 8B
  and Llama 3.3 70B for free/developer tiers from 16 August 2026. Select a supported
  free-tier model and verify account limits before generation. No API call has
  been made. See [Groq deprecations](https://console.groq.com/docs/deprecations)
  and [free-tier limits](https://console.groq.com/docs/rate-limits).
- Use deterministic retrieval metrics and human rubric scoring as the evaluation
  foundation. RAGAS/local judging may supplement it after feasibility/calibration;
  installing RAGAS does not make its default hosted model free.
- Proposed pilot cap: 30 minutes per method and an 8 GiB process-RAM ceiling.
  If exceeded, stop and revise all affected settings before scoring. A full-run
  runtime limit still needs Vithusan's decision after realistic timing estimates.

## Embedding decision and controlled comparison

Selected common model: `jinaai/jina-embeddings-v2-small-en`, pinned model and
custom-code revisions in [model_feasibility.md](model_feasibility.md).
The CPU capability probe passed at **2,048 total input tokens**. Do not assume
the advertised 8,192-token configuration was tested on this machine.

The tested baseline uses CPU FP32, four inference threads, evaluation mode, no query/document prefixes,
and L2-normalized 512-dimensional vectors. For this experiment, mean-pool **content
tokens only**, excluding padding and special tokens, for queries, early chunks,
and late spans. This is a documented adaptation of the stock pooling recipe;
all methods must use it consistently. The complete-window early/late check passed.
Tokenizers normalize internally for encoding; canonical source text remains exact.
CUDA FP32 inference has now passed the 2,048-token LangChain probe on the RTX 3050.
Use CUDA FP32 and batch size 1 as configured; explicit CPU runs remain possible.
Record the device/precision policy for every run. Do not compare CPU and GPU
timings as if chunking alone caused the difference.

## LangChain and readable implementation

LangChain is required for the shared pipeline interfaces. `PreparedCorpusLoader`
now implements `BaseLoader` and returns LangChain `Document` regions from the
validated prepared corpus. `JinaEmbeddings` implements the common embedding API,
and a `ChatGroq` factory is ready for generation setup. Use LangChain retrieval
and prompt interfaces as those phases are implemented. Compatible packages are
pinned to the existing LangChain 0.3 family; do not blindly upgrade integrations.

Keep exact-offset validation and late token pooling as small documented helpers.
A general-purpose loader or text splitter does not establish our research span
invariants automatically. Read the prepared JSONL and metadata rather than
re-extracting the original PDFs for each method. Normal LangChain wrappers must
not silently change pooling, truncate text, or discard page/region metadata.

Late uses the fixed baseline's exact spans, inside deterministic encoder windows
of at most 2,046 content tokens plus two special tokens. A target chunk must fit
inside its assigned window. Window overlap/ownership is finalized and tested in
Phase 8; no chunk can be split or counted twice at a window seam.

## Small development grid

Use the pinned Jina tokenizer for chunk measurements. Numbers below are content
tokens; retokenize each actual substring and account for specials before encoding.
Start with **256 content tokens**, not a claim that 256 is optimal.

| Method | Proposed development variants | Shared rules |
|---|---|---|
| Fixed | 128, 256, 512 tokens | No overlap; retain tail. |
| Sliding | Same three caps; 25% overlap | 32/64/128-token overlap; zero-overlap equivalence is a correctness check. |
| Structure | Same three caps | Prefer section/paragraph boundaries; sentence then token fallback; preserve all eligible text. |
| Semantic | Same three caps | Neighbor-sentence cosine distance; split at the 90th percentile within a region, computed without labels; minimum target cap/4, hard maximum cap. |
| Late | Paired with each fixed variant | Same exact spans/model/pooling; longer contextual encoding only. |

Initial semantic rule: adjacent single sentences, using the common model and
content-only pooling. If fewer than three units exist, pack under the cap. Split
oversized units before scoring, retain offsets, break equal-score ties in source
order, and let the cap override any semantic decision. Record forced breaks and
the realized threshold. A percentile grid of 80/90/95 is an optional sensitivity
study, excluded from the primary tuning budget unless all methods receive a
documented comparable search budget. Concrete parsing/fallback details are tested
in Phase 6 before freezing configurations.

First run matched 256-token comparisons. Then evaluate at most three primary
variants per method on development questions. Choose by primary evidence coverage;
tie-break by lower indexed-token count, then lower runtime. For the primary late
comparison use the fixed method's selected size, even if another late size scores
better. Report per-method independently tuned comparisons only as secondary.
Do not tune with held-out answers or test retrieval scores.

## Benchmark design and review

Initial artifacts: **20 pilot drafts**, 18 answerable, one unanswerable candidate,
one ambiguous case. They cover definitions, rules, exceptions, a procedure,
table values, and a two-page calculation. These are illustrative development
material, not a representative final benchmark or 20 human-approved annotations.

Proposed scored target: **40 development + 80 held-out test**. A suggested mix is
36 answerable/4 unanswerable in development and 72 answerable/8 unanswerable in
test. Aim for both handbooks and lectures and at least six answerable test
questions in each main category: definition, explanation, procedure, eligibility,
exception, table lookup, multi-passage. Review feasibility rather than forcing
unsupported questions to fill a quota. Sparse category results are exploratory.

1. Draft from PDFs and prepared passages, never from a chunker's output.
2. Record atomic facts, exact evidence spans, alternative valid locations, and a
   rubric including numerical conditions and exceptions.
3. Visually inspect each original PDF reference and verify prepared text preserves
   the fact. Assistant inspection and independent human review are separate fields.
4. A named teammate reviews each scored question; resolve requested changes before
   approval. Any edit invalidates the previous annotation hash/review.
5. Assign paraphrases and near-duplicate source versions to the same split family.
   Pilot families remain reserved for development even if pilot rows are removed.
   The current seven-document family map is incomplete for the rest of the corpus;
   extend it through manual title/version/content review before selecting test data.
6. Related documents may remain retrieval distractors in the full corpus; grouping
   prevents tuning/test **question-label** leakage, not document removal.
7. Put ambiguous and extraction-dependent questions in diagnostics, outside primary
   scores. Never label a question unanswerable because one retriever failed.
8. Finalize the count, category balance, source groups, protocol, and reviews; run
   validation and freeze hashes before test scoring. Keep exposed pilot data out
   of test. Do not claim a statistical power guarantee for 80 test questions.

The validator checks supplied labels and recorded review identity; it cannot prove
that a human actually read a PDF or discover every near-duplicate automatically.

## Retrieval, evidence, and context

Use exact cosine search over normalized vectors, common query encoding, no gold
source filters, descending similarity and ascending chunk ID to resolve ties.
Record K = 1, 3, 5, 10 and all retrieval scores.

An atomic evidence unit has one or more alternative contiguous source spans.
Coverage requires **all non-whitespace characters** of at least one alternative
to appear in the union of selected spans from the same document/version. Ignore
only whitespace for coverage; do not award a hit for merely sharing a page.
Separate facts across passages are separate required units. Alternative spans
must each independently support the same fact.

Context assembly proposal: process the top 10 in rank order, subtract source-span
overlap already included, preserve the remaining exact spans, and stop at a common
**1,536 Jina content-token evidence budget**. Skip a candidate whose new evidence
would exceed the budget and try the next; do not silently cut a rule mid-sentence.
Log every skipped candidate, exact included spans, token counts and citation wrappers.
One candidate may yield multiple pieces after overlap removal; record those pieces
explicitly, never pretend their concatenation is one original source span.

This budget measures retrieval evidence using a common tokenizer. It is **not**
the future generator's token count. The generator and its actual input/output
capacity must be verified before protocol freeze for answer experiments. If 1,536
evidence tokens plus prompt/wrappers and an answer do not fit, reduce the common
budget for all methods before final evaluation and record the amendment.

## Outcomes and denominators

| Outcome | Exact draft rule |
|---|---|
| **Primary: complete evidence in budgeted context** | Fraction of answerable questions for which every evidence unit is covered after context assembly. |
| Evidence Recall@K | For each answerable question, covered units / required units in union of top-K spans; macro-average across questions. |
| Complete-evidence success@K | Fraction of answerable questions with all units covered in top-K union. |
| MRR@K | Reciprocal rank of first single chunk completely covering at least one unit alternative; zero when none does. Average over answerable questions. |
| Answer correctness | Human rubric facts satisfied / required facts, plus a separate fully-correct indicator requiring all facts and no contradictory claim. |
| Faithfulness | Supported answer claims / checkable answer claims against actual supplied context; record unsupported claims and denominator. No checkable claims: N/A. |
| Citation support | Supported cited claims / claims with citations; separately report uncited checkable claims. |
| Response relevance | Human ordinal 0 off-topic, 1 partly addresses, 2 directly addresses; report category counts and mean. |
| Abstention | On reviewed unanswerable questions, rate of explicit insufficient-evidence responses without an invented answer; report unsupported-answer rate separately. |
| Efficiency | Chunking/embedding/query/generation seconds, indexed tokens, vectors, storage bytes, peak process RAM where measured; cold/cached timing separate. |

Unanswerable/ambiguous/diagnostic questions do not enter evidence recall or MRR.
An abstention on an answerable question counts as answer incorrect, not a success.
Empty retrieval for an answerable question scores zero. Pipeline crashes/timeouts
count as unsuccessful primary trials; report both conservative all-question
results and completed-run coverage. Do not encode evaluator failures as valid
zero judgments: mark missing and show denominator/coverage.

Compare methods on the same questions. Proposed uncertainty: 2,000 bootstrap
resamples of source/question families with seed 42, computing paired differences
and 95% percentile intervals. Family grouping matters because related questions
are not independent. With few families, intervals/category findings are descriptive;
avoid unsupported claims of significance or a universal winning chunker.

## Generation, repeats, and failure analysis

Generator selection remains open. Freeze exact model/quantization/revision,
prompt, tokenizer, context cap, answer limit and decoding before answer scoring.
Start with one generation pass per method/question using the same low-randomness
settings; hosted responses are not guaranteed deterministic. If runtime
permits variability analysis, preregister three repeats for the same balanced
development-selected subset across all methods, recorded separately. No paid
judge fallback. Human review remains the reference for answer quality.

Store the chain: PDF → prepared evidence → chunk spans → retrieved ranking →
assembled context → answer → judgment. Use the PLAN taxonomy: extraction,
boundary fragmentation, missing reference context, table/list separation,
overflow, retrieval miss, wrong source/version, redundant overlap, context loss,
generation error, benchmark/evaluator error. Allow multiple contributing labels
and an explicitly uncertain primary cause. Review the same questions across
methods; keep diagnostic interventions separate from main scores.

## Development preparation and deferred evaluation gates

- [ ] Vithusan reviews this protocol, resource limits, benchmark workload and primary outcome.
- [x] Add the LangChain adapter and explain the preserved source metadata.
- [ ] Team corrects and cross-reviews the pilot before treating its scores as reviewed evidence; draft runs may proceed for debugging.
- [ ] Before Phase 11: assign remaining source/version families and author reviewed development/test questions.
- [ ] Before Phase 11: record and validate final splits, review hashes and protocol choices.
- [ ] Before final test scoring: freeze benchmark; update the plan with actual completion evidence.

Phase 3 was subsequently authorized and implemented using the 256-token baseline.
See [the Phase 3 handover](phase3_completion.md); these development runs do not
freeze the final benchmark or select settings using test scores.
