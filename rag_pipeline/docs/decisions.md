# Implementation decisions

## Phase 1 — 19 September 2026

| Decision | Reason and consequence |
|---|---|
| JSONL prepared text is the input contract | It matches TXT and carries physical page provenance. Detailed JSON is validated and can supply extra structural metadata. |
| Inputs remain read-only | The legacy validator now accepts an external report path; pipeline reports go under ignored experiment outputs. |
| Excluded and blank prepared pages are barriers | A blank extraction may hide an image or missing passage. Every method will use the same conservative region boundaries. |
| Ordinary nonblank page breaks are allowed within a region | Page layout alone should not force every method to restart. |
| Source text is never normalized in the loader | Exact slicing preserves character offsets and repeated occurrences. |
| Loaded headings use document offsets | The conversion is done once in the loader while original page-local metadata remains untouched. |
| Token fields are nullable in the Phase 1 schema | Model/tokenizer selection belongs to Phase 2. Later chunkers must provide measured counts and tokenizer identity. |
| Minimal runtime has four pinned dependencies | Phase 1 can run without embedding/generation packages. The full requirements file includes the minimal file. |
| New run IDs are mandatory | A previous successful or failed inspection cannot be silently overwritten. |
| Both Git revision and source hashes are recorded | Experimental code may be uncommitted; a commit ID alone would not identify the code actually run. |

## Observed corpus boundaries

- 60 documents; 6,215 physical page records.
- 44 explicit exclusions, retained in metadata but absent from retrievable text.
- 172 included pages have empty/whitespace-only prepared text.
- 169 continuous eligible regions, each containing one or more nonblank pages.

These regions constrain future chunks; they do not modify `final_data` and are
not themselves chunking outputs. The gap policy forms part of the corpus identity.
Changing it later requires a recorded configuration/protocol change and a consistent
rerun for every method.

## Phase 2 — 20 September 2026

- Vithusan selected this computer only; no Colab/GPU assumption.
- Vithusan selected free LLMs; paid generation and judge budgets are zero.
- CPU feasibility passed for pinned Jina v2 small English at 2,048 total tokens.
  Use the same candidate for all methods, subject to development quality checks.
- Mean-pool content tokens only in every query/early/late path. This documented
  adaptation differs from stock pooling of all attended tokens.
- Start development with a 256-token baseline and proposed 128/256/512 grid.
- Twenty pilot questions are drafts, not human-approved ground truth. Pilot
  source families remain development-only even after pilot records are removed.
- Human review, final split authoring and protocol agreement are required before
  benchmark freeze. The current phase remains in progress.

See [Phase 2 status](phase2_status.md), [protocol](experiment_protocol.md) and
[model feasibility](model_feasibility.md) for evidence and limits.

## Decisions still pending

- Confirmation of development retrieval quality and final chunk configuration.
- Reviewed benchmark composition, source-family grouping and held-out splits.
- Lead review of draft primary metrics, context budget, grids and runtime limits.
- Exact free generator/judge models, input capacity and practical runtime.

## Vithusan's follow-up decisions — 20 September 2026

- LangChain is required for shared pipeline interfaces. The earlier exploration
  script uses `PyPDFLoader`; Phase 1/2 currently use direct Python and PyTorch.
  Add a thin LangChain interface without discarding research-specific provenance.
- Use 20 pilot drafts for development through answer generation. The larger
  team-authored benchmark is deferred until before Phase 11 final evaluation.
  This supersedes the earlier requirement to finish all 120 questions in Phase 2.
  It does not approve individual annotations or turn pilot runs into final scores.
- GroqCloud free tier is the preferred generator provider. Exact supported model
  remains pending; old Llama IDs are not assumed available. Paid-call budget stays zero.
- GPU use is allowed. `lspci` identifies RTX 3050 Mobile; unrestricted `nvidia-smi`
  reports 4,096 MiB and driver 595.91.07. Outside the sandbox PyTorch CUDA succeeds
  and a four-element tensor calculation passed. Earlier CPU-only observations
  reflected sandbox restrictions, not absent GPU hardware. No full GPU model
  probe or driver changes were performed.
- Keep Jina v2 small as the provisional shared embedding choice. Its suitability
  for this corpus still requires retrieval checks; no universal-best claim is made.
- Use clear names, short functions, useful comments and teaching examples. Code
  remains AI-assisted; readability and team understanding are the standard.

## Pre-Phase 3 implementation completed — 20 September 2026

- Added `PreparedCorpusLoader` using LangChain `BaseLoader`/`Document`; its full
  corpus run returned 169 eligible regions from 60 sources with unchanged text.
- Added `JinaEmbeddings` using LangChain's `Embeddings` interface. CPU and CUDA
  probes passed, including token pooling, batch padding and query consistency.
- Selected model configuration now defaults to CUDA FP32, batch size 1, and a
  2,048-total-token cap. CUDA peak PyTorch allocation was about 667 MiB; future
  full-corpus resource use is not inferred from this small probe.
- Added a `ChatGroq` client factory, with no API invocation. Exact model ID stays
  unset until a current free-tier model is selected. Configuration records the
  integration's actual near-zero temperature, `1e-8`.
- Added LangChain/Groq dependency pins compatible with the existing 0.3 stack;
  the dependency check passed without upgrading unrelated installed packages.
- Phase 3 development prerequisites are ready. Phase 2's larger benchmark work
  is explicitly deferred until before Phase 11 under Vithusan's revised workflow.

## Phase 3 — 20 September 2026

- Implement the agreed 256-content-token fixed-size baseline with zero overlap.
  Encoder input counts include two special tokens; full chunks therefore use 258.
- Use LangChain `TextSplitter` and the pinned Jina fast tokenizer. Override source
  slicing and metadata updates to preserve exact offsets, case, whitespace and
  repeated passages. Never reconstruct canonical chunk text by decoding tokens.
- Retokenize each candidate substring before accepting it; back off when cutting
  a WordPiece changes its independent token count. Keep shared-character subtokens
  together and record adjustments. This remains a token-based boundary policy.
- Retain all eligible characters and short region tails. The 650 characters
  outside eligible regions are only whitespace around existing barriers.
- Run tokenization on CPU without loading embedding weights. GPU inference stays
  in the embedding stages; no model or precision decision changed.
- Preserve new-run-only output protection and hash nested artifacts. Independent
  read-back verification confirmed exact spans, coverage, page references, token
  caps, repeatability and 122 unchanged prepared-input hashes.
- The full corpus produces 8,627 chunks. This is segmentation validation, not a
  claim of retrieval quality or optimal chunk size. See the
  [Phase 3 handover](phase3_completion.md) for the audit and boundary examples.

## Phase 4 — 20 September 2026

- Implement the protocol's initial sliding-window setting: 256 content tokens,
  64 source-token overlap (25%), nominal stride 192. No parameter tuning was done.
- Keep the fixed and sliding methods in separate folders, with shared exact-offset
  window logic and artifact writing. The LangChain splitter interface is preserved.
- Measure overlap in the original region token stream and separately record its
  exact characters and standalone substring token count. These token counts are
  not assumed interchangeable at WordPiece boundaries.
- Anchor the next window to the accepted previous end minus requested overlap.
  Round starts forward to safe character boundaries; reduce overlap if necessary
  to add new evidence. Both starts and ends must advance; stop at the first window
  that reaches the region tail. Record any overlap reductions.
- The full run has 11,419 chunks and 11,250 adjacent pairs, each sharing exactly
  64 source tokens. No overlap reductions occurred. Tokenization remains CPU-only.
- Report repeated characters and indexed-to-unique token ratios separately from
  unique source coverage. More indexed text is a cost, not evidence of better answers.
- Keep the 256-token fixed baseline unchanged, verified against its saved sample
  records and full-corpus source spans. See the [Phase 4 handover](phase4_completion.md).


## Phase 5 — 21 September 2026

- Keep the 256-content-token cap and zero overlap. Use the same pinned Jina
  tokenizer and LangChain TextSplitter interface as the prior token-based methods.
- Resolve prepared headings, detailed font candidates and usable PDF outlines
  against exact full source lines. Reject ambiguous or unmatched hints; deduplicate
  accepted spans and retain their contributing sources.
- Infer levels from outline, numbering, then prepared levels; unknown levels become
  level-2 siblings. Reset paths at region barriers, coalesce adjacent title lines,
  and treat a repeated current heading as continuation.
- Pack paragraphs within sections. Preserve fitting table blocks and list items;
  use visible rows, conservative sentence boundaries and token windows for larger
  units. Save fallback unit spans and never reconstruct uncertain table facts.
- Preserve tokenizer-ignored PDF bullet glyphs by attaching tokenless units to a
  neighbor. Do not alter the prepared corpus or silently drop those characters.
- The completed full run is phase5_structure_full_v2_20260920; the earlier full
  directory retains the failed tokenless-glyph diagnosis. New run IDs remain mandatory.
- The final corpus output has 13,377 chunks, mean 163.28 content tokens and complete
  eligible-text coverage. Both full runs and the independent audit agree; all 95
  tests pass. Earlier fixed/sliding sample records remain unchanged.
- Heading paths remain inferred metadata, not added embedding text. The review pack
  includes a questionable contents-like ancestor label to make the limitation visible.
  See [Phase 5 handover](phase5_completion.md) for counts, examples and commands.
## Phase 6 — semantic boundary decisions (21 September 2026)

| Decision | Reason and behavior |
|---|---|
| Reuse the pinned Jina model through LangChain `Embeddings` | Boundary scoring uses the same content-only mean pooling and L2 normalization planned for retrieval. |
| Use a LangChain `TextSplitter` with explicit source offsets | The research needs exact slices, blank/excluded-page barriers and detailed boundary traces. |
| Score adjacent source units | Prose uses sentences; lists use items; recognizable tables use physical rows. Oversized units split before encoding. |
| Region-local 90th percentile, linear interpolation, strict `>` | A fixed development rule without benchmark labels; ties stay together unless size forces a cut. |
| 64-token semantic minimum target, 256-token hard cap, no overlap | Size-forced cuts and tails may be shorter; every such chunk is flagged. |
| Cache vectors in ignored SQLite storage | Exact text, model/code hashes, configuration, device and package versions identify the cache. Checksums detect corruption; committed rows survive interruption. |
| Record FP32, seed 42 and deterministic Torch mode | Cache-backed repeats reproduce the scores; bitwise equality across different hardware is not claimed. |
| Keep boundary embeddings separate from final retrieval embeddings | Phase 6 vectors describe scoring units. Phase 7 will embed the resulting chunks. |

The 90th-percentile choice remains untuned. Final configuration selection requires
the reviewed development questions before Phase 11. Extraction fragments can
produce misleading similarities; saved diagnostics keep those cases visible.

## Phase 7 — shared chunk embeddings (21 September 2026)

- The first four chunkers feed one LangChain-compatible encoder and one pinned
  Jina configuration: 512 dimensions, FP32, content-only mean pooling and L2
  normalization. Source metadata is not prepended to the text.
- Phase 7 uses **batch size 8**, configured in `configs/embedding.json`. Both
  batch-one and batch-eight samples passed on the local RTX 3050. Across 2,845
  sample vectors, the largest element difference was below 1.8e-7. The setting
  overrides the adapter default for this stage and applies to all four methods.
- Source chunk artifacts stay immutable. A new embedding run stores copied
  chunks, NumPy vectors, row maps, statistics and manifests for each method.
- A separate final-chunk SQLite cache uses exact text and the complete encoder
  identity. Identical text can reuse a vector across methods; every source chunk
  still has its own row. Phase 6 scoring-unit vectors remain separate.
- Resume is explicit, validates code/config/input identities, verifies completed
  methods and rebuilds incomplete outputs from committed cache rows. A process
  lock prevents concurrent writers. Completed runs require a new ID for repeats.
- Cache hits, misses, inference time and preflight time are recorded separately.
  Method order can affect cache hits, so observed timings are not a cold-runtime
  ranking. Batched fresh inference can differ slightly numerically; saved-vector
  repeats are required to be byte-identical.

## Phase 8 — late contextual embeddings (21 September 2026)

- Reuse Phase 7 fixed-size chunks exactly. Every late record links its original
  early chunk and vector row; source text, spans and pages stay unchanged.
- Reuse the pinned LangChain Jina adapter's token-output method, CUDA FP32,
  content-only mean and L2 normalization. Batch one bounds long-input GPU memory;
  Phase 7's batch-one/eight probe documents the small numerical batch effect.
- Five chunks normally own a window, with one whole neighboring chunk on each
  side. Windows remain inside eligible regions and below 2,048 total tokens.
  Every target chunk has one owner; the token-limit fallback reduces ownership
  and then halo context without splitting targets.
- Verify contextual offsets before pooling. Neighboring tokens affect attention
  but never enter a target chunk's mean. Token counts can differ when a word-piece
  is encoded in context; both independent and pooled counts are retained.
- Cache complete-window pooled results using exact context, target spans and
  model/code identity. Commit a whole window atomically. Recovery uses a new
  run ID and the same cache; completed artifacts stay immutable.
- Measure whole-window early/late parity and compare chunks that appear in two
  overlapping windows. Vector changes and window sensitivity are diagnostics,
  not claims of better retrieval or answer accuracy.
- Keep the 2,048-token experiment limit explicit: long handbooks receive windowed
  contextual encoding, not whole-document encoding. No additional model or API
  service is introduced. See [Phase 8 handover](phase8_completion.md).


## Phase 9 — shared indexing and retrieval (27 September 2026)

- Build one exact FAISS IndexFlatIP per method through LangChain's FAISS store,
  preserving original vector rows and source chunks. Save explicit JSON mappings
  and verify reloaded vectors before searching.
- Encode each pilot question once with the shared pinned CUDA FP32 Jina adapter.
  Check model/tokenizer/pooling compatibility across independent and late vectors.
  Query caches remain separate from document and semantic-boundary caches.
- Search every eligible document, without reference-source filters. Return
  descending normalized-vector inner product, breaking exact ties by chunk ID.
  Retrieve all scores before top-K selection to resolve ties at the cutoff.
- Use K=1,3,5,10 and the common 1,536 Jina-content-token evidence budget. Subtract
  only already included source spans; preserve disjoint pieces explicitly. Skip
  oversized candidates and try the next. Count citation wrappers separately.
- Keep raw retrieval, assembled contexts and post-retrieval pilot evidence checks
  in separate files. Reference answers and evidence labels never influence search.
- Use new run IDs after interruption, reusing committed query vectors. Cached
  repeats produced identical indexes, rankings, contexts and diagnostic records.
- Treat the existing 20 questions as development drafts, not human-approved final
  evidence. Human review and larger benchmark freezing remain due before Phase 11.


## Phase 10 — grounded generation (27 September 2026)

- Use Groq `qwen/qwen3.8-27b` through LangChain `ChatGroq`; the live account's
  model list did not include the proposed Llama model. Disable reasoning output
  with `reasoning_effort=none`. Use one generator/configuration across methods.
- Keep Phase 9 source evidence unchanged. Count the complete prompt with the
  pinned Qwen tokenizer/template, reserving 512 output tokens and a 256-token
  margin under an 8,192-token total limit. Preserve actual provider usage too.
- Prepare all 100 prompts. The first live sample is pilot_001, pilot_004 and
  pilot_019 across every method: 15 requests. All-question generation can continue
  over multiple days; its prepared prompt tokens exceed the daily free allowance.
- Use a shared persistent quota ledger and answer cache. The cache identity
  includes full messages, model/decoding configuration, tokenizer, implementation
  hash and trial ID. New trial IDs require new requests; cache replay is not an
  independent measurement of hosted-model determinism.
- Preserve failures and every retry. Do not switch models or truncate evidence
  silently. Resume only with unchanged inputs/settings/code; never overwrite a
  completed run. Rate/account quota failures remain visible in answer records.
- Read only GROQ_API_KEY from the local environment/ignored .env; do not save it
  or raw SDK error bodies. Disable external LangSmith tracing in this runner.
- Citation labels link answers to exact source spans; automatic label checks do
  not establish factual support. Pilot draft annotations remain unapproved.

See [Phase 10 handover](phase10_completion.md) for actual results and limitations.


## Streamlit explorer — 28 September 2026

- Added a Python inspection interface between Phases 10 and 11 at Vithusan's request.
- Read saved experiment artifacts using recorded source paths and checksums; keep
  algorithm implementations in `rag_pipeline/` and display code in `research_gui/`.
- Compare methods by document character spans, not by method-specific chunk numbers.
- Show unavailable answers explicitly and separate retrieved ranks from packed evidence.
- Display current reference annotations only in a human-review panel. Export notes
  without modifying benchmark approvals or completed outputs.
- This release is an inspection UI. Pipeline execution and background-job controls
  remain a possible later extension. Viewing screens makes no model/API calls.

See the [UI guide](../../research_gui/README.md) and local verification artifacts
under `data/outputs/rag/ui_verification_20260928/`.

## Phase 11 provisional implementation — 6 October 2026

The user authorized evaluation implementation while benchmark review continues.
The new offline command accepts the existing pilot only and requires
`--provisional`. It preserves the separate 153-question intake as drafts.
Final test scoring remains unavailable until benchmark/protocol/configuration
freezing and the remaining experiment workflow are implemented and verified.

Exact source evidence defines retrieval relevance. Human rubric judgments define
answer correctness; citation label existence alone does not establish support.
Missing answers, unreviewed judgments and failed judge outputs remain visible.
The optional pinned RAGAS adapter requires explicit clients and makes no calls
from the offline command. Its live judge/budget workflow remains pending.

See [Phase 11 status](phase11_status.md) for outputs, commands and limitations.
