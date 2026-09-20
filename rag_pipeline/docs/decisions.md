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
