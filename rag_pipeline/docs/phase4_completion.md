# Phase 4 — Sliding-window chunking

Implemented on 20 September 2026, using the Phase 2 protocol's starting setting:
**256 content tokens per chunk, with 64 source tokens shared between neighbors**.
The nominal stride is `256 - 64 = 192` tokens. A full chunk still has 258 encoder
input tokens after adding the two Jina special tokens.

## The simple idea

Fixed-size chunking starts each chunk where the previous one ends. Sliding-window
chunking starts a little earlier, repeating some text to give the next chunk context.

```text
Fixed:    [tokens   0–255][tokens 256–511][tokens 512–767]
Sliding:  [tokens   0–255]
                    [tokens 192–447]
                              [tokens 384–639]
```

The ranges above illustrate normal token windows, not character positions. The
implementation records exact source character positions for every saved chunk.
It keeps short tails and stops as soon as a window reaches the end of a region.
It does not add another window containing only an already-covered tail.

## What was implemented

| File | Purpose |
|---|---|
| `chunking/sliding_window/chunker.py` | LangChain `SlidingWindowSplitter`, settings checks and measured overlap metadata. |
| `chunking/sliding_window/run.py` | Entry point for the method. |
| `configs/sliding_window.json` | The 256-token / 64-token-overlap configuration. |
| `common/token_windows.py` | Exact token/character boundary rules shared with fixed-size chunking. |
| `common/chunk_run.py` | Shared output writer, coverage checks, statistics and manifests. |
| `tests/test_sliding_window.py` | Overlap, tail, Unicode, source mapping, zero-overlap and failure checks. |

The splitter inherits the fixed method's LangChain interfaces and page metadata
handling. Both methods use the same pinned Jina tokenizer and preserve exact source
substrings. The fixed method now calls the shared helpers; its saved baseline spans
remain unchanged. No additional dependencies were needed.

## Exact overlap rules

- Overlap is measured in the **original region's token stream**. With normal
  256-token windows, the next start is 192 tokens after the previous start.
- After accepting a chunk, propose the next start at its token end minus 64.
  Do not split subtokens that refer to the same source character.
- Recount every chunk as an independent encoder input. A partial WordPiece word
  can tokenize differently outside the original region; back off the end when
  needed to keep the 256-content-token cap.
- Starts and ends must both advance. If a candidate adds no new evidence, reduce
  its overlap until it does. The requested overlap is a maximum under these safety
  rules, and any reduction is recorded. **No reduction occurred in this full run.**
- `overlap` must be an integer satisfying `0 <= overlap < chunk_size`.
- Leading/inter-token/trailing whitespace follows the same ownership rule as the
  fixed baseline. Shared spans repeat the original whitespace exactly.
- Excluded and blank pages remain barriers. Overlap never crosses a region or
  document boundary. Normal nonblank page boundaries may occur inside a chunk.

Each chunk's `method_metadata` records:

| Field | Meaning |
|---|---|
| `requested_overlap_tokens` | Configured overlap, 64 in this run. |
| `nominal_stride_tokens` | Configured size minus overlap, 192. |
| `overlap_with_previous_region_tokens` | Actual shared tokens measured in the original region. |
| `overlap_with_previous_characters` | Length of the exact shared source substring. |
| `overlap_text_content_tokens` | Token count when that shared substring is encoded by itself; this can differ from the source-stream count. |
| `overlap_reduced` | Whether actual overlap is smaller than requested; false on a region's first chunk. |

Existing fields include content/input counts, region token positions and end-boundary
backoff counts. `token_count` includes special tokens; `content_token_count` does not.

## Commands

Run from the repository root with the existing environment and local tokenizer
bundle. CPU tokenization is enough; this stage loads no embedding weights and
makes no API calls.

```bash
# First document by source name from each member.
.venv/bin/python -m rag_pipeline chunk-sliding --run-id my_sliding_sample --sample

# All 60 documents.
.venv/bin/python -m rag_pipeline chunk-sliding --run-id my_sliding_full
```

Use a new run ID. Existing successful or failed directories cannot be overwritten.
The command also accepts `--config`, `--method-config` and `--model-config`.
For another development setting, copy `configs/sliding_window.json`, edit its
size/overlap, and pass the copy with `--method-config`. These defaults have not
been selected using final test results.

### Use it through LangChain

```python
from rag_pipeline.common.langchain_loader import PreparedCorpusLoader
from rag_pipeline.common.tokenization import load_tokenizer
from rag_pipeline.chunking.sliding_window.chunker import SlidingWindowSplitter

regions = PreparedCorpusLoader().load()
splitter = SlidingWindowSplitter(load_tokenizer(), chunk_size=256, overlap=64)
chunks = splitter.split_documents(regions[:1])

print(chunks[1].metadata["method_metadata"])
```

Use the CLI for saved research artifacts, including IDs, manifests and statistics.

## Outputs and results

The full output is under:

```text
data/outputs/rag/phase4_sliding_full_20260920/
    run_config.json
    corpus_snapshot.json
    environment.json
    stage_manifest.json
    sliding_window/variant_001/
        chunks.jsonl
        chunk_stats.json
        inspection_examples.json
        inspection_report.md
        overlap_examples.json
        overlap_report.md
        stage_manifest.json
```

The later embedding stage will read `chunks.jsonl`. Embeddings have not yet been
generated. Inspection labels and Markdown markers are only in reports; they are
not added to chunk text.

| Check | Result |
|---|---|
| Deterministic member sample | 3 documents, 739 chunks |
| Full corpus | 60 documents, 169 regions, **11,419 chunks** |
| Unique eligible text covered | **11,424,026 characters — 100%** |
| Repeated characters from overlap | 3,764,857 |
| Content tokens in all chunks | 2,904,243 |
| Maximum content / encoder input tokens | 256 / 258 |
| Adjacent pairs within regions | 11,250, all sharing exactly 64 source tokens |
| Empty / overflow / redundant-tail chunks | 0 / 0 / 0 |
| Observed full-run time | About 82 seconds; includes loading, validation and artifact writing |
| Automated tests | **62 pipeline + 17 cleaning = 79 passed** |

### Compared with the fixed-size baseline

| Measure | Fixed-size | Sliding window |
|---|---:|---:|
| Documents | 60 | 60 |
| Content-token cap | 256 | 256 |
| Configured overlap | 0 | 64 |
| Chunks | 8,627 | 11,419 |
| Content tokens to embed | 2,184,219 | 2,904,243 |
| Unique eligible characters | 11,424,026 | 11,424,026 |

Sliding windows create **2,792 additional chunks** and approximately **33% more
content tokens**. These are future embedding inputs, not already indexed vectors.
Total chunk characters are about 1.33 times unique eligible characters. The token
ratio uses standalone chunk counts; source-stream and standalone tokenization
can differ slightly. Timings are observed runs, not a controlled performance study.

## Verification and review

The independent audit reads saved JSONL and checks exact text, page references,
token limits, union coverage, actual overlap, advancing windows and artifact hashes.
It also compares two complete sliding runs, checks zero-overlap output against all
8,627 saved Phase 3 spans, and checks prepared input hashes against Phase 1.
All checks passed: both runs have identical canonical records, all baseline spans
match, and all 122 prepared-input hashes remain unchanged. The fixed-size CLI was
also rerun on its three-member sample; all 555 canonical records and IDs match
the original Phase 3 sample.

- [Full statistics](../../data/outputs/rag/phase4_sliding_full_20260920/sliding_window/variant_001/chunk_stats.json)
- [Shared text marked for each member](../../data/outputs/rag/phase4_sliding_full_20260920/sliding_window/variant_001/overlap_report.md)
- [Same definition and table under both methods](../../data/outputs/rag/phase4_verification_20260920/boundary_comparison.md)
- [Independent verification](../../data/outputs/rag/phase4_verification_20260920/verification.json)

These are ignored local artifacts. Reproduce the independent check on this set of
saved runs with:

```bash
.venv/bin/python data/outputs/rag/phase4_verification_20260920/check_outputs.py
```

The LCD definition split by the fixed baseline is complete in a sliding window.
This demonstrates boundary behavior, not a measured improvement in retrieval or
answers. Overlap can also repeat irrelevant material and crowd other evidence out
of the generator's context. We will measure both effects in the later stages.

The prepared corpus remains unchanged, and the larger benchmark review remains
due before Phase 11. **Next: Phase 5 — structure-aware chunking**, when requested.
