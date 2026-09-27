# Phase 8 — Late chunking

Implemented on **21 September 2026**. The full run and its cached repeat cover
all 60 documents. Phase 9 indexing and retrieval are the next stage.

## Recorded results

| Check or measurement | Result |
|---|---|
| Exact matched baseline chunks | **8,627**, each with a 512-dimensional normalized FP32 vector. |
| Documents / eligible regions | **60 / 169**. |
| Encoder windows | **1,804**; maximum **1,794** total tokens; no budget fallback required. |
| Planned total window input | **3,022,356 tokens**, about 1.373× independent fixed-chunk input. |
| Independent/contextual token-count differences | **210 chunks**, recorded without changing source spans. |
| Whole-window early/late parity | Maximum element difference **1.49e-8**, below the **1e-6** tolerance. |
| Same-chunk window-placement comparisons | **3,270**; mean cosine **0.97538**, minimum **0.88825**. |
| Early/late vector comparisons | Mean cosine **0.95412**, minimum **0.73017**; not an accuracy score. |
| GPU peak allocated / reserved | Approximately **597 / 848 MiB**. |
| Full process peak host memory | Approximately **2.07 GiB**. |
| Final vector matrix | Approximately **16.85 MiB**. |
| Full run elapsed / actual inference | Approximately **149.5 / 113.6 seconds**; includes **117 cached windows**, so partly warm. |
| Cached repeat | **1,804 hits**, zero inference; approximately **29.2 seconds** including validation. |
| Tests | **105 pipeline + 17 cleaning = 122 passed**. |

Completed artifacts:

- [Full-run inspection example](../../data/outputs/rag/phase8_late_full_20260921/late/variant_001/inspection_report.md).
- [Full-run statistics](../../data/outputs/rag/phase8_late_full_20260921/late/variant_001/chunk_stats.json).
- [Independent audit and repeat comparison](../../data/outputs/rag/phase8_late_full_20260921_audit.json).
- Cached repeat: `data/outputs/rag/phase8_late_repeat_20260921/`.

The audit checks original source slices/pages, unique ownership, window limits,
token membership, cache context, matrix rows/checksums, and seam calculations.
It also checks all **122 prepared files** against Phase 1 and the **six paired
early artifacts** against preflight hashes. Matrices, row maps, window records
and comparison records match the cached repeat byte for byte; chunk records
match except for their execution `run_id`.

## What happens in simple English

The fixed-size baseline gave us 8,627 chunks. In Phase 7, the model read each
chunk separately before creating its vector. In Phase 8, it reads a longer
passage first. We then average only the token vectors belonging to each chunk.
This gives a chunk access to surrounding context while preserving its exact
text, source pages and boundaries.

For example, the inspection report shows a chunk beginning **“it self-sufficient
with respect to operational expenses…”**. The preceding chunk explains that
UCSC earns money through consultancy and research. Late encoding can see that
earlier part of the sentence. Whether this helps retrieve the right
answer is a question for the later evaluation; a changed vector alone does not
prove improvement.

## Shared model and controlled comparison

- Same pinned `jinaai/jina-embeddings-v2-small-en` model, tokenizer, CUDA FP32,
  512 dimensions, no prefixes, content-only mean pooling and L2 normalization.
- Same exact fixed-size spans and prepared text; new `late` chunk IDs link back
  to the original `early_chunk_id` and Phase 7 matrix row.
- Special tokens and padding never enter the pooling mean. Neighboring tokens
  contribute through the model's attention, but never through the chunk's mean.
- Phase 8 uses batch **one** for longer inputs; Phase 7 used batch **eight**.
  Phase 7's comparison found differences below 1.8e-7 between those batch sizes.
  Both settings are recorded rather than claimed to be numerically identical.
- Preflight rejects mismatched model, tokenizer, pooling, precision, device,
  code, corpus or damaged baseline artifacts. The selected baseline currently
  uses the 256-content-token configuration.

This is a windowed late-chunking experiment. Large handbooks are not encoded
as one whole document. Content-only pooling remains the shared experimental
adaptation documented in Phase 2, rather than the stock Jina pooling recipe.

## Window policy

Usually, **five consecutive chunks own a window**. One additional chunk on each
side supplies context when available. Those neighbors may also appear in a
different window, but each indexed chunk has exactly one owner.

```text
Window A: [owned 1 2 3 4 5] [neighbor 6]
Window B: [neighbor 5] [owned 6 7 8 9 10] [neighbor 11]
```

Windows stop at document boundaries, excluded pages and blank-page gaps.
Ordinary nonblank page breaks can remain inside a window. Every window is
retokenized and checked against the 2,048-total-token limit. If necessary, the
planner reduces the number of owned chunks, then removes outer neighbors;
it never cuts a target chunk. Every adjustment is recorded.

Window-local token offsets determine pooling membership. Tokens that cross a
target boundary cause an explicit failure. Repeated offsets for Unicode
subtokens stay together. The runtime encoder's token positions must match
the tokenizer-only plan before any vectors are cached.

Independent and contextual token counts can differ at word-piece boundaries.
`token_count` continues to describe the isolated chunk; `pooled_content_tokens`
records how many tokens actually entered the contextual mean. Source text and
character spans remain identical.

## LangChain and code layout

The existing `JinaEmbeddings` class implements LangChain's `Embeddings` interface.
Late chunking calls its `encode_tokens()` helper, then applies exact-span pooling.
A normal `embed_documents()` call would encode and pool each chunk independently,
so the small pooling helper is necessary for this experiment.

| File | Responsibility |
|---|---|
| `chunking/late/chunker.py` | Window ownership, token alignment and contextual pooling. |
| `chunking/late/cache.py` | Atomic context-aware cache and whole-window parity probe. |
| `chunking/late/run.py` | Baseline validation, GPU execution, vectors, mappings and reports. |
| `chunking/late/audit.py` | Independent re-reading of sources, offsets, cache and saved vectors. |
| `configs/late.json` | Baseline location, five owned chunks, one neighbor per side and input cap. |
| `tests/test_late.py` | Controlled ownership, gap, Unicode, pooling and cache-recovery checks. |

## Commands

Run GPU commands in a terminal with access to the NVIDIA GPU. No new package,
model download or paid API is needed. Choose a new run ID for every execution.

```bash
.venv/bin/python -m rag_pipeline chunk-late --run-id my_late_sample --sample
.venv/bin/python -m rag_pipeline chunk-late --run-id my_late_full
.venv/bin/python -m rag_pipeline chunk-late --run-id my_late_repeat --cache-only
.venv/bin/python -m rag_pipeline.chunking.late.audit --run-id my_late_full --repeat-id my_late_repeat
```

`--sample` selects the first source by name from each member. Baseline integrity
is still checked across the full corpus. Copy `configs/late.json` and use
`--method-config` if the paired early run has a different directory name.

After interruption, start with a **new run ID** and the same settings. Each
successful window was committed atomically and is reused; only missing windows
need inference. The earlier incomplete run stays visible. Phase 8 deliberately
uses this recovery workflow rather than Phase 7's same-run `--resume` command.

The separate SQLite cache lives under `data/outputs/rag/late_window_cache/`.
Its keys include model/implementation identity, complete window text and pooling
spans. A vector is never reused merely because the target chunk text matches.
Changed context creates a different key. Corrupt or missing cache entries fail
explicitly in `--cache-only` mode. Cached repeats preserve exact saved vectors;
fresh inference on different hardware is not promised to be bitwise identical.

## Output files

Each run writes `data/outputs/rag/<run-id>/late/variant_001/`:

| Artifact | Contents |
|---|---|
| `chunks.jsonl` | Exact original spans, new late IDs, baseline links and parent-window metadata. |
| `embeddings.npy` | One 512-dimensional vector per owned chunk. |
| `embedding_rows.jsonl` | Ordered vector-to-chunk/source mapping and checksums. |
| `encoder_windows.jsonl` | Context spans, ownership, token membership and context-cache keys. |
| `early_late_comparison.jsonl` | Same-span vector similarities and available left/right context. |
| `window_seams.jsonl` | Similarity of the same chunk under two different overlapping windows. |
| `chunk_stats.json`, `embedding_stats.json` | Counts, timing, memory, parity and validation results. |
| `inspection_report.md` | A source example with earlier context and its exact indexed chunk. |

Root configuration, corpus and environment snapshots identify the experiment.
Manifests hash the artifacts. Audit reports are written beside the completed
run so that its original manifest remains valid. Intermediate token vectors
are discarded; pooled window vectors are kept in the cache for verification.

## How to interpret the diagnostics

Cosine similarities compare vector representations. They are **not accuracy
percentages**. `window_seams.jsonl` exposes sensitivity to context-window
placement; it does not remove that limitation. Actual retrieval and answer
quality will be measured using the shared pipeline and reviewed benchmark.

Inference timing excludes model loading and the separately timed extra forward
pass for whole-window parity. Total elapsed time includes preparation and output
validation. Cache hits and misses are reported separately; a partly cached run
must not be presented as a cold runtime comparison. GPU peaks include diagnostic
inference and exclude other applications and driver memory.
