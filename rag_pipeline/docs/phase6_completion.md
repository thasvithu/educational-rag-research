# Phase 6 — Semantic chunking

Completed on **21 September 2026**. This is the fourth chunking method in the comparison.

## The simple idea

Semantic chunking uses meaning to help choose where a chunk ends.

1. Read the same prepared text used by the other methods.
2. Locate sentences, list items and visible table rows without changing the text.
3. Embed each unit using the pinned **Jina v2 small English** model.
4. Compare each unit with the next using cosine distance. A larger distance means
   the model sees less similarity.
5. Start a new chunk at a sufficiently large change, while enforcing the token cap.
6. Save the chunks, scores and reasons for every boundary decision.

The implementation uses LangChain's `TextSplitter`, `Document` and `Embeddings`
interfaces. The source-offset and diagnostic rules are explicit research code.
Comments explain why those checks are needed.

## Exact rules for this development configuration

| Setting | Value / behavior |
|---|---|
| Final chunk size | At most **256 content tokens**, plus two encoder special tokens. |
| Minimum target | **64 tokens** before a semantic cut; hard-cap cuts and region tails may be shorter. |
| Overlap | None. |
| Comparison | Adjacent individual units, using cosine distance. |
| Threshold | Region's **90th percentile**, calculated with linear interpolation. |
| Threshold ties | A distance must be strictly greater; equal scores stay together unless size forces a cut. |
| Fewer than three units | Pack under the cap without computing a similarity threshold. |
| Oversized unit | Split using exact tokenizer offsets before embedding; record the original unit span. |
| Gap handling | Never cross blank or excluded pages; ordinary page boundaries remain eligible. |
| Inference | Pinned local Jina model, CUDA FP32, batch size 1, content-only mean pooling, L2 normalization. |

The cap takes priority. For example, a 230-token chunk followed by an 80-token
sentence must end even when those passages are related. A 40-token chunk followed
by a 240-token unit also ends early: it is flagged as below the minimum target.
The last short chunk is always retained.

The 90th percentile is the predeclared starting rule, **not an optimal threshold**.
No held-out benchmark questions or answers were used. Choosing final configurations
on reviewed development questions remains work for the evaluation stage.

Content-only pooling excludes CLS/SEP and padding. This is the shared experimental
adaptation agreed in Phase 2, not a claim to reproduce Jina's stock `encode()` pooling.

## Text handling and limitations

Prose uses the existing conservative sentence helper, which protects common
abbreviations, initials and decimal values. Course codes are not separators.
List items and recognizable table rows remain units when they fit; uncertain tables
are not reconstructed. Tokenizer-ignored PDF bullets attach to a neighboring unit.

Every chunk is an exact contiguous source slice. No headings, labels, summaries or
normalized sentences are inserted into its text. Every eligible character remains
covered once. Unit offsets in `boundary_diagnostics.jsonl` are **region-local**;
add `region_char_start` to obtain document-level offsets.

Similarity is a model estimate, not proof of a topic change. An isolated word left
by PDF extraction can score as a large change. Conversely, two different university
rules can have similar wording. The diagnostics preserve these cases for later
failure analysis. Sentence parsing is a documented heuristic, not a language parser.

## Code map

| File | Purpose |
|---|---|
| `chunking/semantic/units.py` | Locate exact scoring units; preserve glyphs and split oversized units. |
| `chunking/semantic/chunker.py` | LangChain splitter, threshold calculation, packing and boundary reasons. |
| `chunking/semantic/cache.py` | LangChain embedding cache with checksums and reproducible keys. |
| `chunking/semantic/reporting.py` | Score examples, boundary counts, cache and GPU measurements. |
| `chunking/semantic/run.py` | Connect the selected model, tokenizer, cache and shared artifact writer. |
| `configs/semantic.json` | Versioned method settings. |
| `tests/test_semantic.py` | Controlled topic changes, caps, offsets, short regions and cache corruption. |

The existing `embedding/model_adapter.py` already supplies the required Jina
capability, so Phase 6 reuses it. No new dependency is needed.

## Commands

From the project root, using a **new run ID** each time:

```bash
.venv/bin/python -m rag_pipeline chunk-semantic --run-id my_semantic_sample --sample
.venv/bin/python -m rag_pipeline chunk-semantic --run-id my_semantic_full
.venv/bin/python -m rag_pipeline chunk-semantic --run-id my_semantic_repeat --cache-only
```

The sample chooses one document per member, deterministically by source name.
The default device is CUDA; use a terminal with access to the NVIDIA GPU. An explicit
`--device cpu` uses a separate configuration/cache. CPU and CUDA results must not
silently share a cache identity. `--cache-only` reads the original configuration's
saved vectors without loading the model; it fails if any required vector is missing.

The command also accepts `--config`, `--method-config` and `--model-config`.
An interrupted run keeps committed cache rows. Start again under a new run ID;
successful earlier vector computations are reused, and chunks are regenerated.
Completed experiment directories are never overwritten.

## Artifacts

```text
data/outputs/rag/<run-id>/
    run_config.json
    corpus_snapshot.json
    environment.json
    stage_manifest.json
    semantic/variant_001/
        chunks.jsonl
        chunk_stats.json
        boundary_embedding_config.json
        boundary_diagnostics.jsonl
        semantic_stats.json
        semantic_examples.json
        semantic_report.md
        inspection_examples.json
        inspection_report.md
        stage_manifest.json
```

**Phase 7 should read `chunks.jsonl`.** The SQLite vectors in
`data/outputs/rag/semantic_cache/` represent scoring units, not these final chunks.
They must not be used as the semantic method's retrieval index.

Each cache key includes exact unit text, method configuration, tokenizer identity,
model and implementation hashes, device, precision and relevant package versions.
Vector checksums, dimensions, finite values and normalization are checked on load.
Each diagnostics unit records its key and checksum. Cache-only repeats preserve
the original scores; bitwise equality across different hardware is not promised.

Timing separates boundary inference, model loading and cache operations from total
chunking time. Cache hits and misses show when a run reused work. Final-chunk
embedding time is zero because that stage belongs to Phase 7.

## Full-corpus results

Completed artifact: `data/outputs/rag/phase6_semantic_full_20260921/semantic/variant_001/`.
The three-document sample is `phase6_semantic_sample_20260921`.

| Measurement | Result |
|---|---:|
| Documents / eligible regions | 60 / 169 |
| Semantic chunks | **11,927** |
| Content-token minimum / maximum | 1 / 256 |
| Mean / median content tokens | 183.13 / 205 |
| Eligible characters covered | **11,424,026 — 100%** |
| Overlap / empty chunks / overflow | 0 / 0 / 0 |
| Scoring units, including short regions | 103,574 |
| Chunks ending at a semantic cut | 5,337 |
| Chunks ending because the next unit would exceed the cap | 6,421 |
| Region-ending chunks | 169 |
| Chunks below the 64-token target | 181 |
| Regions too short for similarity scoring | 12 |
| Fragments produced by pre-splitting oversized units | 736 |

More than half of the chunk endings are size-driven. This is expected under the
shared hard cap and is recorded separately from semantic choices. It must not be
reported as though every boundary represents a detected topic change.

### Size comparison

| Method | Chunks | Mean content tokens | Total content tokens |
|---|---:|---:|---:|
| Fixed-size | 8,627 | 253.18 | 2,184,219 |
| Sliding window | 11,419 | 254.33 | 2,904,243 |
| Structure-aware | 13,377 | 163.28 | 2,184,164 |
| Semantic | 11,927 | 183.13 | 2,184,166 |

These describe size and storage workload, not retrieval or answer quality. Small
token-total differences without overlap come from independent substring tokenization.

### Observed resources

The sample took about **55 seconds**, encoding 6,583 new units. The subsequent full
run took about **642 seconds (10.7 minutes)**, with **81,343 new units** and **22,213
cache hits**. It reused the sample cache, so this is **not a completely cold run**.
Full-run boundary inference accounted for about 376 seconds, model loading for
2.1 seconds, and recorded cache operations for 151 seconds. Remaining time includes
loading, parsing, tokenization, validation and writing. These are development-run
observations on the laptop, not controlled hardware benchmarks.

Torch peak GPU memory was about **294.7 MiB allocated / 304 MiB reserved**. These
figures exclude other applications and driver overhead. The boundary-vector SQLite
cache occupies about **351.3 MiB**. No API calls or paid services were used.

## Verification results

**89 pipeline tests and 17 cleaning tests passed (106 total).** The independent
audit checked all 11,927 chunks and recomputed **103,399 scored boundaries** from
the saved vectors. It verified exact text, page references, token counts, complete
coverage, cache keys/checksums, thresholds, boundary reasons and artifact hashes.
All **122 canonical prepared-file hashes** still match Phase 1.

The full cache-only repeat, `phase6_semantic_repeat_20260921`, took about **113
seconds**, with 103,556 cache hits, zero misses and no model load or inference.
Every chunk record (excluding the execution-specific `run_id`) and the complete
boundary diagnostics were identical. The canonical record fingerprint is:

```text
c00e17fc81051f80cd92b43788e59b3efd153d1c583e547e68b074966ba07d6b
```

Regression samples for fixed-size (555 chunks), sliding-window (739), and
structure-aware (775) exactly match the corresponding earlier full-run records.

## What to review

- [Topic change, related sentences and an extraction-fragment limitation](../../data/outputs/rag/phase6_verification_20260921/review_examples.md)
- [Full chunk statistics](../../data/outputs/rag/phase6_semantic_full_20260921/semantic/variant_001/chunk_stats.json)
- [Boundary decisions, cache counts and GPU measurements](../../data/outputs/rag/phase6_semantic_full_20260921/semantic/variant_001/semantic_stats.json)
- [Independent verification](../../data/outputs/rag/phase6_verification_20260921/verification.json)
- [Earlier-method regression check](../../data/outputs/rag/phase6_verification_20260921/baseline_regression.json)

The teaching examples show a scholarship-to-network-services cut at cosine
distance **0.310351**, above threshold **0.297359**. Related sentences describing
the Network Operating Centre stay together at distance **0.129897**. A separate
example shows an isolated extraction fragment triggering a misleading cut.
These examples explain behavior; they are not a representative quality estimate.

These artifacts are local and ignored by Git. Repeat the independent audit with:

```bash
.venv/bin/python data/outputs/rag/phase6_verification_20260921/check_outputs.py
```

**Next: Phase 7 — shared embeddings for the first four methods**, when requested.
Retrieval quality, answer quality and final threshold selection remain to be
evaluated. The larger reviewed benchmark is still due before Phase 11.
