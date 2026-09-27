# Phase 7 — Shared chunk embeddings

Completed on **21 September 2026** for the four independently encoded chunking methods.

## What this stage does

Each finished chunk becomes a list of **512 numbers**, called an embedding vector.
Later, retrieval will compare a question's vector with these chunk vectors.
This stage creates the vectors and their source links; indexing and retrieval
remain later phases.

All four methods use the same local Jina model through LangChain's `Embeddings`
interface. The model sees the exact chunk text. Source names, page numbers and
inferred headings stay in metadata and are not added to the text.

| Shared choice | Setting |
|---|---|
| Model | Pinned `jinaai/jina-embeddings-v2-small-en` from Phase 2. |
| Precision / device | Float32 on the local NVIDIA GPU. |
| Dimensions | 512 per chunk. |
| Pooling | Mean of content-token representations, excluding padding and CLS/SEP. |
| Normalization | L2; each vector has length approximately one. |
| Document / query prefixes | Both empty. |
| Input text | Exact source chunk, with original whitespace. |
| Length handling | Retokenize and validate; never silently truncate. |
| Randomness | Seed 42 and deterministic Torch algorithms. |

Content-only pooling is the shared experimental adaptation agreed in Phase 2;
it is not the stock Jina `encode()` pooling. Query encoding uses the same model,
tokenization and pooling through `JinaEmbeddings.embed_query()`.

The Phase 7 batch size is **8**, configured in `configs/embedding.json`. This overrides
the adapter's default for this stage and is included in the vector identity.
The initial batch-one probe is retained separately from the batch-eight probe.
The largest element difference across all 2,845 batch-one/batch-eight sample vectors was below **1.8e-7**; peak reserved GPU memory in the batch-eight probe was about **412 MiB**. Model weights, precision, pooling and input text remain the same across methods.

## One row, one chunk

Each method has a separate `embeddings.npy` matrix. Row 0 corresponds to row 0
of `embedding_rows.jsonl`, which identifies the original `chunk_id`. The record
also includes document, source, member, pages, offsets, token count, source-text
hash, embedding identity and vector checksum.

Identical text may reuse the same vector, but each chunk occurrence still receives
its own matrix row and source mapping. This matters when the same sentence occurs
on different pages or in several handbook versions.

Each method's directory also contains a copy of its original chunk records. The
`run_id` inside those records still identifies the **chunking** run; the embedding
stage manifest identifies the later **embedding** run. The source chunk files and
earlier completed manifests are preserved.

## Cache and interrupted runs

Final chunk vectors use their own SQLite cache under
`data/outputs/rag/chunk_embedding_cache/`. The Phase 6 sentence/unit cache is separate.

A cache key depends on exact text and the shared encoder identity: model and code
hashes, tokenizer, pooling, normalization, prompts, device, precision, batch policy,
seed and relevant package versions. A method name is not part of this key because
the independent encoder treats identical text identically. Changing an encoder
setting creates a different cache identity.

Every successful inference batch is committed. If a run is interrupted:

1. Start the same command with `--resume` and the same run ID and settings.
2. Completed method artifacts are verified and retained.
3. An incomplete method rebuilds its files in the original row order, reusing
   committed vectors and computing only missing ones.
4. Resume is rejected if source files, configuration or relevant code changed.

A process lock prevents two writers from using the same run. Completed runs are
immutable. To repeat a completed run, use a new ID; add `--cache-only` to require
existing vectors and avoid loading the model. A missing or corrupt cache entry
fails explicitly. Caches preserve exact recorded vectors; bitwise equality across
fresh inference batches or different hardware is not promised.

## Commands

From the project root, use the existing environment. Default input paths point to
the completed Phase 3–6 full runs; edit a copy of `configs/embedding.json` and pass
`--input-config` if teammates have different run directory names.

```bash
.venv/bin/python -m rag_pipeline embed-chunks --run-id my_embedding_sample --sample
.venv/bin/python -m rag_pipeline embed-chunks --run-id my_embeddings
.venv/bin/python -m rag_pipeline embed-chunks --run-id my_embeddings --resume
.venv/bin/python -m rag_pipeline embed-chunks --run-id my_embedding_repeat --cache-only
```

Use the resume command only after interruption/failure. Preserve `--sample` when
resuming a sample. `--device cpu` is an explicit separate vector configuration;
GPU inference requires a GPU-enabled terminal. The runner currently targets the
project's Linux environment, including its process lock and memory measurements.
No new dependency or API service is required.

## Code and files

| File | Responsibility |
|---|---|
| `embedding/model_adapter.py` | Existing pinned Jina implementation behind LangChain. |
| `embedding/vector_cache.py` | Shared batched encoder, cache keys, vector checks and measurements. |
| `embedding/inputs.py` | Validate finished chunks against manifests and the prepared corpus. |
| `embedding/encode_chunks.py` | Ordered vector export, row maps, resume, locks and manifests. |
| `configs/embedding.json` | Four input artifact paths and common inference batch size. |
| `tests/test_embeddings.py` | Controlled batching, corruption, overflow, mapping and recovery checks. |

```text
data/outputs/rag/<run-id>/
    run_config.json
    embedding_config.json
    corpus_snapshot.json
    environment.json
    attempts/environment_001.json
    embedding_summary.json
    cache_stats.json
    stage_manifest.json
    fixed_size/variant_001/
        chunks.jsonl
        embeddings.npy
        embedding_rows.jsonl
        embedding_stats.json
        stage_manifest.json
    sliding_window/variant_001/...
    structure_aware/variant_001/...
    semantic/variant_001/...
```

Only a method with a **complete** stage manifest is ready for downstream use.
Vector files use float32 NumPy arrays and require no pickle loading. Each source
chunk is checked against the current corpus, including exact offsets, source pages,
token limits and its deterministic ID, before inference begins.

## Measurements and interpretation

Statistics separate model loading, actual inference and cache operations. Indexed
token counts include every chunk occurrence; inferred token counts cover only
cache misses. File sizes and cumulative GPU/process-memory peaks are recorded.
The run manifest preserves attempt history and preflight-validation time.

Later methods may benefit from vectors cached by earlier methods. Therefore their
observed timings are **not a controlled cold-speed ranking**. Cache-only timing
measures reuse/export, not embedding inference. On a resumed job, method statistics
describe the successful attempt; inspect failed attempts separately for their
recorded work and elapsed timestamps.

## Verification and handover

The full run is saved under:

```text
data/outputs/rag/phase7_embeddings_full_20260921/
```

All four methods contain chunks from all 60 documents and share encoder identity
`d32e9c49b9612c8029b9cf12baf2954b2f969ed465a96511a71b844529cd3e83`.

| Method | Matrix shape | Matrix MiB | Observed stage seconds | Cache hits / new vectors |
|---|---:|---:|---:|---:|
| Fixed-size | 8,627 × 512 | 16.85 | 46.56 | 568 / 8,059 |
| Sliding window | 11,419 × 512 | 22.30 | 53.84 | 1,934 / 9,485 |
| Structure-aware | 13,377 × 512 | 26.13 | 56.65 | 1,570 / 11,807 |
| Semantic | 11,927 × 512 | 23.30 | 48.01 | 2,685 / 9,242 |
| **Total** | **45,350 × 512** | **88.57** | **205.06** | **6,757 / 38,593** |

The stage times include vector export and validation. Actual new-vector inference
totaled about **165 seconds**. Preflight validation took another **44 seconds**;
the run took about **4.2 minutes including preflight**. The sample cache was already
available, so this is not a fully cold measurement.

Torch peak GPU memory was about **361.8 MiB allocated / 416 MiB reserved**; peak
process RAM was about **1.97 GiB**. GPU measurements exclude other applications and
driver overhead. The final-chunk cache contains **41,284 unique text vectors** and
occupies about **164.9 MiB**. The matrices still have 45,350 rows because repeated
text occurrences keep separate source identities. Row maps and copied chunk text
are additional storage, recorded separately in each method's statistics.

No API calls or paid services were used.

### Verified checks

**99 pipeline tests and 17 cleaning tests passed — 116 total.** The new tests cover
batch order, duplicate text, oversized inputs, wrong vector dimensions, NaN/zero
vectors, corrupted caches, changed input text, interrupted jobs, concurrent writers
and preservation of completed method artifacts during resume.

The independent full-corpus audit checked every vector's bytes, normalization,
cache key/checksum, row identity, text hash, real source span/pages and actual token
count. All four methods share the recorded encoder identity. All **122 canonical
prepared files** and the input chunk artifacts are unchanged.

The completed repeat is `phase7_embeddings_repeat_20260921`. It used **45,350 cache
hits, zero misses, zero inference and no model load**. Its matrices, row maps and
copied chunks are byte-identical to the GPU run. Export/validation took about
**23.2 seconds**, with another **43.6 seconds** of preflight checks.

### Review and reproduce

- [Readable vector/source examples and method comparison](../../data/outputs/rag/phase7_verification_20260921/phase7_embeddings_full_20260921_review.md)
- [Independent full-corpus audit](../../data/outputs/rag/phase7_verification_20260921/phase7_embeddings_full_20260921_audit.json)
- [Detailed time, token, memory and storage measurements](../../data/outputs/rag/phase7_embeddings_full_20260921/embedding_summary.json)
- [Batch-one versus batch-eight numerical comparison](../../data/outputs/rag/phase7_verification_20260921/batch_comparison.json)

The original batch-one sample, `phase7_embeddings_sample_20260921`, and the
selected batch-eight sample, `phase7_embeddings_batch8_sample_20260921`, are
retained as feasibility records. Use the **full** run for downstream work.

Repeat the saved-output audit with:

```bash
.venv/bin/python data/outputs/rag/phase7_verification_20260921/check_outputs.py --compare phase7_embeddings_repeat_20260921
```

These outputs are local and ignored by Git. The checks establish vector and source
integrity; retrieval and answer quality remain to be measured on the benchmark.

**Next: Phase 8 — late chunking**, when requested. It will contextualize longer
source windows before pooling the fixed baseline's matched chunk spans.
