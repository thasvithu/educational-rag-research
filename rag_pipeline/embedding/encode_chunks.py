"""Resumable shared embedding stage for four independently chunked corpora."""

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import resource

import numpy as np

from ..common.config import ROOT, load_config, within, validate_run_id
from ..common.load_corpus import digest_json, file_hash, load_corpus, require
from ..common.run_manifest import create_run, environment, finish_run, now, write_json
from ..common.tokenization import load_tokenizer
from .inputs import METHODS, load_inputs, verify_manifest
from .vector_cache import ChunkEmbeddings, embedding_identity, model_factory


@contextmanager
def exclusive_run(run):
    """Release the OS lock even when an embedding job fails or is interrupted."""
    with (run / ".embedding.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another embedding process is using this run") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def atomic_jsonl(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_outputs(variant, rows, dimension, embedding_hash):
    """Read saved files back, including the vector-row-to-source association."""
    vectors = np.load(variant / "embeddings.npy", mmap_mode="r", allow_pickle=False)
    mapping = [json.loads(line) for line in (variant / "embedding_rows.jsonl").read_text().splitlines()]
    require(vectors.dtype == np.dtype("float32") and vectors.shape == (len(rows), dimension), "Saved vector shape/type mismatch")
    require(len(mapping) == len(rows), "Saved mapping row count differs")
    require(np.isfinite(vectors).all() and np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4, rtol=0), "Invalid saved vectors")
    require([json.loads(line) for line in (variant / "chunks.jsonl").read_text().splitlines()] == rows, "Saved chunk copy differs")
    import hashlib
    for index, (record, chunk) in enumerate(zip(mapping, rows)):
        require(record["row_index"] == index and record["chunk_id"] == chunk["chunk_id"], "Vector row order/identity mismatch")
        require(record["embedding_hash"] == embedding_hash, "Mixed vector spaces")
        require(record["text_sha256"] == hashlib.sha256(chunk["text"].encode()).hexdigest(), "Embedding text hash differs")
        require(record["vector_sha256"] == hashlib.sha256(vectors[index].astype("<f4").tobytes()).hexdigest(), "Saved vector checksum differs")
        for key in ("document_id", "source", "member", "char_start", "char_end", "page_numbers", "token_count", "corpus_hash", "config_hash", "method"):
            require(record[key] == chunk[key], f"Vector source mapping differs: {key}")
    return {"rows": len(rows), "dimension": dimension, "finite": True, "normalized": True,
            "ordered_chunk_mapping": True, "vector_checksums": True}


def encode_variant(variant, rows, encoder):
    """Rebuild ordered files from committed cache rows on resume; never append blindly."""
    import hashlib
    before, started = encoder.counters(), time.perf_counter()
    vectors = np.empty((len(rows), encoder.dimension), dtype="<f4")
    mapping = []
    # A small request bounds memory. The adapter batches cache misses internally;
    # the selected inference batch size is recorded in the embedding identity.
    for start in range(0, len(rows), 64):
        batch = rows[start:start+64]
        values = encoder.embed_documents([row["text"] for row in batch])
        vectors[start:start+len(batch)] = values
        for index, chunk, entry in zip(range(start, start+len(batch)), batch, encoder.last_entries):
            mapping.append({"schema_version": 1, "row_index": index, "chunk_id": chunk["chunk_id"],
                            "embedding_hash": encoder.identity_hash, "text_sha256": hashlib.sha256(chunk["text"].encode()).hexdigest(),
                            **{k: chunk[k] for k in ("document_id", "source", "member", "char_start", "char_end", "page_numbers", "token_count", "corpus_hash", "config_hash", "method")}, **entry})
        if start % 1024 == 0:
            print(f"{rows[0]['method']}: {start+len(batch):,}/{len(rows):,} vectors", flush=True)
    temporary = variant / "embeddings.npy.tmp"
    try:
        with temporary.open("wb") as stream:
            np.save(stream, vectors, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(variant / "embeddings.npy")
    finally:
        temporary.unlink(missing_ok=True)
    atomic_jsonl(variant / "embedding_rows.jsonl", mapping)
    atomic_jsonl(variant / "chunks.jsonl", rows)
    validation = validate_outputs(variant, rows, encoder.dimension, encoder.identity_hash)
    delta = {k: v-before[k] for k, v in encoder.counters().items()}
    stats = {"method": rows[0]["method"], "chunks": len(rows), "documents": len({r['document_id'] for r in rows}),
             "dimension": encoder.dimension, "dtype": "float32", "embedding_hash": encoder.identity_hash,
             "indexed_content_tokens": sum(r["method_metadata"]["content_token_count"] for r in rows),
             "indexed_input_tokens": sum(r["token_count"] for r in rows), "elapsed_seconds": time.perf_counter()-started,
             **delta, "gpu_memory": encoder.memory(), "validation": validation,
             "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
             "storage_bytes": {name: (variant / name).stat().st_size for name in ("embeddings.npy", "embedding_rows.jsonl", "chunks.jsonl")},
             "timing_note": "Observed development timing; cache hits are not inference. Methods may reuse identical text from earlier methods/runs."}
    write_json(variant / "embedding_stats.json", stats)
    return stats


def run_embeddings(run_id, *, config_path=ROOT / "rag_pipeline/config.json",
                   input_config=ROOT / "rag_pipeline/configs/embedding.json",
                   model_config=ROOT / "rag_pipeline/configs/models.json",
                   sample=False, device=None, cache_only=False, resume=False):
    preflight_started = time.perf_counter()
    config, tokens = load_config(Path(config_path)), load_tokenizer(Path(model_config))
    batch_size = json.loads(Path(input_config).read_text())["batch_size"]
    identity = embedding_identity(model_config, tokens, device, batch_size)
    corpus = load_corpus(ROOT / config.corpus_dir, config)
    rows_by_method, inputs = load_inputs(input_config, corpus, tokens, sample=sample)
    preflight_seconds = time.perf_counter() - preflight_started
    effective = {"schema_version": 1, "stage": "phase7_embeddings", "corpus_hash": corpus.corpus_hash,
                 "embedding_identity": identity, "sample": sample, "input_sha256": inputs,
                 "writer_sha256": {name: file_hash(Path(__file__).parent / name) for name in ("encode_chunks.py", "inputs.py")}}
    effective["config_hash"] = digest_json(effective)
    validate_run_id(run_id)
    run = within(ROOT, f"{config.output_root}/{run_id}")
    # Both new and resumed runs must remain in ignored storage.
    for path in (run, ROOT / "data/outputs/rag/chunk_embedding_cache"):
        relative = path.relative_to(ROOT).as_posix()
        require(subprocess.run(["git", "check-ignore", "--no-index", "-q", relative + "/probe"], cwd=ROOT).returncode == 0, "Embedding output must be ignored")
        require(not subprocess.check_output(["git", "ls-files", "--", relative], cwd=ROOT, text=True).strip(), "Embedding output must not contain tracked files")
    if not resume:
        run, _ = create_run(ROOT, config, run_id)
    require(run.is_dir(), "Resume run does not exist")
    with exclusive_run(run):
        manifest = json.loads((run / "stage_manifest.json").read_text())
        if resume:
            require(manifest["status"] in {"running", "failed"}, "Completed runs are immutable; use a new run ID")
            require(json.loads((run / "run_config.json").read_text()) == effective, "Resume configuration, code or source changed")
        else:
            write_json(run / "run_config.json", effective)
            write_json(run / "corpus_snapshot.json", corpus.snapshot)
        attempts = manifest.setdefault("attempts", [])
        attempt = {"number": len(attempts)+1, "started_at_utc": now(), "resume": resume, "cache_only": cache_only,
                   "preflight_seconds": preflight_seconds}
        attempts.append(attempt)
        manifest.update(stage="phase7_embeddings", status="running", config_hash=effective["config_hash"], corpus_hash=corpus.corpus_hash)
        write_json(run / "stage_manifest.json", manifest)
        attempt_dir = run / "attempts"
        attempt_dir.mkdir(exist_ok=True)
        snapshot = environment(ROOT)
        write_json(attempt_dir / f"environment_{len(attempts):03}.json", snapshot)
        if not resume:
            write_json(run / "environment.json", snapshot)
        cache = ROOT / "data/outputs/rag/chunk_embedding_cache" / (digest_json(identity) + ".sqlite3")
        write_json(run / "embedding_config.json", {"identity": identity, "embedding_hash": digest_json(identity), "cache_path": cache.relative_to(ROOT).as_posix()})
        encoder = ChunkEmbeddings(cache, identity, tokens, model_factory(model_config, identity["embedding"]["device"], identity["embedding"]["batch_size"]), cache_only=cache_only)
        active_variant, stage = None, None
        try:
            summary = []
            for method in METHODS:
                stage = None
                active_variant = run / method / "variant_001"
                active_variant.mkdir(parents=True, exist_ok=True)
                stage_path = active_variant / "stage_manifest.json"
                if stage_path.exists():
                    previous = json.loads(stage_path.read_text())
                    if previous["status"] == "complete":
                        verify_manifest(active_variant)
                        validate_outputs(active_variant, rows_by_method[method], encoder.dimension, encoder.identity_hash)
                        summary.append(json.loads((active_variant / "embedding_stats.json").read_text()))
                        active_variant = None
                        continue
                stage = {"schema_version": 1, "stage": "embedding", "method": method, "run_id": run_id,
                         "status": "running", "started_at_utc": now(), "embedding_hash": encoder.identity_hash,
                         "config_hash": effective["config_hash"], "corpus_hash": corpus.corpus_hash}
                write_json(stage_path, stage)
                summary.append(encode_variant(active_variant, rows_by_method[method], encoder))
                finish_run(active_variant, stage)
                active_variant = None
            for path, expected in inputs.items():
                require(file_hash(ROOT / path) == expected, f"Chunk input changed during embedding: {path}")
            for path, expected in corpus.snapshot["files"].items():
                require(file_hash(ROOT / config.corpus_dir / path) == expected, f"Prepared input changed: {path}")
            write_json(run / "embedding_summary.json", summary)
            write_json(run / "cache_stats.json", {"path": cache.relative_to(ROOT).as_posix(),
                       "bytes": cache.stat().st_size,
                       "entries": encoder.connection.execute("SELECT count(*) FROM vectors").fetchone()[0],
                       "note": "Shared final-chunk cache; can contain vectors from other runs with the same embedding identity."})
            attempt.update(status="complete", finished_at_utc=now(), counters=encoder.counters(), gpu_memory=encoder.memory())
            finish_run(run, manifest)
            print(json.dumps({"output": str(run), "methods": {s['method']: s['chunks'] for s in summary},
                              "dimension": encoder.dimension, "status": "complete"}, indent=2))
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            attempt.update(status="failed", finished_at_utc=now(), error=error, counters=encoder.counters())
            if active_variant is not None and stage is not None:
                finish_run(active_variant, stage, error=error)
            finish_run(run, manifest, error=error)
            raise
        finally:
            encoder.close()
    return run
