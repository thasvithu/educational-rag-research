"""Shared LangChain encoder cache for final chunks, separate from boundary vectors."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sqlite3
import time

import numpy as np
from langchain_core.embeddings import Embeddings

from ..common.config import ROOT, within
from ..common.load_corpus import digest_json, file_hash, require
from .model_adapter import JinaEmbeddings, verify_bundle


def embedding_identity(model_config, tokens, device=None, batch_size=None):
    """Pin everything affecting a vector, including our non-stock pooling rule."""
    config = json.loads(Path(model_config).read_text())
    require(config.get("schema_version") == 1, "Unsupported model configuration")
    settings = config["embedding"] | {"device": device or config["embedding"]["device"]}
    if batch_size is not None:
        settings["batch_size"] = batch_size
    pins = verify_bundle(within(ROOT, settings["bundle_dir"]))
    require(settings["model_id"] == "jinaai/jina-embeddings-v2-small-en", "Unsupported model")
    require(settings["model_revision"] == pins["repos"][settings["model_id"]], "Wrong model revision")
    require(settings["implementation_revision"] == pins["repos"]["jinaai/jina-bert-implementation"], "Wrong implementation revision")
    require(settings["device"] in {"cpu", "cuda"}, "Invalid device")
    require(settings["dtype"] == "float32" and settings["pooling"] == "content_mean_l2", "Unsupported precision/pooling")
    require(type(settings["batch_size"]) is int and settings["batch_size"] > 0, "Invalid batch size")
    require(settings["max_input_tokens"] == tokens.max_input_tokens, "Encoder/tokenizer limit mismatch")
    return {"schema_version": 1, "purpose": "independent_chunk_embedding",
            "embedding": {k: v for k, v in settings.items() if k != "bundle_dir"},
            "dimension": 512, "tokenizer": tokens.identity(), "bundle_sha256": pins["sha256"],
            "document_prompt": "", "query_prompt": "", "preprocessing": "exact_text_no_strip_no_truncation",
            "normalization": "l2", "seed": 42, "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8", "batch_policy": "ordered_unique_cache_misses",
            "implementation_sha256": {name: file_hash(ROOT / "rag_pipeline" / name) for name in
                                      ("embedding/model_adapter.py", "embedding/vector_cache.py", "common/tokenization.py")},
            "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "tokenizers")}}


def check_vector(vector, dimension):
    vector = np.asarray(vector, dtype="<f4")
    require(vector.shape == (dimension,) and np.isfinite(vector).all(), "Invalid embedding dimension or non-finite value")
    require(abs(float(np.linalg.norm(vector)) - 1) < 1e-4, "Embedding is not L2 normalized")
    return vector


def model_factory(config, device, batch_size):
    def load():
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import torch
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        model = JinaEmbeddings.from_config(config, device=device)
        require(type(batch_size) is int and batch_size > 0, "Invalid inference batch size")
        model.batch_size = batch_size
        return model
    return load


class ChunkEmbeddings(Embeddings):
    """Batch cache misses through the shared Jina adapter; preserve request order.

    Each committed batch survives interruption. Repeated text can reuse a vector
    across methods because method labels do not affect this independent encoder.
    Every occurrence still gets its own row in the exported embedding matrix.
    """

    def __init__(self, path, identity, tokens, factory, *, cache_only=False):
        self.identity, self.tokens, self.factory = identity, tokens, factory
        self.identity_hash = digest_json(identity)
        self.batch_size = identity["embedding"]["batch_size"]
        self.dimension = identity["dimension"]
        self.model, self.cache_only = None, cache_only
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=30)
        self.connection.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vector BLOB NOT NULL, sha256 TEXT NOT NULL)")
        self.hits = self.misses = self.inferred_input_tokens = 0
        self.inference_seconds = self.load_seconds = self.cache_seconds = 0.0
        self.last_entries = []

    def key(self, text):
        return digest_json({"cache_schema": 1, "embedding_hash": self.identity_hash,
                            "text_sha256": hashlib.sha256(text.encode()).hexdigest()})

    def read(self, key):
        row = self.connection.execute("SELECT vector, sha256 FROM vectors WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        data, checksum = row
        require(hashlib.sha256(data).hexdigest() == checksum, "Chunk-vector cache checksum mismatch")
        return check_vector(np.frombuffer(data, dtype="<f4"), self.dimension).copy(), checksum

    def embed_documents(self, texts):
        require(isinstance(texts, list), "texts must be a list")
        # Check even cached texts, before any inference in this request.
        counts = []
        for text in texts:
            require(isinstance(text, str) and bool(text.strip()), "Cannot embed blank text")
            count = self.tokens.count(text)
            require(self.tokens.special_tokens < count <= self.tokens.max_input_tokens,
                    "Text has no content tokens or exceeds input limit; truncation forbidden")
            counts.append(count)
        started = time.perf_counter()
        keys = [self.key(text) for text in texts]
        saved, missing = {}, {}
        for key, text, count in zip(keys, texts, counts):
            if key in saved or key in missing:
                continue
            row = self.read(key)
            if row is None:
                missing[key] = (text, count)
            else:
                saved[key] = row
        self.cache_seconds += time.perf_counter() - started
        require(not (self.cache_only and missing), "Chunk-vector cache miss in --cache-only mode")
        items = list(missing.items())
        for offset in range(0, len(items), self.batch_size):
            batch = items[offset:offset+self.batch_size]
            if self.model is None:
                started = time.perf_counter()
                self.model = self.factory()
                self.load_seconds += time.perf_counter() - started
            started = time.perf_counter()
            values = self.model.embed_documents([value[0] for _, value in batch])
            self.inference_seconds += time.perf_counter() - started
            require(len(values) == len(batch), "Wrong number of returned embedding rows")
            encoded = []
            for (key, _), value in zip(batch, values):
                data = check_vector(value, self.dimension).tobytes()
                encoded.append((key, data, hashlib.sha256(data).hexdigest()))
            started = time.perf_counter()
            with self.connection:
                self.connection.executemany("INSERT OR IGNORE INTO vectors VALUES (?, ?, ?)", encoded)
            for key, _, _ in encoded:
                saved[key] = self.read(key)  # Use the committed value if another writer won this key.
            self.cache_seconds += time.perf_counter() - started
            self.misses += len(batch)
            self.inferred_input_tokens += sum(value[1] for _, value in batch)
        self.hits += len(texts) - len(missing)
        self.last_entries = [{"cache_key": key, "vector_sha256": saved[key][1]} for key in keys]
        return [saved[key][0].tolist() for key in keys]

    def embed_query(self, text):
        # The selected model has no query prefix; pooling is identical.
        return self.embed_documents([text])[0]

    def counters(self):
        return {"cache_hits": self.hits, "cache_misses": self.misses,
                "inferred_input_tokens": self.inferred_input_tokens, "inference_seconds": self.inference_seconds,
                "model_load_seconds": self.load_seconds, "cache_seconds": self.cache_seconds}

    def memory(self):
        if self.model is None or self.identity["embedding"]["device"] != "cuda":
            return {"gpu_inference_used": False}
        import torch
        return {"gpu_inference_used": True, "gpu_name": torch.cuda.get_device_name(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "note": "Cumulative process peaks; excludes other applications and driver overhead."}

    def close(self):
        self.connection.close()
