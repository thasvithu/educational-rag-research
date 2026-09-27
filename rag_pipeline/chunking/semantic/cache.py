"""Persistent boundary vectors, keyed by exact text and complete experiment identity."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sqlite3
import time

import numpy as np
from langchain_core.embeddings import Embeddings

from ...common.config import ROOT, within
from ...common.load_corpus import digest_json, file_hash, require
from ...embedding.model_adapter import JinaEmbeddings, verify_bundle


def boundary_identity(model_config, device=None):
    settings = json.loads(Path(model_config).read_text())["embedding"]
    effective = settings | {"device": device or settings["device"]}
    require(effective["device"] in {"cpu", "cuda"}, "Device must be cpu or cuda")
    pins = verify_bundle(within(ROOT, settings["bundle_dir"]))
    # The adapter itself validates the selected revisions, precision and pooling
    # when loaded. Validate these even on an entirely cached run.
    require(settings["model_id"] == "jinaai/jina-embeddings-v2-small-en", "Unsupported model")
    require(settings["model_revision"] == pins["repos"][settings["model_id"]], "Wrong model revision")
    require(settings["implementation_revision"] == pins["repos"]["jinaai/jina-bert-implementation"], "Wrong implementation revision")
    require(settings["dtype"] == "float32" and settings["pooling"] == "content_mean_l2", "Unsupported pooling/precision")
    require(type(settings["batch_size"]) is int and settings["batch_size"] == 1,
            "Boundary inference currently uses the tested batch size of one")
    require(type(settings["max_input_tokens"]) is int and 3 <= settings["max_input_tokens"] <= 2048, "Invalid input limit")
    files = ["embedding/model_adapter.py", "chunking/semantic/units.py", "chunking/semantic/chunker.py",
             "chunking/semantic/cache.py", "chunking/structure_aware/units.py", "common/token_windows.py"]
    return {"embedding": {k: v for k, v in effective.items() if k != "bundle_dir"},
            "bundle_sha256": pins["sha256"], "dimension": 512,
            "implementation_sha256": {name: file_hash(ROOT / "rag_pipeline" / name) for name in files},
            "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "tokenizers")},
            "deterministic_algorithms": True, "seed": 42, "cublas_workspace_config": ":4096:8"}


class CachedBoundaryEmbeddings(Embeddings):
    """LangChain Embeddings with atomic SQLite rows; corrupt cache entries fail loudly.

    The model is loaded only after a miss. Cached vectors are boundary-unit vectors,
    not the final retrieval embeddings that Phase 7 will create.
    """

    def __init__(self, path, identity, config_hash, factory):
        self.identity, self.config_hash, self.factory = identity, config_hash, factory
        self.model = None
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vector BLOB NOT NULL, sha256 TEXT NOT NULL)")
        self.hits = self.misses = 0
        self.encode_seconds = self.model_load_seconds = self.cache_seconds = 0.0
        self.last_entries = []

    def key(self, text):
        return digest_json({"cache_version": 1, "identity": self.identity, "config_hash": self.config_hash,
                            "text_sha256": hashlib.sha256(text.encode()).hexdigest()})

    def checked(self, data, expected):
        require(hashlib.sha256(data).hexdigest() == expected, "Boundary cache checksum mismatch")
        vector = np.frombuffer(data, dtype="<f4").copy()
        require(vector.shape == (self.identity["dimension"],) and np.isfinite(vector).all(), "Invalid cached vector")
        require(abs(float(np.linalg.norm(vector)) - 1) < 1e-4, "Boundary vector is not normalized")
        return vector

    def embed_documents(self, texts):
        vectors, entries = [], []
        for text in texts:
            started = time.perf_counter()
            key = self.key(text)
            row = self.connection.execute("SELECT vector, sha256 FROM vectors WHERE key = ?", (key,)).fetchone()
            self.cache_seconds += time.perf_counter() - started
            if row is None:
                if self.model is None:
                    started = time.perf_counter()
                    self.model = self.factory()
                    self.model_load_seconds += time.perf_counter() - started
                started = time.perf_counter()
                values = self.model.embed_documents([text])
                self.encode_seconds += time.perf_counter() - started
                require(len(values) == 1, "Boundary encoder returned wrong row count")
                data = np.asarray(values[0], dtype="<f4").tobytes()
                checksum = hashlib.sha256(data).hexdigest()
                vector = self.checked(data, checksum)
                started = time.perf_counter()
                # Commit each row: an interruption loses no previously encoded unit.
                with self.connection:
                    self.connection.execute("INSERT OR IGNORE INTO vectors VALUES (?, ?, ?)", (key, data, checksum))
                # If another process won this key, use its saved vector so the
                # checksum always identifies exactly what later reruns will read.
                saved = self.connection.execute("SELECT vector, sha256 FROM vectors WHERE key = ?", (key,)).fetchone()
                vector, checksum = self.checked(*saved), saved[1]
                self.cache_seconds += time.perf_counter() - started
                self.misses += 1
                if self.misses % 500 == 0:
                    print(f"Boundary embeddings: {self.misses:,} new units, {self.hits:,} cache hits", flush=True)
            else:
                vector = self.checked(*row)
                checksum = row[1]
                self.hits += 1
            vectors.append(vector.tolist())
            entries.append({"cache_key": key, "vector_sha256": checksum})
        self.last_entries = entries
        return vectors

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    def statistics(self):
        result = {"cache_hits": self.hits, "cache_misses": self.misses,
                  "model_load_seconds": self.model_load_seconds, "boundary_embedding_seconds": self.encode_seconds,
                  "cache_seconds": self.cache_seconds, "final_chunk_embedding_seconds": 0,
                  "final_chunk_embeddings_created": False}
        if self.model is not None and self.identity["embedding"]["device"] == "cuda":
            import torch
            result.update(gpu_name=torch.cuda.get_device_name(), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
        return result

    def close(self):
        self.connection.close()


def model_factory(model_config, device):
    def load():
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import torch
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return JinaEmbeddings.from_config(model_config, device=device)
    return load
