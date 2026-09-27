"""Atomic window cache: surrounding text is part of every cache identity."""

import hashlib
import json
import sqlite3
import time

import numpy as np

from ...common.load_corpus import digest_json, require
from ...embedding.vector_cache import check_vector
from .chunker import pool_spans


class WindowCache:
    def __init__(self, path, identity, factory, *, cache_only=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=30)
        self.connection.execute("CREATE TABLE IF NOT EXISTS windows (key TEXT PRIMARY KEY, vectors BLOB NOT NULL, checksum TEXT NOT NULL)")
        self.identity_hash = digest_json(identity)
        self.dimension = identity["dimension"]
        self.factory, self.cache_only, self.model = factory, cache_only, None
        self.hits = self.misses = self.input_tokens = 0
        self.inference_seconds = self.load_seconds = 0.0
        self.parity_probe = None

    def key(self, text, spans):
        return digest_json({"schema_version": 1, "embedding_hash": self.identity_hash,
                            "window_text_sha256": hashlib.sha256(text.encode()).hexdigest(), "spans": spans})

    def encode(self, text, window):
        spans = window["spans"]
        key = self.key(text, spans)
        row = self.connection.execute("SELECT vectors, checksum FROM windows WHERE key=?", (key,)).fetchone()
        if row is None:
            require(not self.cache_only, "Late window cache miss in --cache-only mode")
            if self.model is None:
                started = time.perf_counter()
                self.model = self.factory()
                self.load_seconds += time.perf_counter()-started
            started = time.perf_counter()
            encoded, hidden = self.model.encode_tokens([text])
            vectors, groups = pool_spans(hidden, encoded, spans)
            # Confirm actual encoder offsets agree with tokenizer-only planning.
            content_positions = [i for i, (a, s) in enumerate(zip(
                encoded["attention_mask"][0].tolist(), encoded["special_tokens_mask"][0].tolist())) if a and not s]
            require(groups == [[content_positions[i] for i in indices] for indices in window["token_indices"]],
                    "Runtime token alignment differs from planned offsets")
            require(len(content_positions) == window["content_token_count"]
                    and encoded["input_ids"].shape[1] == window["input_token_count"], "Runtime token count differs")
            self.inference_seconds += time.perf_counter()-started
            self.input_tokens += window["input_token_count"]
            if self.parity_probe is None:
                # The same whole-window span must reproduce independent encoding.
                # This extra forward pass is diagnostic and timed separately.
                started = time.perf_counter()
                whole, _ = pool_spans(hidden, encoded, [(0, len(text))])
                early = np.asarray(self.model.embed_documents([text]), dtype="<f4")
                error = float(np.max(np.abs(whole-early)))
                require(error <= 1e-6, "Whole-window early/late pooling parity failed")
                self.parity_probe = {"max_absolute_difference": error, "tolerance": 1e-6,
                                     "input_tokens": window["input_token_count"], "passed": True,
                                     "elapsed_seconds": time.perf_counter()-started}
            del hidden, encoded
            data = np.stack([check_vector(v, self.dimension) for v in vectors]).astype("<f4").tobytes()
            checksum = hashlib.sha256(data).hexdigest()
            with self.connection:
                self.connection.execute("INSERT OR IGNORE INTO windows VALUES (?, ?, ?)", (key, data, checksum))
            row = self.connection.execute("SELECT vectors, checksum FROM windows WHERE key=?", (key,)).fetchone()
            self.misses += 1
        else:
            self.hits += 1
        data, checksum = row
        require(hashlib.sha256(data).hexdigest() == checksum, "Late cache checksum mismatch")
        require(len(data) == len(spans)*self.dimension*4, "Late cache shape mismatch")
        values = np.frombuffer(data, dtype="<f4").reshape(len(spans), self.dimension)
        for value in values:
            check_vector(value, self.dimension)
        return values.copy(), key

    def stats(self):
        memory = {"gpu_inference_used": False}
        if self.model is not None and self.model.device == "cuda":
            import torch
            memory = {"gpu_inference_used": True, "gpu_name": torch.cuda.get_device_name(),
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        return {"cache_hits": self.hits, "cache_misses": self.misses,
                "inferred_input_tokens": self.input_tokens, "inference_seconds": self.inference_seconds,
                "model_load_seconds": self.load_seconds, "whole_window_parity_probe": self.parity_probe,
                "gpu_memory": memory}

    def close(self):
        self.connection.close()
