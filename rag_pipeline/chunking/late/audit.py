"""Independently re-read late artifacts, source spans and cached window vectors.

Run with --run-id NAME and optionally --repeat-id NAME. The audit writes a new
report beside the run, so a completed run's artifact hashes remain unchanged.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

from ...common.chunk_schema import make_chunk
from ...common.config import ROOT, load_config, validate_run_id, within
from ...common.load_corpus import digest_json, file_hash, load_corpus, require
from ...common.provenance import region_for_span
from ...common.run_manifest import write_json
from ...common.tokenization import load_tokenizer
from ...embedding.inputs import verify_manifest


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def audit(run_id, repeat_id=None):
    validate_run_id(run_id)
    run = ROOT / "data/outputs/rag" / run_id
    verify_manifest(run)
    variant = run / "late/variant_001"
    verify_manifest(variant)
    effective = json.loads((run / "run_config.json").read_text())
    saved = json.loads((run / "embedding_config.json").read_text())
    corpus = load_corpus(ROOT / "final_data", load_config(ROOT / "rag_pipeline/config.json"))
    tokens = load_tokenizer()
    documents = {d.document_id: d for d in corpus.documents}
    for name, checksum in effective["input_sha256"].items():
        require(file_hash(within(ROOT, name)) == checksum, "Paired baseline changed")
    early = {r["chunk_id"]: r for r in lines(within(ROOT, effective["settings"]["early_variant"]) / "chunks.jsonl")}
    chunks = lines(variant / "chunks.jsonl")
    mapping = lines(variant / "embedding_rows.jsonl")
    windows = lines(variant / "encoder_windows.jsonl")
    vectors = np.load(variant / "embeddings.npy", allow_pickle=False)
    require(vectors.shape == (len(chunks), 512) and vectors.dtype == np.dtype("float32"), "Matrix shape/type differs")
    require(np.isfinite(vectors).all() and np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4, rtol=0),
            "Invalid vectors")
    require(len(mapping) == len(chunks), "Missing vector mappings")
    cache_path = within(ROOT, saved["cache_path"])
    connection = sqlite3.connect(cache_path.as_uri() + "?mode=ro", uri=True)
    pooled, owners, window_lookup = {}, {}, {}
    try:
        for window in windows:
            wid = window["window_id"]
            require(wid not in window_lookup, "Duplicate window ID")
            window_lookup[wid] = window
            identity = {k: v for k, v in window.items() if k not in {"window_id", "cache_key"}}
            require(digest_json(identity) == wid, "Window identity differs")
            doc = documents[window["document_id"]]
            start, end = window["char_start"], window["char_end"]
            require(region_for_span(doc, start, end).region_id == window["region_id"], "Window crosses a gap")
            text = doc.text[start:end]
            offsets = tokens.offsets(text)
            require(tokens.count(text) == window["input_token_count"] <= effective["settings"]["max_input_tokens"],
                    "Window input length differs/overflows")
            require(len(offsets) == window["content_token_count"], "Content count differs")
            key = digest_json({"schema_version": 1, "embedding_hash": saved["embedding_hash"],
                               "window_text_sha256": hashlib.sha256(text.encode()).hexdigest(), "spans": window["spans"]})
            require(key == window["cache_key"], "Cache lacks correct context identity")
            cached = connection.execute("SELECT vectors, checksum FROM windows WHERE key=?", (key,)).fetchone()
            require(cached is not None and hashlib.sha256(cached[0]).hexdigest() == cached[1], "Missing/corrupt window cache")
            matrix = np.frombuffer(cached[0], dtype="<f4").reshape(len(window["early_chunk_ids"]), 512)
            require(np.isfinite(matrix).all() and np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-4, rtol=0),
                    "Invalid window cache vectors")
            require(len(window["spans"]) == len(window["token_indices"]) == len(window["early_chunk_ids"]),
                    "Window alignment row counts differ")
            all_indices = []
            for position, (cid, span, indices) in enumerate(zip(window["early_chunk_ids"], window["spans"], window["token_indices"])):
                source = early[cid]
                require(source["document_id"] == doc.document_id and source["region_id"] == window["region_id"], "Mixed sources")
                require(span == [source["char_start"]-start, source["char_end"]-start], "Wrong window-local span")
                hits = [i for i, (a, b) in enumerate(offsets) if b > span[0] and a < span[1]]
                require(hits == indices and bool(hits), "Token pooling selection differs")
                require(all(span[0] <= offsets[i][0] < offsets[i][1] <= span[1] for i in hits), "Neighbor token leaked")
                all_indices.extend(hits)
                pooled[(wid, cid)] = matrix[position].copy()
                if cid in window["owned_chunk_ids"]:
                    require(cid not in owners, "Duplicate ownership")
                    owners[cid] = wid
            require(sorted(all_indices) == list(range(len(offsets))), "Missing/repeated contextual tokens")
        require(len(owners) == len(chunks), "Missing owned chunks")
        ids = set()
        for index, (chunk, record) in enumerate(zip(chunks, mapping)):
            meta = chunk["method_metadata"]
            cid, wid = meta["early_chunk_id"], meta["encoder_window_id"]
            require(chunk["chunk_id"] not in ids, "Duplicate late chunk ID")
            ids.add(chunk["chunk_id"])
            source = early[cid]
            window = window_lookup[wid]
            position = window["early_chunk_ids"].index(cid)
            indices = window["token_indices"][position]
            require(meta["encoder_char_start"] == window["char_start"] and meta["encoder_char_end"] == window["char_end"]
                    and meta["encoder_input_tokens"] == window["input_token_count"], "Chunk window metadata differs")
            require(meta["window_content_token_indices"] == indices and meta["pooled_content_tokens"] == len(indices)
                    and meta["left_context_tokens"] == indices[0]
                    and meta["right_context_tokens"] == window["content_token_count"]-indices[-1]-1,
                    "Chunk token context metadata differs")
            for key in ("text", "char_start", "char_end", "page_numbers", "region_id", "document_id", "source",
                        "member", "chunk_index", "token_count", "tokenizer_id", "corpus_hash"):
                require(chunk[key] == source[key], f"Baseline changed: {key}")
            rebuilt = make_chunk(documents[chunk["document_id"]], start=chunk["char_start"], end=chunk["char_end"],
                                 method="late", corpus_hash=corpus.corpus_hash, config_hash=effective["config_hash"],
                                 run_id=run_id, chunk_index=chunk["chunk_index"], token_count=tokens.count(chunk["text"]),
                                 tokenizer_id=tokens.tokenizer_id, section_path=tuple(chunk["section_path"]),
                                 method_metadata=meta).to_dict()
            require(digest_json(rebuilt) == digest_json(chunk), "Late identity/provenance differs")
            require(owners[cid] == wid and np.array_equal(vectors[index], pooled[(wid, cid)]), "Wrong owned vector")
            require(record["row_index"] == index and record["chunk_id"] == chunk["chunk_id"]
                    and record["early_chunk_id"] == cid and record["encoder_window_id"] == wid, "Wrong vector row")
            require(record["cache_key"] == window["cache_key"] and record["embedding_hash"] == saved["embedding_hash"]
                    and record["text_sha256"] == hashlib.sha256(chunk["text"].encode()).hexdigest(), "Vector identity differs")
            require(record["vector_sha256"] == hashlib.sha256(vectors[index].astype("<f4").tobytes()).hexdigest(), "Vector checksum differs")
        stats = json.loads((variant / "chunk_stats.json").read_text())
        require(stats["chunks"] == len(chunks) and stats["windows"] == len(windows), "Statistics differ")
        if not effective["sample"]:
            require([r["method_metadata"]["early_chunk_id"] for r in chunks] == list(early), "Full baseline coverage/order differs")
        seams = lines(variant / "window_seams.jsonl")
        for seam in seams:
            left = pooled[(seam["owner_window_id"], seam["early_chunk_id"])].astype(float)
            right = pooled[(seam["alternate_window_id"], seam["early_chunk_id"])].astype(float)
            score = float(np.clip(np.dot(left, right)/(np.linalg.norm(left)*np.linalg.norm(right)), -1, 1))
            require(abs(score-seam["cosine_similarity"]) < 1e-12, "Seam diagnostic differs")
        repeat = None
        if repeat_id:
            validate_run_id(repeat_id)
            other = run.parent / repeat_id
            verify_manifest(other)
            other = other / "late/variant_001"
            for name in ("embeddings.npy", "embedding_rows.jsonl", "encoder_windows.jsonl", "window_seams.jsonl", "early_late_comparison.jsonl"):
                require(file_hash(other / name) == file_hash(variant / name), f"Repeat differs: {name}")
            canonical = lambda records: [{k: v for k, v in row.items() if k != "run_id"} for row in records]
            require(canonical(chunks) == canonical(lines(other / "chunks.jsonl")), "Repeated chunk records differ")
            repeat = {"run_id": repeat_id, "vectors_and_mapping_byte_identical": True, "canonical_chunks_identical": True}
        phase1 = json.loads((run.parent / "phase1_clean_environment_20260919/corpus_snapshot.json").read_text())
        require(corpus.snapshot["files"] == phase1["files"], "Prepared inputs differ from Phase 1")
        report = {"passed": True, "run_id": run_id, "chunks": len(chunks), "windows": len(windows),
                  "source_spans_and_pages": True, "exact_baseline_boundaries": True, "complete_unique_ownership": True,
                  "no_gap_crossing_or_pooling_leakage": True, "context_cache_and_vectors": True,
                  "seam_comparisons_verified": len(seams), "unchanged_prepared_files": len(corpus.snapshot["files"]),
                  "unchanged_early_artifacts": len(effective["input_sha256"]), "repeat": repeat}
        destination = run.parent / (run_id + "_audit.json")
        require(not destination.exists(), "Audit report exists; preserve it and use a new run ID")
        write_json(destination, report)
        print(json.dumps(report, indent=2))
        return report
    finally:
        connection.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repeat-id")
    args = parser.parse_args()
    audit(args.run_id, args.repeat_id)
