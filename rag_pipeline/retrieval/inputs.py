"""Verify five completed vector sets and compatibility with the shared query model."""

import json
from pathlib import Path

import numpy as np

from ..common.config import ROOT, within
from ..common.load_corpus import digest_json, file_hash, require
from ..embedding.encode_chunks import validate_outputs
from ..embedding.inputs import validate_rows, verify_manifest

METHODS = ("fixed_size", "sliding_window", "structure_aware", "semantic", "late")


def load_settings(path):
    settings = json.loads(Path(path).read_text())
    require(set(settings) == {"schema_version", "inputs", "pilot", "benchmark_manifest", "k_values",
                             "context_content_tokens", "similarity", "tie_policy", "overlap_policy",
                             "oversize_policy", "faiss_threads"}, "Invalid retrieval settings")
    require(settings["schema_version"] == 1 and set(settings["inputs"]) == set(METHODS), "Provide five method inputs")
    require(settings["k_values"] == [1, 3, 5, 10], "This protocol requires K=1,3,5,10")
    require(type(settings["context_content_tokens"]) is int and settings["context_content_tokens"] > 0, "Invalid evidence budget")
    for name, expected in {"similarity": "normalized_inner_product", "tie_policy": "score_descending_chunk_id_ascending",
                           "overlap_policy": "subtract_included_source_spans", "oversize_policy": "skip_candidate_try_next",
                           "faiss_threads": 1}.items():
        require(settings[name] == expected, f"Unsupported {name}")
    return settings


def compatible_identity(current, saved, method):
    """Batch size/purpose differ for late encoding; model and vector space do not."""
    require(saved["purpose"] == ("late_contextual_chunk_embedding" if method == "late" else "independent_chunk_embedding"),
            "Wrong document embedding purpose")
    for key, value in current.items():
        if key in {"purpose", "batch_policy"}:
            continue
        if key == "embedding":
            value = {k: v for k, v in value.items() if k != "batch_size"}
            expected = {k: v for k, v in saved[key].items() if k != "batch_size"}
        else:
            expected = saved.get(key)
        require(value == expected, f"Query/document embedding identity differs: {key}")


def load_inputs(settings, corpus, tokens, query_model_identity):
    inputs, fingerprints, verified_roots = {}, {}, set()
    for method in METHODS:
        variant = within(ROOT, settings["inputs"][method])
        require(variant.is_relative_to(ROOT / "data/outputs/rag"), "Vector inputs must be research artifacts")
        root = variant.parent.parent
        if root not in verified_roots:
            verify_manifest(root)
            verified_roots.add(root)
        manifest = verify_manifest(variant)
        require(manifest["method"] == method and manifest["corpus_hash"] == corpus.corpus_hash, "Source method/corpus differs")
        config = json.loads((root / "embedding_config.json").read_text())
        require(digest_json(config["identity"]) == config["embedding_hash"], "Embedding identity hash differs")
        compatible_identity(query_model_identity, config["identity"], method)
        chunks = [json.loads(line) for line in (variant / "chunks.jsonl").read_text().splitlines()]
        require(bool(chunks), "Empty source chunks")
        validate_rows(chunks, corpus, tokens, method, chunks[0]["config_hash"], 256)
        validate_outputs(variant, chunks, config["identity"]["dimension"], config["embedding_hash"])
        vectors = np.load(variant / "embeddings.npy", allow_pickle=False)
        paths = [variant / name for name in ("chunks.jsonl", "embeddings.npy", "embedding_rows.jsonl", "stage_manifest.json")]
        paths += [root / "embedding_config.json", root / "stage_manifest.json"]
        hashes = {path.relative_to(ROOT).as_posix(): file_hash(path) for path in paths}
        fingerprints.update(hashes)
        inputs[method] = {"chunks": chunks, "vectors": vectors, "embedding_hash": config["embedding_hash"], "hashes": hashes}
    fixed = {r["chunk_id"]: r for r in inputs["fixed_size"]["chunks"]}
    linked = []
    for row in inputs["late"]["chunks"]:
        parent = row["method_metadata"]["early_chunk_id"]
        require(parent in fixed, "Late chunk has no matched early baseline")
        for key in ("text", "document_id", "char_start", "char_end", "page_numbers"):
            require(row[key] == fixed[parent][key], "Late/early target spans differ")
        linked.append(parent)
    require(linked == list(fixed), "Late/early baseline order or coverage differs")
    return inputs, fingerprints
