"""Validate completed chunk artifacts against the frozen prepared corpus."""

from collections import defaultdict
import json
from pathlib import Path

from ..common.chunk_schema import make_chunk
from ..common.config import ROOT, within
from ..common.load_corpus import digest_json, file_hash, require

METHODS = ("fixed_size", "sliding_window", "structure_aware", "semantic")


def verify_manifest(directory):
    manifest = json.loads((directory / "stage_manifest.json").read_text())
    require(manifest["status"] == "complete", f"Incomplete source stage: {directory}")
    for name, expected in manifest["artifact_sha256"].items():
        require(file_hash(within(directory, name)) == expected, f"Artifact checksum mismatch: {directory / name}")
    return manifest


def validate_rows(rows, corpus, tokens, method, config_hash, cap):
    """Reject stale source text, gaps, duplicate IDs, wrong counts and overflow."""
    docs = {d.document_id: d for d in corpus.documents}
    ids, by_region, by_doc = set(), defaultdict(list), defaultdict(list)
    for row in rows:
        require(row["method"] == method and row["config_hash"] == config_hash, "Mixed method/config chunks")
        require(row["corpus_hash"] == corpus.corpus_hash, "Chunk corpus identity differs")
        require(row["document_id"] in docs, "Unknown chunk source")
        require(row["chunk_id"] not in ids, "Duplicate chunk ID")
        ids.add(row["chunk_id"])
        text = row["text"]
        require(isinstance(text, str) and bool(text.strip()), "Empty chunk")
        count = tokens.count(text)
        content = tokens.count(text, special_tokens=False)
        require(0 < content <= cap and count <= tokens.max_input_tokens, "Chunk exceeds encoder or method limit")
        require(row["token_count"] == count and row["tokenizer_id"] == tokens.tokenizer_id
                and row["method_metadata"]["content_token_count"] == content, "Chunk token metadata mismatch")
        rebuilt = make_chunk(docs[row["document_id"]], start=row["char_start"], end=row["char_end"],
                             method=method, corpus_hash=corpus.corpus_hash, config_hash=config_hash,
                             run_id=row["run_id"], chunk_index=row["chunk_index"], token_count=count,
                             tokenizer_id=tokens.tokenizer_id, section_path=tuple(row["section_path"]),
                             method_metadata=row["method_metadata"]).to_dict()
        require(digest_json(rebuilt) == digest_json(row), "Chunk source text, identity or provenance mismatch")
        by_region[(row["document_id"], row["region_id"])].append(row)
        by_doc[row["document_id"]].append(row)
    require(set(by_doc) == set(docs), "Each input must contain the complete corpus")
    for doc in corpus.documents:
        require([r["chunk_index"] for r in by_doc[doc.document_id]] == list(range(len(by_doc[doc.document_id]))),
                "Chunk document order differs")
        for region in doc.regions:
            chunks = by_region[(doc.document_id, region.region_id)]
            require(bool(chunks), "Missing eligible region")
            require(chunks[0]["char_start"] == region.char_start and chunks[-1]["char_end"] == region.char_end,
                    "Missing region beginning or tail")
            for left, right in zip(chunks, chunks[1:]):
                require(left["char_start"] < right["char_start"] <= left["char_end"] < right["char_end"],
                        "Gap, reordered chunk or duplicate tail")
                require(method == "sliding_window" or left["char_end"] == right["char_start"], "Unintended overlap")


def load_inputs(path, corpus, tokens, *, sample=False):
    settings = json.loads(Path(path).read_text())
    require(set(settings) == {"schema_version", "batch_size", "inputs"} and settings["schema_version"] == 1, "Invalid embedding input config")
    require(type(settings["batch_size"]) is int and settings["batch_size"] > 0, "Invalid embedding batch size")
    require(set(settings["inputs"]) == set(METHODS), "Provide exactly the first four methods")
    selected = {next(d.document_id for d in sorted(corpus.documents, key=lambda d: d.source) if d.member == member and d.regions)
                for member in sorted({d.member for d in corpus.documents})} if sample else {d.document_id for d in corpus.documents}
    outputs, fingerprints = {}, {}
    for method in METHODS:
        source = within(ROOT, settings["inputs"][method])
        require(source.name == "chunks.jsonl" and source.is_relative_to(ROOT / "data/outputs/rag"), "Invalid chunk input path")
        variant = source.parent
        manifest = verify_manifest(variant)
        require(manifest["method"] == method and manifest["corpus_hash"] == corpus.corpus_hash, "Source manifest identity differs")
        require("chunks.jsonl" in manifest["artifact_sha256"] and "chunk_stats.json" in manifest["artifact_sha256"], "Unhashed chunk input")
        stats = json.loads((variant / "chunk_stats.json").read_text())
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        require(stats["scope"] == "full_corpus" and stats["chunks"] == len(rows), "Expected completed full-corpus chunks")
        require(stats["corpus_hash"] == corpus.corpus_hash and stats["config_hash"] == manifest["config_hash"], "Chunk statistics identity differs")
        require(digest_json([digest_json({k: v for k, v in row.items() if k != "run_id"}) for row in rows])
                == stats["canonical_records_sha256"], "Chunk canonical hash differs")
        validate_rows(rows, corpus, tokens, method, manifest["config_hash"], stats["chunk_size_content_tokens"])
        outputs[method] = [row for row in rows if row["document_id"] in selected]
        for name in ("chunks.jsonl", "chunk_stats.json", "stage_manifest.json"):
            file = variant / name
            fingerprints[file.relative_to(ROOT).as_posix()] = file_hash(file)
    return outputs, fingerprints
