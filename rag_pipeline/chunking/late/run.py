"""Run the matched late experiment using completed Phase 7 fixed-size vectors."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np

from ...common.chunk_schema import make_chunk
from ...common.config import ROOT, load_config, within
from ...common.load_corpus import digest_json, file_hash, load_corpus, require
from ...common.run_manifest import create_run, environment, finish_run, write_json
from ...common.tokenization import load_tokenizer
from ...embedding.encode_chunks import atomic_jsonl, validate_outputs
from ...embedding.inputs import validate_rows, verify_manifest
from ...embedding.vector_cache import embedding_identity, model_factory
from .cache import WindowCache
from .chunker import plan_windows


def load_settings(path):
    settings = json.loads(Path(path).read_text())
    require(set(settings) == {"schema_version", "early_variant", "core_chunks", "halo_chunks", "max_input_tokens"}
            and settings["schema_version"] == 1, "Invalid late configuration")
    for key, minimum in (("core_chunks", 1), ("halo_chunks", 0), ("max_input_tokens", 3)):
        require(type(settings[key]) is int and settings[key] >= minimum, f"Invalid {key}")
    return settings


def preflight(settings, corpus, tokens, independent_identity):
    early = within(ROOT, settings["early_variant"])
    require(early.is_relative_to(ROOT / "data/outputs/rag"), "Early artifacts must be in research output storage")
    verify_manifest(early)
    early_run = early.parent.parent
    root_manifest = verify_manifest(early_run)
    saved = json.loads((early_run / "embedding_config.json").read_text())
    expected = saved["identity"]
    # Only inference batch size changes: long windows use batch one on the 4 GB GPU.
    comparable = independent_identity | {"embedding": independent_identity["embedding"] | {
        "batch_size": expected["embedding"]["batch_size"]}}
    require(comparable == expected and digest_json(expected) == saved["embedding_hash"],
            "Early/late model, tokenizer, pooling, precision, device or implementation differ")
    require(root_manifest["corpus_hash"] == corpus.corpus_hash, "Early corpus differs")
    rows = [json.loads(line) for line in (early / "chunks.jsonl").read_text().splitlines()]
    require(bool(rows), "Empty early baseline")
    validate_rows(rows, corpus, tokens, "fixed_size", rows[0]["config_hash"], 256)
    validate_outputs(early, rows, expected["dimension"], saved["embedding_hash"])
    paths = [early / name for name in ("chunks.jsonl", "embeddings.npy", "embedding_rows.jsonl", "stage_manifest.json")]
    paths += [early_run / "embedding_config.json", early_run / "stage_manifest.json"]
    return rows, np.load(early / "embeddings.npy", allow_pickle=False), {
        path.relative_to(ROOT).as_posix(): file_hash(path) for path in paths}, saved["embedding_hash"]


def cosine(left, right):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    return float(np.clip(np.dot(left, right)/(np.linalg.norm(left)*np.linalg.norm(right)), -1, 1))


def distribution(values):
    return {"count": len(values), "minimum": float(min(values)), "mean": float(np.mean(values)),
            "maximum": float(max(values))} if values else {"count": 0}


def run_late(run_id, *, config_path=ROOT / "rag_pipeline/config.json",
             method_config=ROOT / "rag_pipeline/configs/late.json",
             model_config=ROOT / "rag_pipeline/configs/models.json", sample=False, cache_only=False):
    started = time.perf_counter()
    config, tokens = load_config(Path(config_path)), load_tokenizer(Path(model_config))
    settings = load_settings(method_config)
    independent = embedding_identity(model_config, tokens, batch_size=1)
    corpus = load_corpus(ROOT / config.corpus_dir, config)
    baseline, early_vectors, input_hashes, early_hash = preflight(settings, corpus, tokens, independent)
    if sample:
        selected = {next(d.document_id for d in sorted(corpus.documents, key=lambda d: d.source)
                         if d.member == member and d.regions) for member in sorted({d.member for d in corpus.documents})}
    else:
        selected = {d.document_id for d in corpus.documents}
    early_index = {r["chunk_id"]: i for i, r in enumerate(baseline)}
    baseline = [r for r in baseline if r["document_id"] in selected]
    documents = {d.document_id: d for d in corpus.documents}
    windows = plan_windows(baseline, documents, tokens, **{k: settings[k] for k in
                           ("core_chunks", "halo_chunks", "max_input_tokens")})
    identity = independent | {"purpose": "late_contextual_chunk_embedding", "batch_policy": "one_context_window",
                              "window_policy": {k: settings[k] for k in ("core_chunks", "halo_chunks", "max_input_tokens")},
                              "late_implementation_sha256": {name: file_hash(Path(__file__).with_name(name))
                                                             for name in ("chunker.py", "cache.py")}}
    embedding_hash = digest_json(identity)
    # Sample/full scope and execution ID must not change the identity of a chunk.
    method_identity = {"schema_version": 1, "embedding_hash": embedding_hash,
                       "early_config_hash": baseline[0]["config_hash"], "gap_policy": config.gap_policy}
    config_hash = digest_json(method_identity)
    effective = {"settings": settings, "identity": identity, "method_identity": method_identity,
                 "config_hash": config_hash, "sample": sample, "input_sha256": input_hashes,
                 "early_embedding_hash": early_hash, "writer_sha256": file_hash(Path(__file__))}
    preflight_seconds = time.perf_counter()-started
    run, manifest = create_run(ROOT, config, run_id)
    variant = run / "late/variant_001"
    variant.mkdir(parents=True)
    manifest.update(stage="phase8_late_chunking", method="late", config_hash=config_hash, corpus_hash=corpus.corpus_hash)
    write_json(run / "stage_manifest.json", manifest)
    cache = None
    try:
        write_json(run / "run_config.json", effective)
        write_json(run / "environment.json", environment(ROOT))
        write_json(run / "corpus_snapshot.json", corpus.snapshot)
        cache_path = ROOT / config.output_root / "late_window_cache" / (embedding_hash + ".sqlite3")
        cache = WindowCache(cache_path, identity, model_factory(model_config, identity["embedding"]["device"], 1),
                            cache_only=cache_only)
        write_json(run / "embedding_config.json", {"identity": identity, "embedding_hash": embedding_hash,
                                                   "cache_path": cache_path.relative_to(ROOT).as_posix()})
        owned, occurrences = {}, defaultdict(list)
        for number, window in enumerate(windows):
            doc = documents[window["document_id"]]
            text = doc.text[window["char_start"]:window["char_end"]]
            values, cache_key = cache.encode(text, window)
            window["cache_key"] = cache_key
            for index, (chunk_id, vector) in enumerate(zip(window["early_chunk_ids"], values)):
                occurrences[chunk_id].append((window["window_id"], vector))
                if chunk_id in window["owned_chunk_ids"]:
                    require(chunk_id not in owned, "Duplicate window ownership")
                    owned[chunk_id] = (vector, window, index)
            if number % 100 == 0 or number+1 == len(windows):
                print(f"late: {number+1:,}/{len(windows):,} windows; {len(owned):,}/{len(baseline):,} owned chunks", flush=True)
        records, mapping, comparisons, seams = [], [], [], []
        matrix = np.empty((len(baseline), identity["dimension"]), dtype="<f4")
        for index, early in enumerate(baseline):
            vector, window, position = owned[early["chunk_id"]]
            indices = window["token_indices"][position]
            metadata = {"content_token_count": early["method_metadata"]["content_token_count"],
                        "early_chunk_id": early["chunk_id"], "early_embedding_row": early_index[early["chunk_id"]],
                        "encoder_window_id": window["window_id"], "encoder_char_start": window["char_start"],
                        "encoder_char_end": window["char_end"], "encoder_input_tokens": window["input_token_count"],
                        "pooled_content_tokens": len(indices), "window_content_token_indices": indices,
                        "left_context_tokens": indices[0],
                        "right_context_tokens": window["content_token_count"]-indices[-1]-1}
            row = make_chunk(documents[early["document_id"]], start=early["char_start"], end=early["char_end"],
                             method="late", corpus_hash=corpus.corpus_hash, config_hash=config_hash, run_id=run_id,
                             chunk_index=early["chunk_index"], token_count=early["token_count"],
                             tokenizer_id=tokens.tokenizer_id, section_path=tuple(early["section_path"]),
                             method_metadata=metadata).to_dict()
            records.append(row)
            matrix[index] = vector
            mapping.append({"schema_version": 1, "row_index": index, "chunk_id": row["chunk_id"],
                            "embedding_hash": embedding_hash, "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
                            "vector_sha256": hashlib.sha256(vector.astype("<f4").tobytes()).hexdigest(),
                            "cache_key": window["cache_key"], "encoder_window_id": window["window_id"],
                            "early_chunk_id": early["chunk_id"], **{k: row[k] for k in
                            ("document_id", "source", "member", "char_start", "char_end", "page_numbers", "token_count",
                             "corpus_hash", "config_hash", "method")}})
            comparisons.append({"early_chunk_id": early["chunk_id"], "late_chunk_id": row["chunk_id"],
                                "cosine_similarity": cosine(vector, early_vectors[early_index[early["chunk_id"]]]),
                                "left_context_tokens": metadata["left_context_tokens"],
                                "right_context_tokens": metadata["right_context_tokens"]})
            for other_id, other in occurrences[early["chunk_id"]]:
                if other_id != window["window_id"]:
                    seams.append({"early_chunk_id": early["chunk_id"], "owner_window_id": window["window_id"],
                                  "alternate_window_id": other_id, "cosine_similarity": cosine(vector, other)})
        np.save(variant / "embeddings.npy", matrix, allow_pickle=False)
        atomic_jsonl(variant / "chunks.jsonl", records)
        atomic_jsonl(variant / "embedding_rows.jsonl", mapping)
        atomic_jsonl(variant / "encoder_windows.jsonl", windows)
        atomic_jsonl(variant / "early_late_comparison.jsonl", comparisons)
        atomic_jsonl(variant / "window_seams.jsonl", seams)
        # Re-read the saved matrix and source mapping before completing the stage.
        validation = validate_outputs(variant, json.loads(json.dumps(records)), identity["dimension"], embedding_hash)
        for name, expected in corpus.snapshot["files"].items():
            require(file_hash(within(ROOT / config.corpus_dir, name)) == expected, f"Prepared input changed: {name}")
        for name, expected in input_hashes.items():
            require(file_hash(within(ROOT, name)) == expected, f"Early baseline changed: {name}")
        stats = {"method": "late", "scope": "sample" if sample else "full_corpus", "chunks": len(records),
                 "documents": len(selected), "windows": len(windows), "embedding_hash": embedding_hash,
                 "corpus_hash": corpus.corpus_hash, "config_hash": config_hash,
                 "max_encoder_input_tokens": max(w["input_token_count"] for w in windows),
                 "planned_encoder_input_tokens": sum(w["input_token_count"] for w in windows),
                 "window_budget_adjustments": sum(w["budget_adjustments"] for w in windows),
                 "indexed_content_tokens": sum(r["method_metadata"]["content_token_count"] for r in records),
                 "contextual_token_count_differences": sum(r["method_metadata"]["content_token_count"] !=
                                                          r["method_metadata"]["pooled_content_tokens"] for r in records),
                 "early_late_cosine": distribution([c["cosine_similarity"] for c in comparisons]),
                 "window_seam_cosine": distribution([c["cosine_similarity"] for c in seams]),
                 "canonical_records_sha256": digest_json([{k: v for k, v in r.items() if k != "run_id"} for r in records]),
                 "validation": validation, "prepared_files_unchanged": len(corpus.snapshot["files"]),
                 "early_artifacts_unchanged": len(input_hashes), "preflight_seconds": preflight_seconds,
                 "elapsed_seconds": time.perf_counter()-started,
                 "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                 **cache.stats(), "note": "Cosine changes measure representations, not retrieval/answer improvement. "
                 "A new run reuses committed whole-window cache entries after an interruption; completed runs are immutable."}
        write_json(variant / "chunk_stats.json", stats)
        write_json(variant / "embedding_stats.json", stats)
        # One traceable example with a reference word and available earlier context.
        import re
        example = next((i for i, r in enumerate(records) if re.match(r"\s*(it|they|these|this)\b", r["text"], re.I)
                        and r["method_metadata"]["left_context_tokens"] > 0), 0)
        row, comparison = records[example], comparisons[example]
        doc = documents[row["document_id"]]
        (variant / "inspection_report.md").write_text(
            "# Late chunking inspection\n\n"
            f"{len(records):,} matched chunks; {len(windows):,} context windows.\n\n"
            "Every indexed vector pools only its own chunk's contextual tokens. Neighboring tokens affect attention, "
            "but never enter that chunk's pooling mean. Excluded and blank-page gaps remain barriers.\n\n"
            f"## Reference-word example\n\nSource: {row['source']}; pages: {row['page_numbers']}.\n\n"
            f"Early chunk: {comparison['early_chunk_id']}\n\nLate chunk: {row['chunk_id']}\n\n"
            f"Early/late cosine: {comparison['cosine_similarity']:.6f}. This is not a quality score.\n\n"
            "### Earlier context (last 1,200 characters)\n\n~~~~text\n" +
            doc.text[max(row['method_metadata']['encoder_char_start'], row['char_start']-1200):row['char_start']] +
            "\n~~~~\n\n### Exact indexed chunk\n\n~~~~text\n" + row['text'] +
            "\n~~~~\n\nInspect encoder_windows.jsonl for complete context spans and window_seams.jsonl for "
            "the same chunks pooled under alternate overlapping windows. Differences are diagnostic only.\n", encoding="utf-8")
        finish_run(variant, {"schema_version": 1, "stage": "phase8_late_chunking", "method": "late",
                             "corpus_hash": corpus.corpus_hash, "config_hash": config_hash, "run_id": run_id})
        finish_run(run, manifest)
        print(json.dumps({"output": str(variant), **stats}, indent=2))
    except BaseException as exc:
        finish_run(run, manifest, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if cache is not None:
            cache.close()
    return run
