"""Run the fixed-size baseline on validated LangChain documents and save evidence."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import time

from ...common.chunk_schema import make_chunk
from ...common.config import ROOT, load_config, within
from ...common.langchain_loader import PreparedCorpusLoader
from ...common.load_corpus import digest_json, file_hash, require
from ...common.run_manifest import create_run, environment, finish_run, now, write_json
from ...common.tokenization import experiment_config, load_tokenizer
from .chunker import FixedSizeSplitter, load_settings


def run_fixed_size(run_id: str, *, config_path: Path = ROOT / "rag_pipeline/config.json",
                   method_config: Path = ROOT / "rag_pipeline/configs/fixed_size.json",
                   model_config: Path = ROOT / "rag_pipeline/configs/models.json",
                   sample: bool = False) -> Path:
    config = load_config(config_path)
    settings = load_settings(method_config)
    tokens = load_tokenizer(model_config)
    splitter = FixedSizeSplitter(tokens, settings["chunk_size"])
    effective = experiment_config(settings, tokens, config.gap_policy)
    run, manifest = create_run(ROOT, config, run_id)
    manifest.update(stage="phase3_fixed_size", config_hash=effective["config_hash"])
    write_json(run / "stage_manifest.json", manifest)
    variant = run / "fixed_size/variant_001"
    variant.mkdir(parents=True)
    stage = {"schema_version": 1, "stage": "chunking", "method": "fixed_size", "run_id": run_id,
             "status": "running", "started_at_utc": now(), "config_hash": effective["config_hash"]}
    write_json(variant / "stage_manifest.json", stage)
    try:
        started = time.perf_counter()
        write_json(run / "run_config.json", {"corpus": config.to_dict(), "experiment": effective,
                                           "sample": sample, "model_settings": json.loads(model_config.read_text())})
        write_json(run / "environment.json", environment(ROOT))
        loader = PreparedCorpusLoader(config_path)
        regions = loader.load()
        corpus = loader.corpus
        write_json(run / "corpus_snapshot.json", corpus.snapshot)
        sources = {d.document_id: d for d in corpus.documents}
        selected = set(sources)
        if sample:
            selected = {next(d.document_id for d in sorted(corpus.documents, key=lambda d: d.source)
                             if d.member == member and d.regions)
                        for member in sorted({d.member for d in corpus.documents})}
        regions = [region for region in regions if region.metadata["document_id"] in selected]

        counts, sizes, totals, examples, hashes = Counter(), [], Counter(), {}, []
        per_source = defaultdict(lambda: {"chunks": 0, "characters": 0, "content_tokens": 0})
        with (variant / "chunks.jsonl").open("x", encoding="utf-8") as stream:
            for region in regions:
                doc = sources[region.metadata["document_id"]]
                pieces = splitter.split_documents([region])
                # Stronger than non-whitespace coverage: every region character,
                # including newlines and ignored tokenizer glyphs, is retained.
                require("".join(p.page_content for p in pieces) == region.page_content, "Region coverage differs")
                expected_start = region.metadata["char_start"]
                for piece in pieces:
                    meta = piece.metadata
                    require(meta["char_start"] == expected_start, "Overlap or gap in fixed-size spans")
                    expected_start = meta["char_end"]
                    record = make_chunk(doc, start=meta["char_start"], end=meta["char_end"], method="fixed_size",
                                        corpus_hash=corpus.corpus_hash, config_hash=effective["config_hash"], run_id=run_id,
                                        chunk_index=counts[doc.document_id], token_count=meta["token_count"],
                                        tokenizer_id=meta["tokenizer_id"], method_metadata=meta["method_metadata"]).to_dict()
                    require(record["text"] == piece.page_content, "LangChain/source slice mismatch")
                    require(list(record["page_numbers"]) == meta["page_numbers"], "LangChain/source page mismatch")
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    counts[doc.document_id] += 1
                    count = record["method_metadata"]["content_token_count"]
                    sizes.append(count)
                    totals["characters"] += len(record["text"])
                    totals["encoder_input_tokens"] += record["token_count"]
                    totals["content_tokens"] += count
                    totals["boundary_backoffs"] += record["method_metadata"]["boundary_backoffs"]
                    totals["cross_page_chunks"] += len(record["page_numbers"]) > 1
                    per_source[doc.source]["chunks"] += 1
                    per_source[doc.source]["characters"] += len(record["text"])
                    per_source[doc.source]["content_tokens"] += count
                    hashes.append(digest_json({k: v for k, v in record.items() if k != "run_id"}))
                    examples.setdefault(doc.member, record)
                    if len(record["page_numbers"]) > 1:
                        examples.setdefault("cross_page", record)
                    if "table_structure_review" in record["quality_flags"]:
                        examples.setdefault("table_flagged", record)
                    examples["last_chunk"] = record
                require(expected_start == region.metadata["char_end"], "Missing region tail")
        require(bool(sizes), "No chunks produced")
        stats = {"method": "fixed_size", "scope": "member_sample" if sample else "full_corpus",
                 "documents": len(selected), "regions": len(regions), "chunks": len(sizes),
                 "chunk_size_content_tokens": settings["chunk_size"], "overlap": 0,
                 "content_token_count": {"min": min(sizes), "max": max(sizes), "mean": statistics.mean(sizes),
                                         "median": statistics.median(sizes), "histogram": dict(sorted(Counter(sizes).items()))},
                 "token_count_convention": "encoder input tokens including two special tokens",
                 "max_encoder_input_tokens": max(sizes) + tokens.special_tokens,
                 "retained_region_character_coverage": 1.0, "unintended_overlap_characters": 0,
                 "empty_chunks": 0, "overflow_chunks": 0, "totals": dict(totals),
                 "per_document": dict(per_source), "elapsed_seconds": time.perf_counter() - started,
                 "canonical_records_sha256": digest_json(hashes),
                 "canonical_hash_note": "ordered record hashes excluding run_id; timings are not chunk content",
                 "corpus_hash": corpus.corpus_hash, "config_hash": effective["config_hash"]}
        write_json(variant / "chunk_stats.json", stats)
        write_json(variant / "inspection_examples.json", examples)
        lines = ["# Fixed-size chunk inspection", "", f"Scope: {stats['scope']}; {len(sizes):,} chunks.", "",
                 "Each excerpt below is the complete saved chunk, not a reflowed preview.", ""]
        for label, record in examples.items():
            lines += [f"## {label}", "", f"Source: `{record['source']}`; physical pages: {list(record['page_numbers'])}.",
                      f"Span: [{record['char_start']}, {record['char_end']}); input tokens: {record['token_count']}.", "",
                      "~~~~text", record["text"], "~~~~", ""]
        (variant / "inspection_report.md").write_text("\n".join(lines), encoding="utf-8")
        for name, expected in corpus.snapshot["files"].items():
            require(file_hash(within(ROOT / config.corpus_dir, name)) == expected, f"Input changed during run: {name}")
        stage["corpus_hash"] = manifest["corpus_hash"] = corpus.corpus_hash
        finish_run(variant, stage)
        finish_run(run, manifest)
        print(json.dumps({"output": str(variant), "documents": len(selected), "chunks": len(sizes),
                          "max_content_tokens": max(sizes), "coverage": 1.0}, indent=2))
    except Exception as exc:
        finish_run(variant, stage, error=f"{type(exc).__name__}: {exc}")
        finish_run(run, manifest, error=f"{type(exc).__name__}: {exc}")
        raise
    return variant
