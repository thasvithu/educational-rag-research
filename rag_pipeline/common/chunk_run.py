"""Shared artifact writer for token-window chunkers; algorithms stay in their method folders."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import time

from .chunk_schema import make_chunk
from .config import ROOT, load_config, within
from .langchain_loader import PreparedCorpusLoader
from .load_corpus import digest_json, file_hash, require
from .run_manifest import create_run, environment, finish_run, now, write_json
from .tokenization import experiment_config


def write_overlap_report(variant: Path, examples: dict) -> None:
    """Show shared text explicitly without inserting markers into saved chunks."""
    lines = ["# Sliding-window overlap inspection", "",
             "One adjacent pair per member. Display labels are not part of indexed text.", ""]
    for member, example in examples.items():
        left, right = example["previous"], example["current"]
        shared = example["shared_text"]
        count = right["metadata"]["method_metadata"]["overlap_with_previous_region_tokens"]
        lines += [f"## {member}: {example['source']}", "",
                  f"Shared source tokens: {count}; shared characters: {len(shared)}.", ""]
        for label, text in (("Previous chunk: unique prefix", left["page_content"][:-len(shared)] if shared else left["page_content"]),
                            ("SHARED TEXT — suffix of previous and prefix of next", shared),
                            ("Next chunk: new suffix", right["page_content"][len(shared):])):
            lines += [f"### {label}", "", "~~~~text", text, "~~~~", ""]
    (variant / "overlap_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_token_chunker(run_id: str, *, config_path: Path, model_config: Path,
                      settings: dict, splitter, tokens, sample: bool = False) -> Path:
    config = load_config(config_path)
    method = settings["method"]
    phase = {"fixed_size": "phase3", "sliding_window": "phase4", "structure_aware": "phase5", "semantic": "phase6"}[method]
    effective = experiment_config(settings, tokens, config.gap_policy)
    run, manifest = create_run(ROOT, config, run_id)
    manifest.update(stage=f"{phase}_{method}", config_hash=effective["config_hash"])
    write_json(run / "stage_manifest.json", manifest)
    variant = run / method / "variant_001"
    variant.mkdir(parents=True)
    stage = {"schema_version": 1, "stage": "chunking", "method": method, "run_id": run_id,
             "status": "running", "started_at_utc": now(), "config_hash": effective["config_hash"]}
    write_json(variant / "stage_manifest.json", stage)
    try:
        started = time.perf_counter()
        write_json(run / "run_config.json", {"corpus": config.to_dict(), "experiment": effective,
                                           "sample": sample, "model_settings": json.loads(model_config.read_text())})
        write_json(run / "environment.json", environment(ROOT))
        if method == "semantic":
            write_json(variant / "boundary_embedding_config.json", {
                "identity": splitter.embeddings.identity, "config_hash": effective["config_hash"],
                "cache_path": splitter.cache_path, "cache_only": splitter.cache_only})
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
        if method == "structure_aware":
            from ..chunking.structure_aware.reporting import prepare_structure
            resolved = prepare_structure(corpus, selected, ROOT / config.corpus_dir, variant)
            for region in regions:
                region.metadata["resolved_headings"] = resolved[region.metadata["document_id"]]

        counts, sizes, totals, examples, hashes = Counter(), [], Counter(), {}, []
        overlap_counts = Counter()
        overlap_examples = {}
        per_source = defaultdict(lambda: {"chunks": 0, "characters": 0, "content_tokens": 0})
        with (variant / "chunks.jsonl").open("x", encoding="utf-8") as stream:
            for region in regions:
                doc = sources[region.metadata["document_id"]]
                if method == "semantic":
                    print(f"Semantic region: {doc.source} / {region.metadata['region_id']}", flush=True)
                try:
                    pieces = splitter.split_documents([region])
                except ValueError as exc:
                    raise ValueError(f"{doc.source}, {region.metadata['region_id']}: {exc}") from exc
                if method == "semantic":
                    with (variant / "boundary_diagnostics.jsonl").open("a", encoding="utf-8") as diagnostics:
                        for report in splitter.last_diagnostics:
                            diagnostics.write(json.dumps(report, ensure_ascii=False, allow_nan=False) + "\n")
                # Validate the union, counting repeated evidence only once.
                previous_start, covered_end = None, region.metadata["char_start"]
                totals["unique_source_characters"] += len(region.page_content)
                totals["unique_region_tokens"] += tokens.count(region.page_content, special_tokens=False)
                for piece_index, piece in enumerate(pieces):
                    meta = piece.metadata
                    start, end = meta["char_start"], meta["char_end"]
                    if previous_start is None:
                        require(start == covered_end, "Missing region beginning")
                    else:
                        require(previous_start < start <= covered_end < end, "Gap or non-advancing window")
                    repeated = covered_end - start
                    if settings["overlap"] == 0:
                        require(repeated == 0, "Unexpected overlap in zero-overlap run")
                    if method == "sliding_window":
                        require(repeated == meta["method_metadata"]["overlap_with_previous_characters"],
                                "Measured overlap differs")
                        require(0 <= meta["method_metadata"]["overlap_with_previous_region_tokens"] <= settings["overlap"],
                                "Overlap exceeds configured source-token budget")
                    totals["repeated_characters"] += repeated
                    previous_start, covered_end = start, end
                    record = make_chunk(doc, start=meta["char_start"], end=meta["char_end"], method=method,
                                        corpus_hash=corpus.corpus_hash, config_hash=effective["config_hash"], run_id=run_id,
                                        chunk_index=counts[doc.document_id], token_count=meta["token_count"],
                                        tokenizer_id=meta["tokenizer_id"], section_path=tuple(meta.get("section_path", ())),
                                        method_metadata=meta["method_metadata"]).to_dict()
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
                    if method == "sliding_window" and piece_index:
                        overlap_counts[meta["method_metadata"]["overlap_with_previous_region_tokens"]] += 1
                        totals["reduced_overlap_windows"] += meta["method_metadata"]["overlap_reduced"]
                        if doc.member not in overlap_examples:
                            overlap_examples[doc.member] = {
                                "source": doc.source, "previous": pieces[piece_index-1].model_dump(),
                                "current": piece.model_dump(),
                                "shared_text": piece.page_content[:repeated]}
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
                require(covered_end == region.metadata["char_end"], "Missing region tail")
        require(bool(sizes), "No chunks produced")
        stats = {"method": method, "scope": "member_sample" if sample else "full_corpus",
                 "documents": len(selected), "regions": len(regions), "chunks": len(sizes),
                 "chunk_size_content_tokens": settings["chunk_size"], "overlap": settings["overlap"],
                 "content_token_count": {"min": min(sizes), "max": max(sizes), "mean": statistics.mean(sizes),
                                         "median": statistics.median(sizes), "histogram": dict(sorted(Counter(sizes).items()))},
                 "token_count_convention": "encoder input tokens including two special tokens",
                 "max_encoder_input_tokens": max(sizes) + tokens.special_tokens,
                 "retained_region_character_coverage": 1.0,
                 "character_redundancy_ratio": totals["characters"] / totals["unique_source_characters"],
                 "indexed_to_unique_content_token_ratio": totals["content_tokens"] / totals["unique_region_tokens"],
                 "unintended_overlap_characters": 0,
                 "empty_chunks": 0, "overflow_chunks": 0, "totals": dict(totals),
                 "overlap_region_token_histogram": dict(sorted(overlap_counts.items())),
                 "per_document": dict(per_source), "elapsed_seconds": time.perf_counter() - started,
                 "canonical_records_sha256": digest_json(hashes),
                 "canonical_hash_note": "ordered record hashes excluding run_id; timings are not chunk content",
                 "corpus_hash": corpus.corpus_hash, "config_hash": effective["config_hash"]}
        write_json(variant / "chunk_stats.json", stats)
        write_json(variant / "inspection_examples.json", examples)
        lines = [f"# {method} chunk inspection", "", f"Scope: {stats['scope']}; {len(sizes):,} chunks.", "",
                 "Each excerpt below is the complete saved chunk, not a reflowed preview.", ""]
        for label, record in examples.items():
            lines += [f"## {label}", "", f"Source: `{record['source']}`; physical pages: {list(record['page_numbers'])}.",
                      f"Span: [{record['char_start']}, {record['char_end']}); input tokens: {record['token_count']}.", "",
                      "~~~~text", record["text"], "~~~~", ""]
        (variant / "inspection_report.md").write_text("\n".join(lines), encoding="utf-8")
        if method == "sliding_window":
            write_json(variant / "overlap_examples.json", overlap_examples)
            write_overlap_report(variant, overlap_examples)
        if method == "structure_aware":
            from ..chunking.structure_aware.reporting import write_structure_report
            write_structure_report(variant)
        if method == "semantic":
            from ..chunking.semantic.reporting import write_semantic_report
            write_semantic_report(variant, splitter, sources)
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
