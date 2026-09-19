"""Phase 1 inspection CLI. No chunking, embeddings, or API calls."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

from .common.config import ROOT, load_config, within
from .common.load_corpus import file_hash, load_corpus
from .common.provenance import pages_for_span
from .common.run_manifest import create_run, environment, finish_run, write_json


def inspection_examples(corpus, characters: int) -> dict:
    samples = []
    for member in sorted({d.member for d in corpus.documents}):
        document = next(d for d in sorted(corpus.documents, key=lambda d: d.source)
                        if d.member == member and d.regions)
        region = document.regions[0]
        end = min(region.char_end, region.char_start + characters)
        samples.append({"member": member, "source": document.source, "document_id": document.document_id,
                        "char_start": region.char_start, "char_end": end,
                        "page_numbers": [p.page_number for p in pages_for_span(document, region.char_start, end)],
                        "text": document.text[region.char_start:end],
                        "headings": [asdict(h) for p in document.pages for h in p.headings
                                     if region.char_start <= h.char_start < end]})
    cross_page = None
    for document in corpus.documents:
        for left, right in zip(document.pages, document.pages[1:]):
            if not left.is_barrier and not right.is_barrier:
                start, end = max(left.char_start, left.char_end-160), min(right.char_end, right.char_start+160)
                cross_page = {"source": document.source, "char_start": start, "char_end": end,
                              "page_numbers": [p.page_number for p in pages_for_span(document, start, end)],
                              "text": document.text[start:end]}
                break
        if cross_page:
            break
    exclusions = []
    for document in corpus.documents:
        numbers = [p.page_number for p in document.pages if not p.include_in_chunking]
        if numbers:
            exclusions.append({"source": document.source, "excluded_page_numbers": numbers,
                               "eligible_regions": [asdict(r) for r in document.regions],
                               "policy": "Chunks must fit entirely within one eligible region; raw/excluded text is never loaded."})
    return {"selection": "Deterministic source order; illustrative examples, not a random quality audit",
            "member_samples": samples, "cross_page_example": cross_page, "excluded_gap_examples": exclusions}


def inspect(config_path: Path, run_id: str, root: Path = ROOT) -> Path:
    config = load_config(config_path, root)
    run, manifest = create_run(root, config, run_id)
    try:
        write_json(run / "run_config.json", config.to_dict())
        write_json(run / "environment.json", environment(root))
        corpus_dir, source_dir = within(root, config.corpus_dir), within(root, config.source_dir)
        # Internal validation first prevents unsafe metadata paths reaching the legacy validator.
        corpus = load_corpus(corpus_dir, config)
        input_manifest = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
        for row in input_manifest["documents"]:
            within(source_dir, row["source"])
        write_json(run / "corpus_snapshot.json", corpus.snapshot)
        result = subprocess.run([
            sys.executable, str(root / "02_data_cleaning/validate_corpus.py"),
            "--input-dir", str(source_dir), "--output-dir", str(corpus_dir),
            "--report-path", str(run / "validation_report.json"),
        ], capture_output=True, text=True)
        if result.returncode != 0:
            # Do not mask a validator error behind successful corpus loading.
            raise ValueError(f"PDF/source integrity validation failed. {result.stderr.strip()} {result.stdout.strip()}")
        report = json.loads((run / "validation_report.json").read_text(encoding="utf-8"))
        if not report["passed"]:
            raise ValueError(f"Source validator reported errors: {report['errors']}")
        examples = inspection_examples(corpus, config.sample_characters)
        write_json(run / "inspection_examples.json", examples)
        write_json(run / "document_summary.json", [
            {"document_id": d.document_id, "source": d.source, "member": d.member,
             "characters": len(d.text), "pages": len(d.pages), "regions": [asdict(r) for r in d.regions],
             "excluded_pages": [p.page_number for p in d.pages if not p.include_in_chunking],
             "blank_included_pages": [p.page_number for p in d.pages if p.include_in_chunking and p.is_blank],
             "quality_flags": sorted({f for p in d.pages for f in p.quality_flags})}
            for d in corpus.documents])
        # Detect changes made during the run, without writing anything to the corpus.
        for name, expected in corpus.snapshot["files"].items():
            if file_hash(within(corpus_dir, name)) != expected:
                raise ValueError(f"Input changed during inspection: {name}")
        lines = ["# Phase 1 corpus inspection", "", "PDF/source and prepared-text integrity checks passed.", "",
                 f"- Documents: {corpus.snapshot['documents']}", f"- Physical pages: {corpus.snapshot['pages']}",
                 f"- Excluded pages: {corpus.snapshot['excluded_pages']}",
                 f"- Blank included pages: {corpus.snapshot['blank_included_pages']}",
                 f"- Eligible text regions: {corpus.snapshot['eligible_regions']}", "",
                 "All offsets are document-level Python characters, end exclusive. Text is not normalized by the loader.",
                 "", "## One sample per member", ""]
        for sample in examples["member_samples"]:
            lines += [f"### {sample['member']}: {sample['source']}", "",
                      f"Pages: {sample['page_numbers']}; span: [{sample['char_start']}, {sample['char_end']})", "",
                      "~~~~text", sample["text"], "~~~~", ""]
        lines += ["## Ordinary page break: allowed within a chunk", ""]
        cross = examples["cross_page_example"]
        if cross:
            lines += [f"Source: {cross['source']}; pages: {cross['page_numbers']}; span: [{cross['char_start']}, {cross['char_end']})", "",
                      "~~~~text", cross["text"], "~~~~", ""]
        else:
            lines += ["No adjacent nonblank pages in this corpus.", ""]
        lines += ["## Excluded-page gaps: chunks cannot cross", ""]
        for example in examples["excluded_gap_examples"]:
            lines += [f"Source: {example['source']}", "", f"Excluded physical pages: {example['excluded_page_numbers']}", "",
                      "| Region | Physical pages | Document span |", "|---|---|---|"]
            for region in example["eligible_regions"]:
                numbers = region["page_numbers"]
                lines.append(f"| {region['region_id']} | {numbers[0]}–{numbers[-1]} | [{region['char_start']}, {region['char_end']}) |")
            lines += ["", example["policy"], ""]
        lines += ["Full machine-readable examples are in inspection_examples.json.", "",
                  "## Scope", "", "These are integrity checks, not a semantic accuracy score. Tables, formulas and image content still require evidence review.",
                  "No chunks, token counts, embeddings, retrieval results or answers were generated.", ""]
        (run / "inspection_report.md").write_text("\n".join(lines), encoding="utf-8")
        manifest["corpus_hash"] = corpus.corpus_hash
        finish_run(run, manifest)
        print(json.dumps({"status": "complete", "output": str(run),
                          **{k: corpus.snapshot[k] for k in ("documents", "pages", "members", "excluded_pages", "blank_included_pages", "eligible_regions")}}, indent=2))
    except Exception as exc:
        finish_run(run, manifest, error=f"{type(exc).__name__}: {exc}")
        raise
    return run


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("inspect", help="Validate the corpus and write a new Phase 1 inspection run")
    command.add_argument("--config", type=Path, default=ROOT / "rag_pipeline/config.json")
    command.add_argument("--run-id", required=True, help="New output folder name; existing runs are never overwritten")
    args = parser.parse_args(argv)
    try:
        inspect(args.config.resolve(), args.run_id)
    except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as exc:
        print(f"Inspection failed: {exc}", file=sys.stderr)
        return 1
    return 0
