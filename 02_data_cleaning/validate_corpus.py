"""Validate corpus integrity against the original PDFs and saved page provenance."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import pymupdf

from clean_pdfs import ROOT, discover_pdfs, sha256_file, write_json_atomic


def validate(output_dir: Path, input_dir: Path, allow_subset: bool = False) -> dict:
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    errors = []
    if manifest["pipeline_sha256"] != sha256_file(Path(__file__).with_name("clean_pdfs.py")):
        errors.append("Cleaner source differs from the version recorded in the manifest")
    preparation = manifest.get("text_preparation")
    if preparation and preparation["script_sha256"] != sha256_file(Path(__file__).with_name("prepare_texts.py")):
        errors.append("Text preparation source differs from the version recorded in the manifest")
    checked_pages = 0
    corpus = [json.loads(line) for line in (output_dir / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    corpus_by_source = {row["source"]: row for row in corpus}
    if len(corpus_by_source) != len(corpus):
        errors.append("Duplicate source entries in corpus.jsonl")
    expected_sources = set()
    for row in manifest["documents"]:
        if row["status"] != "processed":
            errors.append(f"Failed document: {row['source']}")
            continue
        source = input_dir / row["source"]
        if sha256_file(source) != row["source_sha256"]:
            errors.append(f"Source changed: {row['source']}")
        text = (output_dir / row["text_path"]).read_text(encoding="utf-8")
        detail = json.loads((output_dir / row["metadata_path"]).read_text(encoding="utf-8"))
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != row["clean_sha256"]:
            errors.append(f"Clean text hash differs: {row['source']}")
        with pymupdf.open(source) as pdf:
            if len(detail["pages"]) != len(pdf) or len(pdf) != row["page_count"]:
                errors.append(f"Page count differs: {row['source']}")
        if detail["source_sha256"] != row["source_sha256"]:
            errors.append(f"Metadata source hash differs: {row['source']}")
        if detail["cleaning_config"] != manifest["config"]:
            errors.append(f"Configuration differs: {row['source']}")
        if text != "\n\n".join(p["clean_text"] for p in detail["pages"]):
            errors.append(f"Assembled document differs: {row['source']}")
        for index, page in enumerate(detail["pages"]):
            checked_pages += 1
            if page["page_number"] != index + 1:
                errors.append(f"Page sequence differs: {row['source']}:{index + 1}")
            if text[page["char_start"]:page["char_end"]] != page["clean_text"]:
                errors.append(f"Page span differs: {row['source']}:{index + 1}")
            if not 0 <= page["char_start"] <= page["char_end"] <= len(text):
                errors.append(f"Page span outside text: {row['source']}:{index + 1}")
            if preparation:
                if not page["include_in_chunking"] and (page["clean_text"] or not page.get("excluded_text")):
                    errors.append(f"Invalid exclusion: {row['source']}:{index + 1}")
                for heading in page["headings"]:
                    if page["clean_text"][heading["char_start"]:heading["char_end"]] != heading["text"]:
                        errors.append(f"Heading span differs: {row['source']}:{index + 1}")
        if row["include_in_default_corpus"]:
            expected_sources.add(row["source"])
            entry = corpus_by_source.get(row["source"])
            if entry is None or entry["text"] != text or entry["document_id"] != row["document_id"]:
                errors.append(f"Corpus content differs: {row['source']}")
            elif len(entry["pages"]) != len(detail["pages"]) or any(text[p["char_start"]:p["char_end"]] != d["clean_text"] for p, d in zip(entry["pages"], detail["pages"])):
                errors.append(f"Corpus page span differs: {row['source']}")
            elif preparation and any(
                p.get("include_in_chunking") != d["include_in_chunking"] or p.get("headings") != d["headings"]
                for p, d in zip(entry["pages"], detail["pages"])
            ):
                errors.append(f"Corpus preparation metadata differs: {row['source']}")
    if set(corpus_by_source) != expected_sources:
        errors.append("Corpus membership differs from manifest")
    if checked_pages != manifest["summary"]["pages_processed"]:
        errors.append("Manifest total page count differs")
    all_sources = {p.relative_to(input_dir).as_posix() for p in discover_pdfs(input_dir)}
    manifest_sources = {r["source"] for r in manifest["documents"]}
    if all_sources - manifest_sources and not allow_subset:
        errors.append("Some input PDFs are missing from the manifest; use --allow-subset only for an intentional subset run")
    result = {
        "passed": not errors,
        "documents_checked": sum(r["status"] == "processed" for r in manifest["documents"]),
        "pages_checked": checked_pages,
        "corpus_documents_checked": len(corpus),
        "members": dict(Counter(r["member"] for r in manifest["documents"] if r["status"] == "processed")),
        "input_pdfs_outside_this_manifest": sorted(all_sources - manifest_sources),
        "errors": errors,
    }
    write_json_atomic(output_dir / "validation_report.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "final_data")
    parser.add_argument("--allow-subset", action="store_true")
    args = parser.parse_args()
    result = validate(args.output_dir.resolve(), args.input_dir.resolve(), args.allow_subset)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
