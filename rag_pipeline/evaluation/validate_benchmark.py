"""Validate local annotations and refuse to freeze unfinished benchmarks."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from ..common.config import ROOT, load_config, within
from ..common.load_corpus import digest_json, file_hash, load_corpus, require
from ..common.provenance import pages_for_span

SPLITS = {"pilot", "development", "test", "diagnostic"}
ANSWERABILITY = {"answerable", "unanswerable", "ambiguous", "extraction_diagnostic"}
CATEGORIES = {"definition", "explanation", "procedure", "eligibility", "exception", "table_lookup", "multi_passage", "abstention", "ambiguity"}


def annotation_hash(row):
    """A review becomes stale if its question, rubric, evidence, or split changes."""
    return digest_json({k: v for k, v in row.items() if k != "review"})


def validate_records(records, corpus, manifest):
    docs = {d.document_id: d for d in corpus.documents}
    errors, blockers, ids, groups, documents = [], [], set(), {}, {}
    if manifest.get("schema_version") != 1:
        errors.append("Unsupported benchmark manifest schema")
    if manifest.get("corpus_hash") != corpus.corpus_hash:
        errors.append("Benchmark corpus hash differs from loaded corpus")
    if not records:
        blockers.append("No annotations")
    for row in records:
        qid = row.get("question_id", "<missing>")
        try:
            require(isinstance(qid, str) and qid and qid not in ids, "Missing or duplicate question_id")
            ids.add(qid)
            require(row["schema_version"] == 1, "Unsupported question schema")
            require(row["corpus_hash"] == corpus.corpus_hash, "Question corpus hash mismatch")
            for key in ("question", "reference_answer", "scope", "group_id", "author"):
                require(isinstance(row[key], str) and bool(row[key].strip()), f"Missing {key}")
            require(row["language"] == "en", "Current protocol evaluates English only")
            require(row["category"] in CATEGORIES, "Unknown category")
            require(row["answerability"] in ANSWERABILITY, "Unknown answerability")
            split = row["split"]
            require(split in SPLITS, "Unknown split")
            if row["answerability"] in {"ambiguous", "extraction_diagnostic"}:
                require(split in {"pilot", "diagnostic"}, "Ambiguous/extraction cases cannot enter scored splits")
            bucket = "development" if split == "pilot" else split
            reserved = manifest.get("reserved_development_groups", [])
            require(not (bucket == "test" and row["group_id"] in reserved), "Exposed pilot group cannot enter test")
            if bucket != "diagnostic":
                require(groups.setdefault(row["group_id"], bucket) == bucket, "Question group leaks across development/test")
            require(isinstance(row["rubric"], list) and row["rubric"] and
                    all(isinstance(x, str) and x.strip() for x in row["rubric"]), "Missing answer rubric")
            units = row["evidence_units"]
            require(isinstance(units, list), "Evidence units must be a list")
            if row["answerability"] == "answerable":
                require(bool(units), "Answerable question needs evidence")
            else:
                require(isinstance(row.get("rationale"), str) and bool(row["rationale"].strip()), "Missing non-answerability rationale")
            unit_ids = set()
            for unit in units:
                require(unit["unit_id"] not in unit_ids, "Duplicate evidence unit ID")
                unit_ids.add(unit["unit_id"])
                require(isinstance(unit["claim"], str) and bool(unit["claim"].strip()), "Missing atomic claim")
                require(isinstance(unit["alternatives"], list) and unit["alternatives"], "Evidence unit has no alternative")
                for span in unit["alternatives"]:
                    doc = docs[span["document_id"]]
                    require(span["source"] == doc.source and span["source_sha256"] == doc.document_id,
                            "Wrong source identity")
                    require(span["clean_sha256"] == doc.clean_sha256, "Prepared text hash mismatch")
                    start, end = span["char_start"], span["char_end"]
                    pages = pages_for_span(doc, start, end)
                    require(span["text"] == doc.text[start:end], "Evidence text differs from exact source span")
                    require(span["page_numbers"] == [p.page_number for p in pages], "Wrong physical page numbers")
                    source_group = manifest["document_groups"][doc.document_id]
                    require(not (bucket == "test" and source_group in reserved), "Exposed source family cannot enter test")
                    if bucket != "diagnostic":
                        require(groups.setdefault("source:" + source_group, bucket) == bucket,
                                "Source/version family leaks across development/test")
                        require(documents.setdefault(doc.document_id, bucket) == bucket,
                                "Document leaks across development/test")
            review = row["review"]
            require(review["status"] in {"pending", "changes_requested", "approved"}, "Invalid review status")
            if review["status"] != "approved":
                blockers.append(f"{qid}: human review pending")
            else:
                require(review.get("reviewer_kind") == "human" and isinstance(review.get("reviewer"), str)
                        and bool(review["reviewer"].strip()) and review["reviewer"] != row["author"],
                        "Approval requires an independent named human reviewer")
                require(review.get("pdf_verified") is True, "Reviewer must verify PDF evidence/scope")
                require(review.get("reviewed_content_hash") == annotation_hash(row), "Review is stale")
                require(isinstance(review.get("notes"), str) and bool(review["notes"].strip()), "Review needs notes")
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            errors.append(f"{qid}: {exc}")
    counts = Counter(r.get("split", "invalid") for r in records)
    for split in ("development", "test"):
        if counts[split] != manifest.get("target_counts", {}).get(split):
            blockers.append(f"{split}: {counts[split]} records; target {manifest.get('target_counts', {}).get(split)}")
    if manifest.get("protocol_status") != "approved":
        blockers.append("Experiment protocol has not been approved by the research lead")
    return {"valid": not errors, "freeze_ready": not errors and not blockers, "errors": errors,
            "blockers": blockers, "counts": dict(counts), "questions": len(records),
            "reviewed": sum(r.get("review", {}).get("status") == "approved" for r in records)}


def validate_directory(directory, corpus):
    manifest = json.loads((directory / "benchmark_manifest.json").read_text())
    require(isinstance(manifest["files"], dict) and bool(manifest["files"]), "No benchmark files listed")
    rows, hashes = [], {}
    for split, name in manifest["files"].items():
        require(split in SPLITS, "Unknown manifest split")
        path = within(directory, name)
        hashes[name] = file_hash(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                require(row["split"] == split, f"File/split mismatch in {name}")
                rows.append(row)
    report = validate_records(rows, corpus, manifest)
    # Confirm the original PDFs inspected by reviewers still match their identities.
    source_ids = {span["document_id"] for row in rows for unit in row.get("evidence_units", [])
                  for span in unit.get("alternatives", [])}
    source_hashes = {}
    for doc in corpus.documents:
        if doc.document_id in source_ids:
            actual = file_hash(within(ROOT / "data", doc.source))
            source_hashes[doc.source] = actual
            if actual != doc.document_id:
                report["errors"].append(f"Original PDF changed: {doc.source}")
    report["source_pdf_hashes"] = source_hashes
    report["protocol_hashes"] = {name: file_hash(ROOT / "rag_pipeline/docs" / name)
                                 for name in ("experiment_protocol.md", "annotation_guide.md", "model_feasibility.md")}
    report["valid"] = not report["errors"]
    report["freeze_ready"] = report["valid"] and not report["blockers"]
    report["artifact_hashes"] = hashes
    report["manifest_sha256"] = file_hash(directory / "benchmark_manifest.json")
    report["corpus_hash"] = corpus.corpus_hash
    frozen_path = directory / "frozen_snapshot.json"
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text())
        for key in ("artifact_hashes", "manifest_sha256", "corpus_hash", "source_pdf_hashes", "protocol_hashes"):
            if frozen.get(key) != report[key]:
                report["errors"].append(f"Frozen benchmark changed: {key}")
        report["valid"] = not report["errors"]
        report["freeze_ready"] = report["valid"] and not report["blockers"]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "data/benchmark")
    parser.add_argument("--require-ready", action="store_true", help="Exit nonzero until all freeze gates pass")
    parser.add_argument("--freeze", action="store_true", help="Write a new immutable snapshot only after all gates pass")
    args = parser.parse_args(argv)
    try:
        directory = args.directory.resolve()
        require(directory.is_relative_to(ROOT / "data"), "Benchmark must remain under ignored data/")
        corpus = load_corpus(ROOT / "final_data", load_config(ROOT / "rag_pipeline/config.json"))
        report = validate_directory(directory, corpus)
        if args.freeze:
            require(report["freeze_ready"], "Freeze refused: " + "; ".join(report["errors"] + report["blockers"]))
            # Exclusive create prevents silently replacing a previously frozen benchmark.
            with (directory / "frozen_snapshot.json").open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2)
                stream.write("\n")
        print(json.dumps(report, indent=2))
        return 0 if report["valid"] and (not args.require_ready or report["freeze_ready"]) else 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Benchmark validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
