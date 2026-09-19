"""Build a traceable, structure-preserving text corpus from educational PDFs.

Run with the project's Python 3.11 environment. No API calls or source PDF edits.
See README.md in this directory for the cleaning policy and output schema.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import unicodedata

import pymupdf
import pymupdf4llm

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_VERSION = "1.0.0"
PAGE_SEPARATOR = "\n\n"
LIGATURES = str.maketrans({"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"})
PAGE_NUMBER = re.compile(r"^(?:page\s+)?(?:\d{1,4}|[ivxlcdm]{1,8})(?:\s+(?:of|/)\s+\d{1,4})?$", re.I)
COURSE_CODE = re.compile(r"\b[A-Z]{2,5}\s*[- ]?\s*\d{3,5}\b")


@dataclass(frozen=True)
class CleaningConfig:
    margin_fraction: float = 0.09
    repeated_page_fraction: float = 0.20
    min_repeated_pages: int = 3
    min_token_coverage: float = 0.99
    low_text_characters: int = 40


def clean_text(text: str) -> str:
    """Conservative Unicode/whitespace cleaning; preserve language and structure.

    NFC, rather than NFKC, preserves superscripts, subscripts and math alphabets.
    Hard hyphens, repeated body text, stopwords, case and Indic joiners survive.
    Horizontal spacing is retained for layout tables and code.
    """
    text = unicodedata.normalize("NFC", text).translate(LIGATURES)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub("\u00ad\n(?=[a-z])", "", text)
    text = text.replace("\u00ad", "").replace("\ufeff", "")
    text = text.replace("\u00a0", " ").replace("\u202f", " ")
    text = "".join(c for c in text if unicodedata.category(c) != "Cc" or c in "\n\t")
    # Only known Word/Wingdings list markers at line starts are normalized.
    text = re.sub(r"(?m)^(\s*)[\uf0b7\uf0a7](?=\s|\S)", r"\1•", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip("\n")


def token_counts(text: str) -> Counter:
    text = clean_text(text).casefold().replace("<br>", " ")
    return Counter(re.findall(r"\w+", text, flags=re.UNICODE))


def preservation_metrics(reference: str, candidate: str) -> dict:
    """Token multiset recall catches omissions; it does not prove reading order."""
    expected, actual = token_counts(reference), token_counts(candidate)
    missing = expected - actual
    total = sum(expected.values())
    numeric_missing = {k: v for k, v in missing.items() if any(c.isdigit() for c in k)}
    protected_words = {"not", "no", "never", "unless", "except", "only", "must", "shall", "before", "after", "minimum", "maximum", "and", "or"}
    expected_symbols = Counter(c for c in clean_text(reference) if unicodedata.category(c) == "Sm")
    actual_symbols = Counter(c for c in clean_text(candidate) if unicodedata.category(c) == "Sm")
    return {
        "token_coverage": round(1 - sum(missing.values()) / total, 6) if total else 1.0,
        "missing_numeric_tokens": numeric_missing,
        "missing_rule_words": {k: v for k, v in missing.items() if k in protected_words},
        "missing_math_symbols": dict(expected_symbols - actual_symbols),
        "missing_token_sample": dict(missing.most_common(15)),
    }


def acceptable(metrics: dict, config: CleaningConfig) -> bool:
    return (metrics["token_coverage"] >= config.min_token_coverage
            and not metrics["missing_numeric_tokens"] and not metrics["missing_math_symbols"]
            and not metrics["missing_rule_words"])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_pdfs(input_dir: Path, members: list[str] | None = None) -> list[Path]:
    folders = [input_dir / member for member in members] if members else [
        p for p in sorted(input_dir.iterdir()) if p.is_dir() and p.name != "outputs" and not p.name.startswith(".")
    ]
    files = []
    for folder in folders:
        if not folder.is_dir():
            raise ValueError(f"Member directory does not exist: {folder}")
        if not folder.resolve().is_relative_to(input_dir.resolve()):
            raise ValueError("Member paths must be inside the input directory")
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            with path.open("rb") as stream:
                header = stream.read(1024)
            if path.suffix.lower() == ".pdf" or b"%PDF-" in header:
                files.append(path)
    return sorted(set(files))


def page_snapshot(page: pymupdf.Page) -> dict:
    """Retain the original text and physical lines for audit and structure work."""
    page.remove_rotation()  # In memory only; coordinate system now matches extraction.
    data = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT)
    lines = []
    for block_index, block in enumerate(data["blocks"]):
        for line in block.get("lines", []):
            spans = line["spans"]
            text = "".join(span["text"] for span in spans)
            if not text.strip():
                continue
            lines.append({
                "text": text,
                "bbox": [round(n, 3) for n in line["bbox"]],
                "block": block_index,
                "font_size": round(max(s["size"] for s in spans), 2),
                "bold": any(s["flags"] & 16 for s in spans),
            })
    images = page.get_image_info()
    image_area = max((abs(pymupdf.Rect(i["bbox"]) & page.rect) for i in images), default=0)
    return {
        "page_number": page.number + 1,
        "page_label": page.get_label(),
        "width": page.rect.width,
        "height": page.rect.height,
        "raw_text": page.get_text("text", sort=False),
        "source_lines": lines,
        "image_count": len(images),
        "largest_image_area_fraction": round(image_area / abs(page.rect), 4),
    }


def margin_key(line: dict, height: float, config: CleaningConfig) -> tuple | None:
    x0, y0, x1, y1 = line["bbox"]
    if y1 <= height * config.margin_fraction:
        side = "header"
    elif y0 >= height * (1 - config.margin_fraction):
        side = "footer"
    else:
        return None
    text = re.sub(r"\s+", " ", clean_text(line["text"])).strip()
    if not text or len(text) > 160:
        return None
    # Exact wording and approximate vertical position, not global repetition.
    return side, round(((y0 + y1) / 2 / height) / 0.025), text


def find_margin_removals(snapshots: list[dict], config: CleaningConfig) -> dict[int, list[dict]]:
    occurrences = defaultdict(set)
    for page in snapshots:
        for line in page["source_lines"]:
            key = margin_key(line, page["height"], config)
            if key:
                occurrences[key].add(page["page_number"])
    threshold = max(config.min_repeated_pages, math.ceil(len(snapshots) * config.repeated_page_fraction))
    result = defaultdict(list)
    for page in snapshots:
        for line in page["source_lines"]:
            key = margin_key(line, page["height"], config)
            if not key:
                continue
            text = key[2]
            # Number-only text must be in the outermost 6% of the page.
            edge = line["bbox"][3] <= page["height"] * 0.06 or line["bbox"][1] >= page["height"] * 0.94
            reason = None
            if edge and PAGE_NUMBER.fullmatch(text):
                reason = "isolated_margin_page_number"
            elif len(text) >= 4 and len(occurrences[key]) >= threshold and page["page_number"] != min(occurrences[key]):
                reason = "repeated_positional_" + key[0]
            if reason:
                result[page["page_number"]].append({**line, "reason": reason})
    return dict(result)


def layout_fallback(doc: pymupdf.Document, page_number: int) -> str:
    """Poppler fixed-width text preserves cell alignment when Markdown loses text."""
    with pymupdf.open() as single:
        single.insert_pdf(doc, from_page=page_number, to_page=page_number)
        result = subprocess.run(
            ["pdftotext", "-layout", "-enc", "UTF-8", "-", "-"],
            input=single.tobytes(), capture_output=True, check=True, timeout=90,
        )
    return clean_text(result.stdout.decode("utf-8", errors="strict").replace("\f", "\n"))


def fenced_layout(text: str) -> str:
    longest = max((len(m.group()) for m in re.finditer(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{text}\n{fence}"


def quality_flags(snapshot: dict, clean: str, metrics: dict, config: CleaningConfig) -> list[str]:
    flags = []
    raw = clean_text(snapshot["raw_text"])
    if not raw.strip():
        flags.append("no_extractable_text")
        if snapshot["image_count"]:
            flags.append("ocr_candidate")
    elif len(raw.strip()) < config.low_text_characters:
        flags.append("low_text_page")
    if snapshot["largest_image_area_fraction"] >= 0.20:
        flags.append("substantial_image_content")
    if "\ufffd" in raw or "\ufffd" in clean:
        flags.append("replacement_characters")
    if any(unicodedata.category(c) == "Co" for c in raw):
        flags.append("unmapped_private_use_characters")
    words = re.findall(r"\b[^\W\d_]+\b", raw)
    if len(words) >= 50 and sum(len(w) == 1 and w.casefold() not in {"a", "i"} for w in words) / len(words) > 0.18:
        flags.append("fragmented_text_or_formula")
    if not acceptable(metrics, config):
        flags.append("text_preservation_check_failed")
    if len(re.findall(r"[\u0b80-\u0bff\u0d80-\u0dff]", raw)) >= 20:
        flags.append("contains_tamil_or_sinhala")
    return flags


def clean_pdf(pdf_path: Path, output_dir: Path, source_root: Path, config: CleaningConfig | None = None) -> dict:
    """Clean one complete PDF, write text + provenance JSON, and return its manifest row."""
    config = config or CleaningConfig()
    source = pdf_path.relative_to(source_root).as_posix()
    doc_id = sha256_file(pdf_path)
    relative_stem = Path(source).with_suffix("")
    # Hash prevents collisions between differently named/encoded versions of a PDF.
    destination = output_dir / relative_stem.parent / f"{relative_stem.name}__{doc_id[:12]}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    text_path = Path(str(destination) + ".txt")
    json_path = Path(str(destination) + ".json")
    records = []
    with pymupdf.open(pdf_path) as doc:
        if doc.needs_pass:
            raise ValueError("Password-protected PDF cannot be extracted without a password")
        snapshots = [page_snapshot(page) for page in doc]
        removals = find_margin_removals(snapshots, config)
        font_counts = Counter()
        for snapshot in snapshots:
            for line in snapshot["source_lines"]:
                font_counts[round(line["font_size"])] += len(line["text"])
        body_size = font_counts.most_common(1)[0][0] if font_counts else 12
        sizes = sorted((s for s in font_counts if s > body_size), reverse=True)[:6]

        def header_id(span, page=None):
            size = round(span["size"])
            return "#" * (sizes.index(size) + 1) + " " if size in sizes else ""

        document_metadata = dict(doc.metadata)
        toc = doc.get_toc()
        for index, snapshot in enumerate(snapshots):
            page = doc[index]
            removed = removals.get(index + 1, [])
            margin_warnings = []
            # Do not apply redactions supplied by the source PDF's author.
            if any(a.type[0] == pymupdf.PDF_ANNOT_REDACT for a in page.annots()):
                removed = []
                margin_warnings.append("existing_redaction_annotation_review")
            backup = pymupdf.open()
            if removed:
                backup.insert_pdf(doc, from_page=index, to_page=index)
            for item in removed:
                page.add_redact_annot(pymupdf.Rect(item["bbox"]), fill=False)
            if removed:
                # Modify text only in the in-memory document. Never save over a PDF.
                page.apply_redactions(images=0, graphics=0, text=0)
            reference = clean_text(page.get_text("text", sort=False))
            if removed:
                expected = token_counts(snapshot["raw_text"]) - token_counts("\n".join(r["text"] for r in removed))
                if expected - token_counts(reference):
                    # A font bounding box can overlap adjacent text. Restore the page
                    # rather than silently deleting anything beyond the nominated lines.
                    doc.delete_page(index)
                    doc.insert_pdf(backup, start_at=index)
                    page = doc[index]
                    reference = clean_text(page.get_text("text", sort=False))
                    removed = []
                    margin_warnings.append("margin_removal_skipped_to_preserve_text")
            backup.close()
            attempts = []
            tables = []
            method = "pymupdf4llm_markdown"
            candidate = ""
            if reference.strip():
                try:
                    converted = pymupdf4llm.to_markdown(
                        doc, pages=[index], hdr_info=header_id, page_chunks=True,
                        margins=0, write_images=False, embed_images=False,
                        ignore_images=True, force_text=True, show_progress=False,
                        table_strategy="lines_strict", fontsize_limit=0,
                    )[0]
                    candidate = clean_text(converted["text"])
                    tables = converted["tables"]
                except Exception as exc:
                    attempts.append({"method": method, "error": f"{type(exc).__name__}: {exc}"})
            metrics = preservation_metrics(reference, candidate)
            course_table = len(COURSE_CODE.findall(reference)) >= 5 and not tables
            if reference.strip() and (not acceptable(metrics, config) or course_table):
                attempts.append({"method": method, **metrics})
                try:
                    layout = layout_fallback(doc, index)
                    layout_metrics = preservation_metrics(reference, layout)
                    attempts.append({"method": "poppler_layout", **layout_metrics})
                    if acceptable(layout_metrics, config):
                        candidate, metrics, method = fenced_layout(layout), layout_metrics, "poppler_layout"
                    else:
                        # The original text survives even when both layout engines disagree.
                        candidate, method = fenced_layout(reference), "pymupdf_plain_fallback"
                        metrics = preservation_metrics(reference, reference)
                except Exception as exc:
                    attempts.append({"method": "poppler_layout", "error": f"{type(exc).__name__}: {exc}"})
                    candidate, method = fenced_layout(reference), "pymupdf_plain_fallback"
                    metrics = preservation_metrics(reference, reference)
            flags = quality_flags(snapshot, candidate, metrics, config)
            flags.extend(margin_warnings)
            if method != "pymupdf4llm_markdown":
                flags.append("layout_fallback_review")
            if method == "pymupdf_plain_fallback":
                flags.append("reading_order_review_required")
            if tables:
                flags.append("table_structure_review")
            headings = [
                {"text": line["text"], "bbox": line["bbox"], "font_size": line["font_size"], "inferred": True}
                for line in snapshot["source_lines"] if line not in [
                    {k: v for k, v in r.items() if k != "reason"} for r in removed
                ] and len(line["text"].split()) <= 18 and (
                    line["font_size"] > body_size + 0.5 or line["bold"]
                )
            ]
            records.append({
                **snapshot, "clean_text": candidate, "extraction_method": method,
                "removed_margin_lines": removed, "heading_candidates": headings,
                "detected_tables": tables, "preservation": metrics,
                "extraction_attempts": attempts, "quality_flags": sorted(set(flags)),
            })
        full_text, spans = assemble_document(records)
        for record, span in zip(records, spans):
            record.update(span)
        payload = {
            "schema_version": 1, "pipeline_version": PIPELINE_VERSION,
            "document_id": doc_id, "source": source,
            "source_sha256": doc_id, "pdf_metadata": document_metadata,
            "pdf_toc": toc, "body_font_size_estimate": body_size,
            "cleaning_config": asdict(config), "pages": records,
        }
    write_text_atomic(text_path, full_text)
    write_json_atomic(json_path, payload)
    return {
        "document_id": doc_id, "source": source, "source_sha256": doc_id,
        "member": Path(source).parts[0], "status": "processed",
        "text_path": text_path.relative_to(output_dir).as_posix(),
        "metadata_path": json_path.relative_to(output_dir).as_posix(),
        "page_count": len(records), "clean_characters": len(full_text),
        "has_extractable_text": any(r["clean_text"].strip() for r in records),
        "clean_sha256": hashlib.sha256(full_text.encode("utf-8")).hexdigest(),
        "raw_characters": sum(len(r["raw_text"]) for r in records),
        "removed_margin_lines": sum(len(r["removed_margin_lines"]) for r in records),
        "flagged_pages": sum(bool(r["quality_flags"]) for r in records),
        "quality_flags": dict(Counter(f for r in records for f in r["quality_flags"])),
        "extraction_methods": dict(Counter(r["extraction_method"] for r in records)),
    }


def assemble_document(pages: list[dict]) -> tuple[str, list[dict]]:
    """Join pages without synthetic page headings. Offsets use Python characters."""
    pieces, spans, offset = [], [], 0
    for index, page in enumerate(pages):
        if index:
            pieces.append(PAGE_SEPARATOR)
            offset += len(PAGE_SEPARATOR)
        text = page["clean_text"]
        spans.append({"char_start": offset, "char_end": offset + len(text)})
        pieces.append(text)
        offset += len(text)
    return "".join(pieces), spans


def write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(path: Path, value) -> None:
    write_text_atomic(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def build_indexes(output_dir: Path, rows: list[dict], config: CleaningConfig, input_dir: Path) -> dict:
    source_hashes, text_hashes = {}, {}
    corpus = []
    review_rows = []
    for row in rows:
        row["include_in_default_corpus"] = False
        if row["status"] != "processed":
            review_rows.append({"source": row["source"], "page_number": "", "flags": "document_extraction_failed", "method": "", "preview": row["error"]})
            continue
        duplicate = source_hashes.get(row["source_sha256"]) or text_hashes.get(row["clean_sha256"])
        row["duplicate_of"] = duplicate
        if not row["has_extractable_text"]:
            row["exclusion_reason"] = "no_extractable_text"
        elif duplicate:
            row["exclusion_reason"] = "exact_source_or_clean_text_duplicate"
        else:
            row["include_in_default_corpus"] = True
            source_hashes[row["source_sha256"]] = row["source"]
            text_hashes[row["clean_sha256"]] = row["source"]
        detail = json.loads((output_dir / row["metadata_path"]).read_text(encoding="utf-8"))
        page_spans = []
        for page in detail["pages"]:
            page_spans.append({k: page[k] for k in ["page_number", "page_label", "char_start", "char_end", "quality_flags"]})
            if page["quality_flags"]:
                review_rows.append({
                    "source": row["source"], "page_number": page["page_number"],
                    "flags": ";".join(page["quality_flags"]), "method": page["extraction_method"],
                    "preview": page["clean_text"][:160].replace("\n", " "),
                })
        if row["include_in_default_corpus"]:
            corpus.append({
                "document_id": row["document_id"], "source": row["source"],
                "text": (output_dir / row["text_path"]).read_text(encoding="utf-8"),
                "pages": page_spans, "metadata_path": row["metadata_path"],
                "requires_review": bool(row["flagged_pages"]),
            })
    write_text_atomic(output_dir / "corpus.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in corpus))
    import io
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=["source", "page_number", "flags", "method", "preview"])
    writer.writeheader()
    writer.writerows(review_rows)
    write_text_atomic(output_dir / "review_queue.csv", stream.getvalue())
    summary = {
        "pdfs_found": len(rows), "pdfs_processed": sum(r["status"] == "processed" for r in rows),
        "pdfs_failed": sum(r["status"] == "failed" for r in rows),
        "pages_processed": sum(r.get("page_count", 0) for r in rows),
        "documents_in_corpus": len(corpus),
        "clean_characters": sum(r.get("clean_characters", 0) for r in rows),
        "removed_margin_lines": sum(r.get("removed_margin_lines", 0) for r in rows),
        "flagged_pages": sum(r.get("flagged_pages", 0) for r in rows),
        "quality_flags": dict(sum((Counter(r.get("quality_flags", {})) for r in rows), Counter())),
        "extraction_methods": dict(sum((Counter(r.get("extraction_methods", {})) for r in rows), Counter())),
    }
    poppler = subprocess.run(["pdftotext", "-v"], capture_output=True, text=True, check=True).stderr.splitlines()[0]
    manifest = {
        "schema_version": 1, "pipeline_version": PIPELINE_VERSION,
        "pipeline_sha256": sha256_file(Path(__file__)),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_directory": str(input_dir), "config": asdict(config),
        "environment": {"python": sys.version.split()[0], "pymupdf": importlib.metadata.version("pymupdf"),
                        "pymupdf4llm": importlib.metadata.version("pymupdf4llm"), "poppler": poppler},
        "summary": summary, "documents": rows,
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    write_text_atomic(output_dir / "README.md", f"""# Cleaned educational PDF corpus

Processed {summary['pdfs_processed']} / {summary['pdfs_found']} PDF files and {summary['pages_processed']} pages.
The default corpus contains {len(corpus)} nonempty documents after exact deduplication.

- `corpus.jsonl`: one complete document per line; use the same `text` for all chunking methods.
- Member folders: readable `.txt` (Markdown conventions) and detailed provenance `.json` per source.
- `manifest.json`: source hashes, output hashes, configuration, versions, counts and failures.
- `review_queue.csv`: {summary['flagged_pages']} pages flagged for inspection. Flags are review signals, not proof of corruption.

Headings and tables are inferred from PDF layout. Layout fallbacks are fenced text to retain spacing.
Page numbers and source names live in metadata rather than artificial headings in the text.
Page spans are zero-based Python character offsets with exclusive ends; PDF page numbers are one-based.
Original PDFs are unchanged. Their extracted raw text and all removed margin lines are retained in JSON.

This is an automatically cleaned baseline, not a manually verified gold dataset.
Review flagged evidence pages before building the question/answer benchmark. Image-only content has not been OCRed.
The full cleaning policy, commands and limitations are in `02_data_cleaning/README.md` in the project.
Use the manifest/corpus instead of globbing old generated files from earlier runs.
""")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "final_data")
    parser.add_argument("--members", nargs="+", help="Optional member folder names, e.g. aysha haleema vithusan")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    if not shutil.which("pdftotext"):
        parser.error("Poppler pdftotext is required for layout fallback. See 02_data_cleaning/README.md.")
    input_dir, output_dir = args.input_dir.resolve(), args.output_dir.resolve()
    if not input_dir.is_dir():
        parser.error(f"Input directory does not exist: {input_dir}")
    if output_dir == input_dir or output_dir.is_relative_to(input_dir) or input_dir.is_relative_to(output_dir):
        parser.error("Input and output directories must be separate, non-nested directories")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    files = discover_pdfs(input_dir, args.members)
    if not files:
        parser.error("No PDFs found in member folders")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = CleaningConfig()
    rows = []
    print(f"Cleaning {len(files)} PDFs with {args.workers} worker(s) into {output_dir}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(clean_pdf, p, output_dir, input_dir, config): p for p in files}
        for future in as_completed(jobs):
            path = jobs[future]
            try:
                row = future.result()
                print(f"[{len(rows) + 1}/{len(files)}] {row['source']}: {row['page_count']} pages, {row['flagged_pages']} flagged", flush=True)
            except Exception as exc:
                row = {"source": path.relative_to(input_dir).as_posix(), "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                print(f"FAILED {row['source']}: {row['error']}", file=sys.stderr, flush=True)
            rows.append(row)
    summary = build_indexes(output_dir, sorted(rows, key=lambda r: r["source"]), config, input_dir)
    print(json.dumps(summary, indent=2), flush=True)
    return 1 if summary["pdfs_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
