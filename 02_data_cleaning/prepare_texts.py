"""Prepare the extracted corpus for chunking, keeping page provenance synchronized.

Run after clean_pdfs.py. This is a formatting pass, not OCR or a PDF repair tool.
The original output is backed up before the staged, validated result is installed.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import tempfile

from markdown_it import MarkdownIt

from clean_pdfs import (
    ROOT, CleaningConfig, assemble_document, build_indexes, sha256_file,
    write_json_atomic, write_text_atomic,
)
from validate_corpus import validate

VERSION = "1.0.0"
INLINE = MarkdownIt("commonmark")
# Only interpret emphasis. Do not rewrite backslashes, equations, URLs or code.
INLINE.inline.ruler.enableOnly(["text", "emphasis", "newline"])
CORRUPT_SOURCE = "aysha/UOC_FMF_HB_2018_2022.pdf"
CORRUPT_HASH_PREFIX = "611c6caa4559"
# These pages share the visibly corrupt embedded font encoding documented in
# the audit. Readable pages 18, 19, 25 and 26 are deliberately retained.
CORRUPT_PAGES = set(range(9, 57)) - {18, 19, 25, 26}


def strip_emphasis(text: str) -> str:
    """Remove paired Markdown emphasis while protecting inline code literals."""
    pieces = re.split(r"(`+[^`]*`+)", text)
    for index in range(0, len(pieces), 2):
        # Extractor sometimes emits malformed numeric emphasis: programme**4 **.
        pieces[index] = re.sub(r"\*\*(\d+) +\*\*", r"**\1**", pieces[index])
        tokens = INLINE.parseInline(pieces[index])[0].children or []
        output = ""
        boundary = False
        for token in tokens:
            if token.type in {"em_open", "em_close", "strong_open", "strong_close"}:
                boundary = True
                continue
            if boundary and output and token.content and output[-1].isalnum() and token.content[0].isalnum():
                output += " "
            output += token.content
            boundary = False
        pieces[index] = output
    return "".join(pieces)


def prepare_page(text: str, method: str) -> tuple[str, dict]:
    """Strip extractor formatting; preserve layout, code, footnotes and math."""
    stats = Counter()
    if method != "pymupdf4llm_markdown":
        if text.startswith("```text\n") and text.endswith("\n```"):
            text = text[len("```text\n"):-len("\n```")]
            stats["outer_fences_removed"] += 1
        # Fallback content is source text, not Markdown: stars can be footnotes.
        return "\n".join(line.rstrip() for line in text.splitlines()).strip("\n"), dict(stats)

    result = []
    code_fence = None
    for line in text.splitlines():
        match = re.match(r"^\s*(`{3,}|~{3,})(.*)$", line)
        if match:
            marker = match[1]
            if code_fence is None:
                code_fence = marker
                stats["code_fence_lines_removed"] += 1
                continue
            if marker[0] == code_fence[0] and len(marker) >= len(code_fence) and not match[2].strip():
                code_fence = None
                stats["code_fence_lines_removed"] += 1
                continue
        if code_fence:
            result.append(line.rstrip())
            continue
        if re.fullmatch(r"\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*", line):
            stats["table_alignment_rows_removed"] += 1
            continue
        # Convert generated hard breaks within cells without breaking table rows.
        line, count = re.subn(r"<br\s*/?>", " / " if line.lstrip().startswith("|") else " ", line, flags=re.I)
        stats["html_breaks_replaced"] += count
        line, count = re.subn(r"^(\s{0,3})#{1,6}\s+", r"\1", line)
        stats["heading_prefixes_removed"] += count
        plain = strip_emphasis(line)
        stats["emphasis_lines_cleaned"] += int(plain != line)
        result.append(plain.rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(result)).strip("\n"), dict(+stats)


def page_number_candidates(pages: list[dict]) -> dict[int, set[str]]:
    """Require a repeated physical-to-printed page offset and margin geometry.

    Only bottom-margin standalone Arabic numbers are candidates. This intentionally
    leaves uncertain numbers alone rather than risking credits or equation terms.
    """
    candidates = []
    number_only_margin_pages = set()
    for p in pages:
        lines = p["source_lines"]
        bottom = max((line["bbox"][3] for line in lines), default=0)
        for line in lines:
            value = line["text"].strip()
            if re.fullmatch(r"\d{1,4}", value) and line["bbox"][1] > p["height"] * .85:
                stripped = p["clean_text"].removeprefix("```text\n").removesuffix("\n```").strip()
                eligible = abs(line["bbox"][3] - bottom) < 2 or stripped == value
                if stripped == value and all(x["text"].strip() == value for x in lines):
                    number_only_margin_pages.add((p["page_number"], value))
                candidates.append((p["page_number"], value, p["page_number"] - int(value), eligible))
    support = Counter(offset for _, _, offset in set((n, v, o) for n, v, o, _ in candidates))
    result = {}
    for number, value, offset, eligible in candidates:
        if (support[offset] >= 4 and eligible) or (number, value) in number_only_margin_pages:
            result.setdefault(number, set()).add(value)
    return result


def remove_page_number(text: str, candidates: set[str]) -> tuple[str, list[str]]:
    lines = text.splitlines()
    removed = []
    # At most the final nonempty line; no deletion from tables or body text.
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and lines[-1].strip() in candidates:
        removed.append(lines.pop())
        # Duplicate printed page-number layers can leave a number-only page.
        if lines and all(not line.strip() or line.strip() == removed[0].strip() for line in lines):
            removed.extend(line for line in lines if line.strip())
            lines = []
    return "\n".join(lines).rstrip("\n"), removed


def heading_spans(page: dict, original: str) -> list[dict]:
    """Locate inferred headings in the prepared page (page-local offsets)."""
    candidates = {}
    for line in original.splitlines():
        m = re.match(r"^\s{0,3}(#{1,6})\s+(.+)$", line)
        if m and page["extraction_method"] == "pymupdf4llm_markdown":
            candidates[" ".join(strip_emphasis(m[2]).split())] = (len(m[1]), "extractor_markdown")
    for item in page["heading_candidates"]:
        candidates.setdefault(" ".join(item["text"].split()), (None, "pdf_font_candidate"))
    spans = []
    offset = 0
    for line in page["clean_text"].splitlines(keepends=True):
        normalized = " ".join(line.split())
        if normalized and normalized in candidates:
            level, origin = candidates[normalized]
            start = offset + len(line) - len(line.lstrip())
            spans.append({"text": line.strip(), "char_start": start,
                          "char_end": offset + len(line.rstrip()), "level": level,
                          "origin": origin, "inferred": True})
        offset += len(line)
    return spans


def run(output: Path, input_dir: Path, backup_root: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if "text_preparation" in manifest:
        if manifest["text_preparation"]["script_sha256"] != sha256_file(Path(__file__)):
            raise ValueError("Preparation script changed; regenerate from the extraction backup first.")
        result = validate(output, input_dir)
        if not result["passed"]:
            raise ValueError(result["errors"])
        return {"already_prepared": True, **result}
    rows = manifest["documents"]
    # Read every byte of every TXT and its companion JSON before making edits.
    # Refuse to discard manual content changes. The known wrapper-only edit is safe.
    reconciled = []
    for row in rows:
        if row["status"] != "processed":
            raise ValueError(f"Incomplete extraction: {row['source']}")
        detail = json.loads((output / row["metadata_path"]).read_text(encoding="utf-8"))
        original, _ = assemble_document(detail["pages"])
        current = (output / row["text_path"]).read_text(encoding="utf-8")
        if current != original:
            if current != original.replace("```text", "").replace("```", ""):
                raise ValueError(f"Unreconciled manual text changes: {row['text_path']}")
            reconciled.append(row["text_path"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = backup_root / f"final_data_before_preparation_{stamp}"
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(output, backup)
    totals = Counter()
    reports = []
    with tempfile.TemporaryDirectory(prefix="prepare_corpus_", dir=backup_root) as temporary:
        stage = Path(temporary) / "final_data"
        shutil.copytree(output, stage)
        for row in rows:
            detail = json.loads((stage / row["metadata_path"]).read_text(encoding="utf-8"))
            numbers = page_number_candidates(detail["pages"])
            counts = Counter()
            excluded = []
            for page in detail["pages"]:
                original = page["clean_text"]
                text, changes = prepare_page(original, page["extraction_method"])
                text, removed = remove_page_number(text, numbers.get(page["page_number"], set()))
                changes["page_numbers_removed"] = len(removed)
                page["text_preparation"] = {"changes": changes, "removed_page_number_lines": removed}
                page["preservation_scope"] = "extraction_before_text_preparation"
                page["include_in_chunking"] = True
                # Explicit corpus-specific quarantine, never a generic language heuristic.
                if (row["source"] == CORRUPT_SOURCE
                        and row["source_sha256"].startswith(CORRUPT_HASH_PREFIX)
                        and page["page_number"] in CORRUPT_PAGES):
                    page["excluded_text"] = original
                    page["include_in_chunking"] = False
                    page["exclusion_reason"] = "corrupt_pdf_font_encoding_requires_reextraction"
                    page["quality_flags"] = sorted(set(page["quality_flags"] + ["corrupt_font_encoding_excluded"]))
                    excluded.append(page["page_number"])
                    text = ""
                page["clean_text"] = text
                page["headings"] = heading_spans(page, original)
                counts.update(changes)
                counts["pages_changed"] += int(original != text)
            full, spans = assemble_document(detail["pages"])
            for page, span in zip(detail["pages"], spans):
                page.update(span)
            detail["text_preparation"] = {"version": VERSION, "text_format": "plain_text", "excluded_pages": excluded}
            before = row["clean_characters"]
            row.update(clean_characters=len(full), clean_sha256=hashlib.sha256(full.encode("utf-8")).hexdigest(),
                       has_extractable_text=bool(full.strip()),
                       flagged_pages=sum(bool(p["quality_flags"]) for p in detail["pages"]),
                       quality_flags=dict(Counter(f for p in detail["pages"] for f in p["quality_flags"])))
            write_text_atomic(stage / row["text_path"], full)
            write_json_atomic(stage / row["metadata_path"], detail)
            reports.append({"source": row["source"], "characters_before": before,
                            "characters_after": len(full), "excluded_pages": excluded, "changes": dict(+counts)})
            totals.update(counts)
        # Reuse index construction, but retain the actual original extraction provenance.
        summary = build_indexes(stage, rows, CleaningConfig(**manifest["config"]), input_dir)
        manifest["documents"] = rows
        manifest["summary"] = summary
        preparation = {
            "version": VERSION, "script_sha256": sha256_file(Path(__file__)),
            "markdown_it_version": importlib.metadata.version("markdown-it-py"),
            "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
            "backup_directory": str(backup), "text_format": "plain_text",
            "reconciled_wrapper_only_edits": reconciled,
        }
        manifest["text_preparation"] = preparation
        write_json_atomic(stage / "manifest.json", manifest)
        # Carry usable heading offsets and exclusions into the lightweight corpus too.
        corpus = []
        for line in (stage / "corpus.jsonl").read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            detail = json.loads((stage / entry["metadata_path"]).read_text(encoding="utf-8"))
            for span, page in zip(entry["pages"], detail["pages"]):
                span.update(include_in_chunking=page["include_in_chunking"], headings=page["headings"])
                if not page["include_in_chunking"]:
                    span["exclusion_reason"] = page["exclusion_reason"]
            corpus.append(entry)
        write_text_atomic(stage / "corpus.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in corpus))
        report = {"documents_processed": len(rows), "pages_processed": summary["pages_processed"],
                  "changes": dict(+totals), "excluded_pages": sum(len(r["excluded_pages"]) for r in reports),
                  "preparation": preparation, "documents": reports}
        write_json_atomic(stage / "preparation_report.json", report)
        write_text_atomic(stage / "README.md", f"""# Prepared educational PDF corpus

{len(rows)} documents from three members; {summary['pages_processed']} physical PDF pages retained in metadata.

- Member `.txt` files: prepared plain text, ready for chunking experiments.
- `corpus.jsonl`: the identical text plus source, page offsets and heading candidates.
  Feed only `text` into embedding models. Keep metadata for citations and analysis.
- Detailed `.json`: original extracted text, layout evidence, quality flags and preparation history.
- `preparation_report.json`: changes for every document and the backup location.
- `review_queue.csv`: remaining extraction review signals; these are not an error percentage.

Artificial outer code fences, heading prefixes, emphasis and HTML breaks have been cleaned.
Table pipes and layout spacing remain because they separate cells. Meaningful source
asterisks, code, quotes, formulas, course codes and repeated numeric values are preserved.
Heading candidates are in `pages[].headings`: offsets are page-local, and levels may be null.
For structure-aware chunking, use these candidates and the detailed JSON/PDF outline;
a Markdown-header splitter cannot recover headings from plain text alone.

{report['excluded_pages']} pages with corrupt font encoding in UOC_FMF_HB_2018_2022.pdf are
excluded from chunking text. Their original text is retained as `excluded_text` and
`raw_text` in detailed JSON; physical page numbering is unchanged. Exclusions are listed
in the preparation report. Do not create benchmark questions requiring excluded content.
Blank/image-only pages likewise contribute no missing image text. This pass does not
perform OCR or repair table reading order, formulas or information contained in images.

Use this same frozen text corpus for all chunking methods. Page offsets are zero-based
Python character positions with exclusive ends; PDF page numbers are one-based.
All original PDFs are unchanged. This is a prepared experimental baseline, not a
manually verified gold corpus. Check source PDF evidence when writing evaluation questions.
""")
        result = validate(stage, input_dir)
        if not result["passed"]:
            raise ValueError(f"Staged corpus failed validation: {result['errors']}")
        # All transformations and validation succeeded; publish generated files only.
        for file in stage.rglob("*"):
            if file.is_file():
                target = output / file.relative_to(stage)
                target.parent.mkdir(parents=True, exist_ok=True)
                write_text_atomic(target, file.read_text(encoding="utf-8"))
    return {k: v for k, v in report.items() if k != "documents"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "final_data")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--backup-root", type=Path, default=ROOT / "data/outputs")
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir.resolve(), args.input_dir.resolve(), args.backup_root.resolve()), indent=2))
