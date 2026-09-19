"""Validate and load prepared text without exposing raw or excluded extraction."""

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .config import Config, within
from .provenance import Document, Heading, Page, eligible_regions


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_json(value) -> str:
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                  ensure_ascii=False, allow_nan=False).encode("utf-8"))


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_pages(entry: dict, detail: dict) -> tuple[Page, ...]:
    text = entry["text"]
    require(isinstance(text, str), "Document text must be a string")
    require(len(entry["pages"]) == len(detail["pages"]), "Page counts differ")
    pages, offset = [], 0
    for number, (span, raw) in enumerate(zip(entry["pages"], detail["pages"]), 1):
        start, end = raw["char_start"], raw["char_end"]
        require(type(start) is int and type(end) is int and 0 <= start <= end <= len(text), "Invalid page span")
        require(start == offset, f"Page {number}: noncontiguous page offsets")
        require(type(raw["page_number"]) is int and raw["page_number"] == number, "Invalid page sequence")
        require(text[start:end] == raw["clean_text"], f"Page {number}: text differs")
        for key in ("page_number", "page_label", "char_start", "char_end", "quality_flags", "include_in_chunking", "headings"):
            require(span[key] == raw[key], f"Page {number}: JSONL metadata differs: {key}")
        include = raw["include_in_chunking"]
        require(type(include) is bool, "include_in_chunking must be boolean")
        reason = raw.get("exclusion_reason")
        if not include:
            require(raw["clean_text"] == "" and isinstance(reason, str) and bool(reason), "Excluded page contains text or lacks a reason")
            require(span.get("exclusion_reason") == reason, "Exclusion reason differs")
        flags = raw["quality_flags"]
        require(isinstance(flags, list) and all(isinstance(f, str) for f in flags), "Invalid quality flags")
        headings = []
        for item in raw["headings"]:
            hs, he = item["char_start"], item["char_end"]
            require(type(hs) is int and type(he) is int and 0 <= hs < he <= end-start, "Invalid heading span")
            require(text[start+hs:start+he] == item["text"], "Heading text differs from its span")
            level = item["level"]
            require(level is None or type(level) is int and 1 <= level <= 6, "Invalid heading level")
            require(type(item["inferred"]) is bool and isinstance(item["origin"], str), "Invalid heading provenance")
            headings.append(Heading(item["text"], start+hs, start+he, level, item["origin"], item["inferred"]))
        pages.append(Page(number, raw["page_label"], start, end, include, tuple(flags),
                          tuple(headings), not bool(raw["clean_text"].strip()), reason))
        offset = end
        if number < len(detail["pages"]):
            require(text[end:end+2] == "\n\n", "Invalid page separator")
            offset += 2
    require(offset == len(text), "Unaccounted document text")
    return tuple(pages)


@dataclass(frozen=True)
class Corpus:
    documents: tuple[Document, ...]
    corpus_hash: str
    snapshot: dict


def load_corpus(corpus_dir: Path, config: Config) -> Corpus:
    """Read each companion JSON once, retain only prepared text and safe metadata.

    This checks internal integrity. The inspect CLI additionally runs the existing
    PDF/source-hash validator; loading alone does not certify original PDFs.
    """
    hashes = {}

    def read(name: str, as_json: bool = False):
        content = within(corpus_dir, name).read_bytes()
        hashes[name] = digest_bytes(content)
        value = content.decode("utf-8")
        return json.loads(value) if as_json else value

    manifest = read("manifest.json", True)
    require(manifest["schema_version"] == 1, "Unsupported corpus schema")
    require(manifest.get("text_preparation", {}).get("text_format") == "plain_text", "Run text preparation before loading")
    entries = [json.loads(line) for line in read("corpus.jsonl").splitlines() if line.strip()]
    by_source = {e["source"]: e for e in entries}
    require(len(by_source) == len(entries), "Duplicate JSONL sources")
    rows = manifest["documents"]
    require(len({r["source"] for r in rows}) == len(rows), "Duplicate manifest sources")
    require(len(rows) == config.expected_documents, "Unexpected document count")
    require(dict(Counter(r["member"] for r in rows)) == config.expected_members, "Unexpected member counts")
    expected = {r["source"] for r in rows if r["include_in_default_corpus"]}
    require(set(by_source) == expected, "JSONL membership differs from manifest")
    documents, total_pages, total_chars = [], 0, 0
    for row in rows:
        require(row["status"] == "processed", f"Incomplete extraction: {row['source']}")
        require(type(row["include_in_default_corpus"]) is bool, "Invalid corpus membership flag")
        require(isinstance(row["source"], str) and bool(row["source"])
                and not Path(row["source"]).is_absolute() and ".." not in Path(row["source"]).parts,
                "Invalid source path")
        require(Path(row["source"]).parts[0] == row["member"], "Member does not match source")
        text = read(row["text_path"])
        detail = read(row["metadata_path"], True)
        for key in ("document_id", "source", "source_sha256"):
            require(detail[key] == row[key], f"Document identity differs: {key}")
        require(row["document_id"] == row["source_sha256"], "Document ID differs from source hash")
        require(isinstance(row["document_id"], str) and len(row["document_id"]) == 64
                and all(c in "0123456789abcdef" for c in row["document_id"]), "Invalid document hash")
        require(detail["cleaning_config"] == manifest["config"], "Cleaning configuration differs")
        require(hashes[row["text_path"]] == row["clean_sha256"], f"Text hash differs: {row['source']}")
        require(len(text) == row["clean_characters"], "Text length differs")
        require(len(detail["pages"]) == row["page_count"], "Page count differs")
        total_pages += len(detail["pages"])
        total_chars += len(text)
        # Non-default documents are validated too, but never exposed for retrieval.
        entry = by_source.get(row["source"], {
            "text": text, "pages": detail["pages"], "document_id": row["document_id"],
            "metadata_path": row["metadata_path"],
        })
        require(entry["text"] == text and entry["document_id"] == row["document_id"], "JSONL text/identity differs")
        require(entry["metadata_path"] == row["metadata_path"], "JSONL metadata path differs")
        pages = parse_pages(entry, detail)
        if row["include_in_default_corpus"]:
            require(bool(text.strip()), "Empty document in default corpus")
            documents.append(Document(row["document_id"], row["source"], row["member"], text,
                                      row["clean_sha256"], row["metadata_path"], hashes[row["metadata_path"]],
                                      pages, eligible_regions(pages)))
    require(len({d.document_id for d in documents}) == len(documents), "Duplicate default document IDs")
    require(total_pages == config.expected_pages == manifest["summary"]["pages_processed"], "Unexpected total page count")
    require(total_chars == manifest["summary"]["clean_characters"], "Manifest character total differs")
    require(len(documents) == manifest["summary"]["documents_in_corpus"], "Default corpus count differs")
    identity = {"schema_version": 1, "files": hashes, "gap_policy": config.gap_policy}
    snapshot = {**identity, "corpus_hash": digest_json(identity),
                "documents": len(documents), "manifest_documents": len(rows), "pages": total_pages,
                "characters": total_chars, "members": dict(Counter(d.member for d in documents)),
                "excluded_pages": sum(not p.include_in_chunking for d in documents for p in d.pages),
                "blank_included_pages": sum(p.include_in_chunking and p.is_blank for d in documents for p in d.pages),
                "eligible_regions": sum(len(d.regions) for d in documents),
                "headings": sum(len(p.headings) for d in documents for p in d.pages)}
    return Corpus(tuple(documents), snapshot["corpus_hash"], snapshot)


def load_structure_metadata(corpus_dir: Path, document: Document) -> dict:
    """Explicit lazy access to layout candidates, without raw/excluded text."""
    path = within(corpus_dir, document.metadata_path)
    content = path.read_bytes()
    require(digest_bytes(content) == document.metadata_sha256, "Detailed metadata changed since corpus load")
    detail = json.loads(content)
    return {"pdf_toc": detail["pdf_toc"], "pages": [
        {"page_number": raw["page_number"], "heading_candidates": raw["heading_candidates"],
         "detected_tables": raw["detected_tables"]}
        for raw, page in zip(detail["pages"], document.pages) if not page.is_barrier
    ]}
