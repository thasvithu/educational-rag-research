"""Shared chunk records; construction validates exact text and source boundaries."""

from dataclasses import asdict, dataclass
import re

from .config import validate_run_id
from .load_corpus import digest_json
from .provenance import Document, flags_for_span, pages_for_span, region_for_span

METHODS = {"fixed_size", "sliding_window", "structure_aware", "semantic", "late"}


@dataclass(frozen=True)
class Chunk:
    schema_version: int
    chunk_id: str
    document_id: str
    source: str
    member: str
    corpus_hash: str
    config_hash: str
    method: str
    run_id: str
    chunk_index: int
    region_id: str
    text: str
    char_start: int
    char_end: int
    page_numbers: tuple[int, ...]
    token_count: int | None
    tokenizer_id: str | None
    section_path: tuple[str, ...]
    quality_flags: tuple[str, ...]
    method_metadata: dict

    def to_dict(self) -> dict:
        return asdict(self)


def make_chunk(document: Document, *, start: int, end: int, method: str,
               corpus_hash: str, config_hash: str, run_id: str, chunk_index: int,
               token_count: int | None = None, tokenizer_id: str | None = None,
               section_path: tuple[str, ...] = (), method_metadata: dict | None = None) -> Chunk:
    """No tokenization is guessed here. Phase 2 selects the tokenizer.

    Null token fields are allowed for Phase 1 span inspection. Production chunkers
    must pass the measured count and pinned tokenizer identity together.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    for value in (corpus_hash, config_hash):
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("Corpus/config identity must be a SHA-256 digest")
    validate_run_id(run_id)
    if type(chunk_index) is not int or chunk_index < 0:
        raise ValueError("chunk_index must be a nonnegative integer")
    if (token_count is None) != (tokenizer_id is None):
        raise ValueError("Token count and tokenizer identity must be supplied together")
    if token_count is not None and (type(token_count) is not int or token_count <= 0
                                   or not isinstance(tokenizer_id, str) or not tokenizer_id.strip()):
        raise ValueError("Invalid token measurement")
    if not isinstance(section_path, tuple) or any(not isinstance(s, str) for s in section_path):
        raise ValueError("section_path must be a tuple of strings")
    if method_metadata is not None and not isinstance(method_metadata, dict):
        raise ValueError("method_metadata must be a JSON object")
    metadata = dict(method_metadata or {})
    digest_json(metadata)  # Refuse NaN or nonserializable values.
    region = region_for_span(document, start, end)
    identity = {"schema_version": 1, "corpus_hash": corpus_hash, "document_id": document.document_id,
                "config_hash": config_hash, "method": method, "char_start": start, "char_end": end}
    return Chunk(1, digest_json(identity), document.document_id, document.source, document.member,
                 corpus_hash, config_hash, method, run_id, chunk_index, region.region_id,
                 document.text[start:end], start, end,
                 tuple(p.page_number for p in pages_for_span(document, start, end)),
                 token_count, tokenizer_id, section_path, flags_for_span(document, start, end), metadata)
