"""LangChain splitter: validated sections, packed paragraphs, explicit fallbacks."""

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter

from ...common.load_corpus import require
from ...common.token_windows import split_token_windows
from .headings import normalized
from .units import Unit, paragraph_units, sentence_units, table_row_units


def load_settings(path: Path):
    settings = json.loads(Path(path).read_text())
    expected = {"schema_version": 1, "method": "structure_aware", "algorithm_version": "1", "overlap": 0,
                "heading_policy": "validated_lines_numbering_outline_font_shape_v1", "unknown_level": 2,
                "fallback_policy": "paragraph_list_table_lines_then_sentences_then_tokens_v1",
                "gap_policy": "reset_heading_path_at_each_region"}
    require(set(settings) == set(expected) | {"chunk_size"}, "Unknown/missing structure-aware setting")
    require(all(type(settings[k]) is type(v) and settings[k] == v for k, v in expected.items()),
            "Unsupported structure-aware policy")
    require(type(settings["chunk_size"]) is int and settings["chunk_size"] > 0, "chunk_size must be positive")
    return settings


class StructureAwareSplitter(TextSplitter):
    """Keep source text intact; section names are metadata, never a text prefix.

    `resolved_headings` supplied by the runner use document-level character
    offsets. Plain split_text() has no PDF hints and uses paragraph fallbacks.
    """

    def __init__(self, tokenizer, chunk_size=256):
        require(type(chunk_size) is int and chunk_size > 0, "chunk_size must be positive")
        require(chunk_size + tokenizer.special_tokens <= tokenizer.max_input_tokens, "Chunk cap exceeds encoder limit")
        self.tokens = tokenizer
        super().__init__(chunk_size=chunk_size, chunk_overlap=0, strip_whitespace=False,
                         length_function=lambda text: tokenizer.count(text, special_tokens=False))

    def fits(self, text):
        return (0 < self.tokens.count(text, special_tokens=False) <= self._chunk_size
                and self.tokens.count(text) <= self.tokens.max_input_tokens)

    def attach_tokenless_units(self, text, units):
        """Keep decorative glyph-only units with a neighbor instead of losing them.

        Some PDF bullets use private-use characters ignored by the tokenizer.
        They remain in exact source text, but cannot form an embedding input alone.
        """
        result = []
        for unit in units:
            if self.tokens.count(text[unit.start:unit.end], special_tokens=False) == 0:
                if result:
                    previous = result[-1]
                    result[-1] = Unit(previous.start, unit.end, previous.kind)
                continue
            result.append(Unit(result[-1].end if result else 0, unit.end, unit.kind))
        require(bool(result), "Nonblank input has no encodable content tokens; review this region")
        return result

    def sections(self, text, base, headings):
        sections, stack, start, repeated = [], [], 0, 0
        last_heading_end = None
        for heading in sorted(headings, key=lambda h: h["char_start"]):
            hs, he = heading["char_start"] - base, heading["char_end"] - base
            if not 0 <= hs < he <= len(text):
                continue  # A heading in another region never supplies a path here.
            require(normalized(text[hs:he]) == normalized(heading["text"]), "Resolved heading no longer matches source")
            if stack and heading["level"] == stack[-1]["level"] and normalized(heading["text"]) == normalized(stack[-1]["text"]):
                repeated += 1  # Same heading on a continuation page stays in the section.
                continue
            # Adjacent title lines (or a parent heading immediately followed by
            # a child) belong together; avoid a tiny heading-only chunk.
            heading_only = last_heading_end is not None and not text[last_heading_end:hs].strip()
            if self.tokens.count(text[start:hs], special_tokens=False) and not heading_only:
                sections.append((start, hs, list(stack)))
                start = hs
            while stack and stack[-1]["level"] >= heading["level"]:
                stack.pop()
            stack.append(heading)
            last_heading_end = he
        if text[start:].strip():
            sections.append((start, len(text), list(stack)))
        return sections, repeated

    def pieces(self, text, section_start):
        """Reduce oversized units, keeping every character and split provenance."""
        pieces = []
        units = []
        for original in self.attach_tokenless_units(text, paragraph_units(text)):
            if original.kind == "table_block" and not self.fits(text[original.start:original.end]):
                raw = text[original.start:original.end]
                for row in self.attach_tokenless_units(raw, table_row_units(raw)):
                    units.append((Unit(original.start + row.start, original.start + row.end, row.kind), "table_rows"))
            else:
                units.append((original, "none"))
        for unit, initial_fallback in units:
            raw = text[unit.start:unit.end]
            identity = [section_start + unit.start, section_start + unit.end]
            if self.fits(raw):
                pieces.append((unit.start, unit.end, unit.kind, initial_fallback, identity, 0))
                continue
            sentences = self.attach_tokenless_units(raw, sentence_units(raw))
            for sentence in sentences:
                a, b = unit.start + sentence.start, unit.start + sentence.end
                fragment = text[a:b]
                if self.fits(fragment):
                    pieces.append((a, b, unit.kind, "sentence", identity, 0))
                else:
                    for span in split_token_windows(fragment, self.tokens, self._chunk_size):
                        pieces.append((a + span.char_start, a + span.char_end, unit.kind, "token", identity, span.boundary_backoffs))
        require("".join(text[a:b] for a, b, *_ in pieces) == text, "Structural unit parsing dropped source text")
        return pieces

    def pack_section(self, text, section_start):
        pending = []
        for piece in self.pieces(text, section_start):
            if pending and not self.fits(text[pending[0][0]:piece[1]]):
                yield pending
                pending = []
            pending.append(piece)
        if pending:
            yield pending

    def split_text(self, text):
        return [d.page_content for d in self.create_documents([text])]

    def create_documents(self, texts, metadatas=None):
        if metadatas is None:
            metadatas = [{} for _ in texts]
        require(len(texts) == len(metadatas), "Each text needs matching metadata")
        chunks = []
        for text, original in zip(texts, metadatas):
            base = original.get("char_start", 0)
            require(type(base) is int and base >= 0, "Invalid document offset")
            require(original.get("char_end", base + len(text)) == base + len(text), "Region length differs from metadata")
            if not text.strip():
                continue
            sections, repeated = self.sections(text, base, original.get("resolved_headings", []))
            for section_start, section_end, path in sections:
                section = text[section_start:section_end]
                for pieces in self.pack_section(section, base + section_start):
                    start, end = section_start + pieces[0][0], section_start + pieces[-1][1]
                    chunk_text = text[start:end]
                    require(self.fits(chunk_text), "Structural chunk exceeds token limit")
                    meta = deepcopy({k: v for k, v in original.items() if k not in {"resolved_headings", "pages"}})
                    # Split events are keyed by original unit spans so later summaries
                    # count one long list item once, even if it yields many chunks.
                    events = sorted({(a, b, kind, fallback) for _, _, kind, fallback, (a, b), _ in pieces if fallback != "none"})
                    meta.update(char_start=base + start, char_end=base + end, start_index=start,
                                region_char_start=base, token_count=self.tokens.count(chunk_text),
                                tokenizer_id=self.tokens.tokenizer_id, section_path=[h["text"] for h in path],
                                method_metadata={
                                    "content_token_count": self.tokens.count(chunk_text, special_tokens=False),
                                    "special_token_count": self.tokens.special_tokens,
                                    "boundary_backoffs": sum(p[-1] for p in pieces),
                                    "section_start": base + section_start, "section_end": base + section_end,
                                    "heading_references": path, "missing_heading": not bool(path),
                                    "repeated_heading_continuations_in_region": repeated,
                                    "unit_kind_counts": dict(Counter(p[2] for p in pieces)),
                                    "fallback_types": sorted({p[3] for p in pieces if p[3] != "none"}),
                                    "split_events": [{"char_start": a, "char_end": b, "kind": kind, "fallback": fallback}
                                                     for a, b, kind, fallback in events]})
                    if "pages" in original:
                        meta["pages"] = deepcopy([p for p in original["pages"] if p["char_start"] < base + end and base + start < p["char_end"]])
                        meta["page_numbers"] = [p["page_number"] for p in meta["pages"]]
                        meta["quality_flags"] = sorted({f for p in meta["pages"] for f in p["quality_flags"]})
                    chunks.append(Document(page_content=chunk_text, metadata=meta))
        return chunks
