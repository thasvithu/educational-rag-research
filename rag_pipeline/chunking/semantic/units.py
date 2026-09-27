"""Sentence-like units for boundary scoring, preserving every source character."""

from dataclasses import dataclass, replace

from ...common.load_corpus import require
from ...common.token_windows import split_token_windows
from ..structure_aware.units import paragraph_units, sentence_units, table_row_units


@dataclass(frozen=True)
class SemanticUnit:
    start: int
    end: int
    kind: str
    original_start: int
    original_end: int
    forced_token_split: bool = False
    tokenless_attachments: int = 0


def segment_units(text, tokens, cap):
    """Prose uses sentences; list items and visible table rows are suitable units.

    Abbreviations/initials/decimals use the tested sentence helper. Course codes
    are not separators. Long units are split before scoring, never truncated.
    """
    raw = []
    for block in paragraph_units(text):
        part = text[block.start:block.end]
        if block.kind == "table_block":
            pieces = table_row_units(part)
        elif block.kind == "list_item":
            raw.append(SemanticUnit(block.start, block.end, "list_item", block.start, block.end))
            continue
        else:
            pieces = sentence_units(part)
        for piece in pieces:
            a, b = block.start + piece.start, block.start + piece.end
            raw.append(SemanticUnit(a, b, piece.kind, a, b))

    # Private-use PDF bullets may encode to zero tokens. Keep these glyphs with
    # a neighbor, including a leading run, rather than drop them or embed emptiness.
    joined, leading = [], 0
    for unit in raw:
        if tokens.count(text[unit.start:unit.end], special_tokens=False) == 0:
            if joined:
                previous = joined[-1]
                joined[-1] = replace(previous, end=unit.end, original_end=unit.end,
                                     tokenless_attachments=previous.tokenless_attachments + 1)
            else:
                leading += 1
            continue
        start = joined[-1].end if joined else 0
        joined.append(replace(unit, start=start, original_start=start, tokenless_attachments=leading))
        leading = 0
    require(bool(joined), "Nonblank region has no encodable units")

    result = []
    for unit in joined:
        part = text[unit.start:unit.end]
        if tokens.count(part, special_tokens=False) <= cap and tokens.count(part) <= tokens.max_input_tokens:
            result.append(unit)
        else:
            for span in split_token_windows(part, tokens, cap):
                result.append(replace(unit, start=unit.start + span.char_start, end=unit.start + span.char_end,
                                      forced_token_split=True))
    require("".join(text[u.start:u.end] for u in result) == text, "Semantic unit parsing dropped source text")
    return result
