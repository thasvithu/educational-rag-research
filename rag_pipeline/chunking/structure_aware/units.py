"""Paragraph/list/table-line units and conservative sentence fallback offsets."""

from dataclasses import dataclass
import re

LIST = re.compile(r"^\s*(?:[•*–-]|\d+[.)]|[A-Za-z][.)]|\([a-zivx]+\))\s+")
ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "no", "fig", "st", "vs", "etc", "e.g", "i.e"}


@dataclass(frozen=True)
class Unit:
    start: int
    end: int
    kind: str


def table_line(line):
    # Layout coordinates do not identify cleaned character spans. Use visible
    # separators only; never turn uncertain cells into invented reconstructed rows.
    stripped = line.strip()
    return not LIST.match(line) and (stripped.count("|") >= 2 or
                                    len(re.split(r"[ \t]{2,}", stripped)) >= 2)


def paragraph_units(text):
    cuts = [0] + [m.end() for m in re.finditer(r"\n[ \t]*\n+", text)] + [len(text)]
    # Leading blank lines belong to the first actual unit, other blanks to the left.
    spans, start = [], 0
    for end in sorted(set(cuts[1:])):
        if text[start:end].strip():
            spans.append((start, end))
            start = end
    if spans and start < len(text):
        spans[-1] = (spans[-1][0], len(text))
    units = []
    for start, end in spans:
        lines, offset = [], start
        for line in text[start:end].splitlines(keepends=True):
            lines.append((offset, line))
            offset += len(line)
        rows = [pos for pos, line in lines if table_line(line)]
        items = [pos for pos, line in lines if LIST.match(line)]
        if len(rows) >= 2:
            units.append(Unit(start, end, "table_block"))
            continue
        elif items:
            boundaries, kind = items, "list_item"
        else:
            units.append(Unit(start, end, "paragraph"))
            continue
        if not text[start:boundaries[0]].strip():
            boundaries[0] = start
        boundaries = sorted(set([start] + boundaries + [end]))
        for a, b in zip(boundaries, boundaries[1:]):
            if text[a:b].strip():
                units.append(Unit(a, b, kind))
            elif units:
                previous = units[-1]
                units[-1] = Unit(previous.start, b, previous.kind)
    return units


def table_row_units(text):
    starts, position = [], 0
    for line in text.splitlines(keepends=True):
        if table_line(line):
            starts.append(position)
        position += len(line)
    # Retain a leading header and indentation with the first row. Wrapped cells
    # have uncertain alignment; this is a visible-line fallback, not reconstruction.
    cuts = [0] + starts[1:] + [len(text)]
    return [Unit(a, b, "table_row") for a, b in zip(cuts, cuts[1:]) if a < b]


def sentence_units(text):
    cuts = [0]
    for match in re.finditer(r"[.!?][\"”')\]]*\s+(?=[A-Z0-9])", text):
        # Do not break decimal numbers, initials or common short abbreviations.
        before = text[:match.start() + 1]
        word = re.search(r"([A-Za-z.]+)\.$", before)
        if word and (word[1].casefold() in ABBREVIATIONS or len(word[1]) == 1):
            continue
        if text[cuts[-1]:match.end()].strip():
            cuts.append(match.end())
    cuts.append(len(text))
    return [Unit(a, b, "sentence") for a, b in zip(cuts, cuts[1:]) if a < b]
