"""Immutable source references. All offsets are Python characters, end exclusive."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Heading:
    text: str
    char_start: int
    char_end: int
    level: int | None
    origin: str
    inferred: bool


@dataclass(frozen=True)
class Page:
    page_number: int
    page_label: str
    char_start: int
    char_end: int
    include_in_chunking: bool
    quality_flags: tuple[str, ...]
    headings: tuple[Heading, ...]  # Document-level offsets after loading.
    is_blank: bool
    exclusion_reason: str | None = None

    @property
    def is_barrier(self) -> bool:
        return not self.include_in_chunking or self.is_blank


@dataclass(frozen=True)
class Region:
    region_id: str
    char_start: int
    char_end: int
    page_numbers: tuple[int, ...]


@dataclass(frozen=True)
class Document:
    document_id: str
    source: str
    member: str
    text: str
    clean_sha256: str
    metadata_path: str
    metadata_sha256: str
    pages: tuple[Page, ...]
    regions: tuple[Region, ...]


def eligible_regions(pages: tuple[Page, ...]) -> tuple[Region, ...]:
    """Keep ordinary page separators; never glue text across a barrier page."""
    groups, current = [], []
    for page in pages:
        if page.is_barrier:
            if current:
                groups.append(current)
                current = []
        else:
            current.append(page)
    if current:
        groups.append(current)
    return tuple(Region(f"region_{i:04d}", group[0].char_start, group[-1].char_end,
                        tuple(p.page_number for p in group)) for i, group in enumerate(groups))


def region_for_span(document: Document, start: int, end: int) -> Region:
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(document.text):
        raise ValueError("Invalid document span")
    for region in document.regions:
        if region.char_start <= start < end <= region.char_end:
            if not document.text[start:end].strip():
                raise ValueError("Span has no non-whitespace evidence")
            return region
    raise ValueError("Span crosses an excluded/blank page or lies outside eligible text")


def pages_for_span(document: Document, start: int, end: int) -> tuple[Page, ...]:
    region_for_span(document, start, end)
    return tuple(p for p in document.pages if p.char_start < p.char_end
                 and p.char_start < end and start < p.char_end)


def flags_for_span(document: Document, start: int, end: int) -> tuple[str, ...]:
    return tuple(sorted({f for p in pages_for_span(document, start, end) for f in p.quality_flags}))
