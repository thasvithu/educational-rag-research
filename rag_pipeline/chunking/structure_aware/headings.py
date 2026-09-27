"""Conservative heading resolution. PDF font/outline hints are not ground truth."""

from collections import Counter, defaultdict
import re
import statistics


def normalized(text):
    return " ".join(text.split()).casefold()


def resolve_headings(document, layout):
    """Join three hint sources to real, unique lines on their physical PDF page.

    Prepared heading offsets are already document-level. Layout font hints and
    outline titles have no text offsets, so ambiguous/unmatched hints are rejected.
    No offsets are guessed from page coordinates or the first repeated substring.
    """
    pages = {p.page_number: p for p in document.pages if not p.is_barrier}
    lines, lookup = {}, defaultdict(list)
    for page in pages.values():
        start = page.char_start
        for line in document.text[page.char_start:page.char_end].splitlines(keepends=True):
            if line.strip():
                end = start + len(line.rstrip("\r\n"))
                lines[(start, end)] = (page.page_number, line.rstrip("\r\n"))
                lookup[(page.page_number, normalized(line))].append((start, end))
            start += len(line)

    sizes = [h.get("font_size") for p in layout["pages"] for h in p["heading_candidates"]
             if isinstance(h.get("font_size"), (int, float)) and h["font_size"] > 0]
    font_reference = statistics.median(sizes) if sizes else None
    candidates, decisions = [], []

    def add(origin, title, page_number, span=None, level=None, font_size=None):
        decision = {"origin": origin, "text": title, "page_number": page_number,
                    "status": "rejected", "reason": None}
        decisions.append(decision)
        if page_number not in pages:
            decision["reason"] = "excluded_blank_or_invalid_page"
            return
        matches = lookup[(page_number, normalized(title))]
        if span is not None:
            # Accept an indented full line, but not a bold fragment inside prose.
            matches = [m for m in matches if m[0] <= span[0] < span[1] <= m[1]
                       and normalized(document.text[span[0]:span[1]]) == normalized(title)]
        if len(matches) != 1:
            decision["reason"] = "ambiguous_line" if matches else "no_complete_source_line"
            return
        start, end = matches[0]
        decision.update(char_start=start, char_end=end)
        candidates.append(dict(decision=decision, start=start, end=end, level=level, font_size=font_size))

    for page in pages.values():
        for h in page.headings:
            add("prepared:" + h.origin, h.text, page.page_number, (h.char_start, h.char_end), h.level)
    for page in layout["pages"]:
        for hint in page["heading_candidates"]:
            add("font_candidate", hint["text"], page["page_number"], font_size=hint.get("font_size"))
    for item in layout["pdf_toc"]:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            decisions.append({"origin": "pdf_outline", "status": "rejected", "reason": "invalid_outline_entry"})
            continue
        level, title, page_number = item[:3]
        if not isinstance(title, str) or type(page_number) is not int or type(level) is not int or not 1 <= level <= 6:
            decisions.append({"origin": "pdf_outline", "status": "rejected", "reason": "invalid_outline_entry"})
            continue
        # PowerPoint may add a "Slide N:" label that is not part of the title.
        title = re.sub(r"^Slide\s+\d+:\s*", "", title, flags=re.I)
        add("pdf_outline", title, page_number, level=level)

    groups = defaultdict(list)
    for candidate in candidates:
        groups[(candidate["start"], candidate["end"])].append(candidate)
    accepted = []
    for (start, end), group in sorted(groups.items()):
        _, raw = lines[(start, end)]
        title = " ".join(raw.split())
        outlines = [g for g in group if g["decision"]["origin"] == "pdf_outline"]
        large = any(g["font_size"] and font_reference and g["font_size"] >= 1.15 * font_reference for g in group)
        numbered = re.match(r"^(\d{1,2}(?:\.\d{1,2})*)\.?\s+([A-Za-z].*)$", title)
        bullet = re.match(r"^(?:[•*–-]|\(?[a-zA-Z]\))\s", title)
        words = re.findall(r"[A-Za-z]+", title)
        title_shape = all(w[0].isupper() or w in {"of", "and", "in", "to", "for", "the", "with", "on", "a", "an", "as", "at"}
                          for w in words)
        explicit_title = title_shape and any(g["level"] is not None for g in group)
        reason = None
        if len(title) > 120 or len(title.split()) > 16 or not any(c.isalpha() for c in title):
            reason = "not_short_heading_text"
        elif bullet or re.search(r"\.{3}|https?://|@", title):
            reason = "list_toc_or_contact_line"
        elif re.search(r"[.;,]$", title):
            reason = "sentence_punctuation"
        elif re.match(r"^\d+\.\s", title) and not (outlines or large or title.isupper()):
            reason = "enumerated_line_without_heading_support"
        elif re.search(r"\S[ \t]{3,}\S", re.sub(r"^\s*\d+(?:\.\d+)*\.?\s+", "", raw)):
            reason = "column_like_line"
        elif not (outlines or large or title.isupper() or numbered or explicit_title):
            reason = "insufficient_heading_evidence"
        if reason:
            for g in group:
                g["decision"]["reason"] = reason
            continue
        levels = [g["level"] for g in group if g["level"] is not None]
        if outlines:
            level, level_rule = outlines[0]["level"], "pdf_outline"
        elif numbered:
            level, level_rule = min(6, numbered[1].count(".") + 1), "numbering_depth"
        elif levels:
            level, level_rule = min(levels), "prepared_inferred_level"
        else:
            level, level_rule = 2, "unknown_level_as_sibling_2"
        origins = sorted({g["decision"]["origin"] for g in group})
        accepted.append({"text": title, "char_start": start, "char_end": end,
                         "level": level, "level_rule": level_rule, "origins": origins})
        for i, g in enumerate(group):
            g["decision"].update(status="accepted" if i == 0 else "duplicate", reason="same_source_span" if i else "heading_shape_and_metadata")

    summary = {"source": document.source, "candidate_count": len(decisions), "accepted_headings": len(accepted),
               "status_counts": dict(Counter(d["status"] for d in decisions)),
               "rejection_reasons": dict(Counter(d["reason"] for d in decisions if d["status"] == "rejected")),
               "candidate_origins": dict(Counter(d["origin"] for d in decisions)),
               "accepted_origins": dict(Counter(o for h in accepted for o in h["origins"])),
               "font_candidate_median": font_reference,
               "detected_table_pages": [p["page_number"] for p in layout["pages"] if p["detected_tables"]]}
    return accepted, decisions, summary
