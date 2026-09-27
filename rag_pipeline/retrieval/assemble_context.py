"""Pack exact source fragments under a common evidence-token budget."""

from collections import defaultdict
import json

from ..common.load_corpus import require
from ..common.provenance import pages_for_span


def subtract_intervals(start, end, covered):
    """Return ordered portions of [start, end) not in the covered union."""
    pieces, cursor = [], start
    for left, right in sorted(covered):
        if right <= cursor or left >= end:
            continue
        if left > cursor:
            pieces.append((cursor, left))
        cursor = max(cursor, right)
        if cursor >= end:
            break
    if cursor < end:
        pieces.append((cursor, end))
    return pieces


def assemble_context(hits, documents, tokens, budget=1536):
    require(type(budget) is int and budget >= 0, "Invalid context budget")
    covered, pieces, decisions = defaultdict(list), [], []
    used = 0
    for hit in hits:
        doc = documents[hit["document_id"]]
        start, end = hit["char_start"], hit["char_end"]
        require(hit["text"] == doc.text[start:end], "Retrieved text differs from source")
        remaining = subtract_intervals(start, end, covered[doc.document_id])
        fragments = [(a, b, tokens.count(doc.text[a:b], special_tokens=False)) for a, b in remaining if doc.text[a:b].strip()]
        count = sum(t for _, _, t in fragments)
        decision = {"rank": hit["rank"], "chunk_id": hit["chunk_id"], "new_content_tokens": count,
                    "remaining_source_spans": [[a, b] for a, b, _ in fragments],
                    "overlap_characters_removed": end-start-sum(b-a for a, b in remaining)}
        if not fragments:
            decision["status"] = "skipped_no_new_nonwhitespace_text"
        elif used + count > budget:
            decision["status"] = "skipped_budget"
        else:
            decision["status"] = "included"
            for a, b, token_count in fragments:
                pages = [p.page_number for p in pages_for_span(doc, a, b)]
                citation_id = f"S{len(pieces)+1}"
                # JSON quoting keeps source filenames on one line and preserves
                # their exact version labels. They are metadata, not evidence.
                wrapper = f"[{citation_id}] source={json.dumps(doc.source, ensure_ascii=False)}; physical_pages={pages}; chars=[{a},{b})\n"
                pieces.append({"citation_id": citation_id, "chunk_id": hit["chunk_id"], "retrieval_rank": hit["rank"],
                               "document_id": doc.document_id, "source": doc.source, "member": doc.member,
                               "char_start": a, "char_end": b, "page_numbers": pages, "text": doc.text[a:b],
                               "content_token_count": token_count, "citation_wrapper": wrapper,
                               "wrapper_content_tokens": tokens.count(wrapper, special_tokens=False)})
                covered[doc.document_id].append((a, b))
            used += count
        decisions.append(decision)
    rendered = "\n\n".join(p["citation_wrapper"] + p["text"] for p in pieces)
    return {"schema_version": 1, "evidence_budget": budget, "evidence_content_tokens": used,
            "tokenizer_id": tokens.tokenizer_id, "pieces": pieces, "decisions": decisions,
            "citation_wrapper_tokens": sum(p["wrapper_content_tokens"] for p in pieces),
            "rendered_context": rendered, "rendered_context_content_tokens": tokens.count(rendered, special_tokens=False),
            "token_budget_note": "Evidence budget sums independently tokenized source pieces. Wrappers and separators are extra. "
            "Generator tokenization/capacity will be checked separately in Phase 10."}
