"""Exact source windows shared by fixed-size and sliding-window chunking."""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from .load_corpus import require


@dataclass(frozen=True)
class TokenSpan:
    char_start: int
    char_end: int
    region_token_start: int
    region_token_end: int
    content_token_count: int
    token_count: int  # Encoder input, including special tokens.
    boundary_backoffs: int


def split_token_windows(text, tokens, chunk_size, overlap=0) -> list[TokenSpan]:
    """Partition a region, optionally sharing up to `overlap` source tokens.

    Overlap is measured in the original region's token stream. A substring can
    tokenize differently on its own, so every accepted window is counted again.
    Starts and ends both advance; the first window reaching the tail is the last.
    """
    require(type(chunk_size) is int and chunk_size > 0, "chunk_size must be positive")
    require(type(overlap) is int and 0 <= overlap < chunk_size,
            "overlap must be an integer from zero to chunk_size - 1")
    if not text.strip():
        return []
    offsets = tokens.offsets(text)
    require(bool(offsets), "Nonblank input has no encodable content tokens; review this region")

    # Do not cut between subtokens referring to the same source character.
    safe = [0]
    previous_end = 0
    for i, (start, end) in enumerate(offsets):
        if i and start >= previous_end:
            safe.append(i)
        previous_end = max(previous_end, end)
    safe.append(len(offsets))

    spans = []
    token_start = 0
    while token_start < len(offsets):
        char_start = 0 if token_start == 0 else offsets[token_start][0]
        target = min(token_start + chunk_size, len(offsets))
        candidate = bisect_right(safe, target) - 1
        if safe[candidate] <= token_start:
            candidate = bisect_right(safe, token_start)
        backoffs = 0
        while safe[candidate] > token_start:
            token_end = safe[candidate]
            char_end = len(text) if token_end == len(offsets) else offsets[token_end][0]
            piece = text[char_start:char_end]
            content_count = tokens.count(piece, special_tokens=False)
            input_count = tokens.count(piece)
            if 0 < content_count <= chunk_size and input_count <= tokens.max_input_tokens:
                break
            candidate -= 1
            backoffs += 1
        else:
            raise ValueError("An indivisible token-offset group cannot fit the budget; no text was dropped")

        if spans and token_end <= spans[-1].region_token_end:
            # Retokenization can shrink a high-overlap candidate so much that it
            # adds no new evidence. Reduce overlap instead of saving redundancy.
            token_start = safe[bisect_right(safe, token_start)]
            require(token_start <= spans[-1].region_token_end, "Window adjustment left a gap")
            continue

        spans.append(TokenSpan(char_start, char_end, token_start, token_end,
                               content_count, input_count, backoffs))
        if token_end == len(offsets):
            break  # No extra suffix window after the tail has been covered.
        # Ceiling to a safe start gives at most the requested source-token overlap.
        next_target = max(token_start + 1, token_end - overlap)
        token_start = safe[bisect_left(safe, next_target)]
        require(token_start <= token_end, "Internal error: uncovered token gap")

    require(spans[-1].char_end == len(text), "Internal error: unprocessed document tail")
    return spans
