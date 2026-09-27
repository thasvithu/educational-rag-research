"""Plan whole-chunk context windows and pool only each chunk's own tokens.

Five chunks normally own a window; one neighbor on each side supplies context.
Neighbors can occur in another window, but every baseline chunk has one owner.
All offsets remain Python character offsets into the unchanged prepared text.
"""

from itertools import groupby

from ...common.load_corpus import digest_json, require


def token_membership(offsets, spans):
    """Assign content tokens to exact spans, rejecting tokens crossing a cut.

    Tokenizer offsets may repeat for Unicode subtokens. All such tokens belong
    to the same span; zero-length offsets are never treated as source content.
    """
    result = []
    for start, end in spans:
        require(0 <= start < end, "Invalid pooling span")
        selected = []
        for index, (left, right) in enumerate(offsets):
            if right > left and left < end and right > start:
                require(start <= left and right <= end, "Context token crosses a chunk boundary")
                selected.append(index)
        require(bool(selected), "Chunk has no contextual content tokens")
        result.append(selected)
    return result


def plan_windows(rows, documents, tokens, *, core_chunks=5, halo_chunks=1, max_input_tokens=2048):
    """Greedily reduce ownership if needed; never shorten or split a chunk.

    At an unusually small configured limit, remove right then left halo chunks
    only after reducing the core to one. Each adjustment is recorded.
    """
    require(type(core_chunks) is int and core_chunks > 0, "Invalid core size")
    require(type(halo_chunks) is int and halo_chunks >= 0, "Invalid halo size")
    require(type(max_input_tokens) is int and 3 <= max_input_tokens <= tokens.max_input_tokens,
            "Invalid window limit")
    windows = []
    for (doc_id, region_id), group in groupby(rows, key=lambda row: (row["document_id"], row["region_id"])):
        chunks = list(group)
        doc = documents[doc_id]
        for left, right in zip(chunks, chunks[1:]):
            require(left["char_end"] == right["char_start"], "Baseline must be contiguous without overlap")
        cursor = 0
        while cursor < len(chunks):
            stop = min(len(chunks), cursor + core_chunks)
            left, right = max(0, cursor-halo_chunks), min(len(chunks), stop+halo_chunks)
            adjustments = 0
            while True:
                start, end = chunks[left]["char_start"], chunks[right-1]["char_end"]
                text = doc.text[start:end]
                count = tokens.count(text)
                if count <= max_input_tokens:
                    break
                adjustments += 1
                if stop-cursor > 1:
                    stop -= 1
                    right = min(len(chunks), stop+halo_chunks)
                elif right > stop:
                    right -= 1
                elif left < cursor:
                    left += 1
                else:
                    raise ValueError("A baseline chunk exceeds the context window limit")
            contained = chunks[left:right]
            offsets = tokens.offsets(text)
            spans = [(r["char_start"]-start, r["char_end"]-start) for r in contained]
            membership = token_membership(offsets, spans)
            require(sorted(i for indices in membership for i in indices) == list(range(len(offsets))),
                    "Context tokens are missing or multiply assigned")
            window = {"document_id": doc_id, "region_id": region_id, "char_start": start, "char_end": end,
                      "input_token_count": count, "content_token_count": len(offsets),
                      "early_chunk_ids": [r["chunk_id"] for r in contained],
                      "owned_chunk_ids": [r["chunk_id"] for r in chunks[cursor:stop]],
                      "spans": spans, "token_indices": membership, "budget_adjustments": adjustments}
            window["window_id"] = digest_json(window)
            windows.append(window)
            cursor = stop
    owned = [chunk for window in windows for chunk in window["owned_chunk_ids"]]
    require(owned == [r["chunk_id"] for r in rows] and len(set(owned)) == len(rows),
            "Each baseline chunk must have exactly one ordered owner")
    return windows


def pool_spans(hidden, encoded, spans):
    """Pool after contextual encoding, excluding special and padding positions.

    The LangChain Jina adapter supplies token vectors. This helper is necessary
    because ordinary embed_documents() pools an entire independent input.
    """
    import torch
    require(hidden.ndim == 3 and hidden.shape[0] == 1, "Late inference uses one window per batch")
    offsets = encoded["offset_mapping"][0].tolist()
    valid = (encoded["attention_mask"][0].bool() & ~encoded["special_tokens_mask"][0].bool()).cpu().tolist()
    # Masked tokens cannot trigger crossing checks or enter a mean.
    offsets = [tuple(offset) if keep else (0, 0) for offset, keep in zip(offsets, valid)]
    groups = token_membership(offsets, spans)
    vectors = []
    with torch.inference_mode():
        for indices in groups:
            mask = torch.zeros(hidden.shape[1], dtype=torch.bool, device=hidden.device)
            mask[indices] = True
            vector = (hidden[0] * mask.unsqueeze(-1)).sum(0) / len(indices)
            vectors.append(torch.nn.functional.normalize(vector, p=2, dim=0))
    result = torch.stack(vectors)
    require(bool(torch.isfinite(result).all()), "Invalid contextual vectors")
    return result.cpu().numpy(), groups
