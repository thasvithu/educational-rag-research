"""LangChain sliding windows using the fixed baseline's exact-offset rules."""

import json
from pathlib import Path

from ...common.load_corpus import require
from ...common.token_windows import split_token_windows
from ..fixed_size.chunker import FixedSizeSplitter


def load_settings(path: Path) -> dict:
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {"schema_version": 1, "method": "sliding_window", "algorithm_version": "1",
                "boundary_policy": "safe_token_offsets_with_retokenization_check",
                "whitespace_policy": "leading_to_first_intertoken_to_left_trailing_to_last",
                "overlap_policy": "previous_end_minus_overlap_safe_start_with_forward_progress"}
    require(set(settings) == set(expected) | {"chunk_size", "overlap"}, "Unknown/missing sliding-window setting")
    require(all(type(settings[k]) is type(v) and settings[k] == v for k, v in expected.items()),
            "Unsupported sliding-window policy")
    size, overlap = settings["chunk_size"], settings["overlap"]
    require(type(size) is int and size > 0, "chunk_size must be positive")
    require(type(overlap) is int and 0 <= overlap < size, "Require 0 <= overlap < chunk_size")
    return settings


class SlidingWindowSplitter(FixedSizeSplitter):
    """Reuse LangChain interfaces and exact page/character metadata handling.

    The normal stride is chunk_size - overlap. Adjustments for source characters
    and independent token counts are explicit in each chunk's metadata.
    """

    def __init__(self, tokenizer, chunk_size=256, overlap=64):
        super().__init__(tokenizer, chunk_size)
        require(type(overlap) is int and 0 <= overlap < chunk_size, "Require 0 <= overlap < chunk_size")
        self._chunk_overlap = overlap

    def split_spans(self, text):
        return split_token_windows(text, self.tokens, self._chunk_size, self._chunk_overlap)

    def create_documents(self, texts, metadatas=None):
        if metadatas is None:
            metadatas = [{} for _ in texts]
        require(len(texts) == len(metadatas), "Each text needs matching metadata")
        result = []
        for text, metadata in zip(texts, metadatas):
            chunks = super().create_documents([text], [metadata])
            previous = None
            for chunk in chunks:
                current = chunk.metadata
                method = current["method_metadata"]
                shared_chars = 0 if previous is None else previous["char_end"] - current["char_start"]
                shared_tokens = (0 if previous is None else
                                 previous["method_metadata"]["region_token_end"] - method["region_token_start"])
                method.update(requested_overlap_tokens=self._chunk_overlap,
                              nominal_stride_tokens=self._chunk_size - self._chunk_overlap,
                              overlap_with_previous_region_tokens=shared_tokens,
                              overlap_with_previous_characters=shared_chars,
                              overlap_text_content_tokens=self.tokens.count(
                                  chunk.page_content[:shared_chars], special_tokens=False) if shared_chars else 0,
                              overlap_reduced=previous is not None and shared_tokens < self._chunk_overlap)
                previous = current
            result.extend(chunks)
        return result
