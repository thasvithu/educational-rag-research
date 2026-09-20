"""A LangChain text splitter that preserves exact source slices and token limits."""

from bisect import bisect_right
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter

from ...common.load_corpus import require
from ...common.tokenization import ResearchTokenizer


@dataclass(frozen=True)
class TokenSpan:
    char_start: int
    char_end: int
    region_token_start: int
    region_token_end: int
    content_token_count: int
    token_count: int  # Actual encoder input, including CLS/SEP.
    boundary_backoffs: int


def load_settings(path: Path) -> dict:
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {"schema_version": 1, "method": "fixed_size", "algorithm_version": "1", "overlap": 0,
                "boundary_policy": "safe_token_offsets_with_retokenization_check",
                "whitespace_policy": "leading_to_first_intertoken_to_left_trailing_to_last"}
    require(set(settings) == set(expected) | {"chunk_size"}, "Unknown/missing fixed-size setting")
    require(all(type(settings[k]) is type(v) and settings[k] == v for k, v in expected.items()),
            "Unsupported fixed-size policy or overlap")
    require(type(settings["chunk_size"]) is int and settings["chunk_size"] > 0, "chunk_size must be positive")
    return settings


class FixedSizeSplitter(TextSplitter):
    """Use LangChain's split_text/split_documents/transform_documents interface.

    Boundaries come from the Jina token stream, not sentence or paragraph layout.
    We slice the original string: decoding tokens can change case, spaces and
    accents. Each slice is retokenized because cutting a WordPiece word may change
    its standalone encoding. Reduce the window if needed to keep the hard cap.
    """

    def __init__(self, tokenizer: ResearchTokenizer, chunk_size: int = 256):
        require(type(chunk_size) is int and chunk_size > 0, "chunk_size must be a positive integer")
        require(chunk_size + tokenizer.special_tokens <= tokenizer.max_input_tokens,
                "Content budget plus special tokens exceeds the encoder limit")
        self.tokens = tokenizer
        super().__init__(chunk_size=chunk_size, chunk_overlap=0,
                         length_function=lambda text: tokenizer.count(text, special_tokens=False),
                         strip_whitespace=False)

    def split_spans(self, text: str) -> list[TokenSpan]:
        if not text.strip():
            return []
        offsets = self.tokens.offsets(text)
        require(bool(offsets), "Nonblank input has no encodable content tokens; review this region")

        # Some tokenizers map multiple subtokens to the same source character.
        # A safe cut must lie after all previous token ends, never inside a group.
        safe = [0]
        previous_end = 0
        for i, (start, end) in enumerate(offsets):
            if i and start >= previous_end:
                safe.append(i)
            previous_end = max(previous_end, end)
        safe.append(len(offsets))

        spans = []
        token_start, char_start = 0, 0
        while token_start < len(offsets):
            target = min(token_start + self._chunk_size, len(offsets))
            candidate = bisect_right(safe, target) - 1
            # If one indivisible character group exceeds the nominal window,
            # inspect that whole group; fail explicitly if it cannot fit.
            if safe[candidate] <= token_start:
                candidate = bisect_right(safe, token_start)
            backoffs = 0
            while safe[candidate] > token_start:
                token_end = safe[candidate]
                char_end = len(text) if token_end == len(offsets) else offsets[token_end][0]
                piece = text[char_start:char_end]
                content_count = self.tokens.count(piece, special_tokens=False)
                input_count = self.tokens.count(piece)
                if 0 < content_count <= self._chunk_size and input_count <= self.tokens.max_input_tokens:
                    break
                candidate -= 1
                backoffs += 1
            else:
                raise ValueError("An indivisible token-offset group cannot fit the budget; no text was dropped")

            spans.append(TokenSpan(char_start, char_end, token_start, token_end,
                                   content_count, input_count, backoffs))
            char_start, token_start = char_end, token_end
        require(char_start == len(text), "Internal error: unprocessed document tail")
        return spans

    def split_text(self, text: str) -> list[str]:
        return [text[span.char_start:span.char_end] for span in self.split_spans(text)]

    def create_documents(self, texts: list[str], metadatas: list[dict] | None = None) -> list[Document]:
        """Update offsets instead of LangChain's default first-match text search."""
        if metadatas is None:
            metadatas = [{} for _ in texts]
        require(len(texts) == len(metadatas), "Each text needs matching metadata")
        chunks = []
        for text, original in zip(texts, metadatas):
            base = original.get("char_start", 0)
            require(type(base) is int and base >= 0, "Invalid document offset")
            require(original.get("char_end", base + len(text)) == base + len(text), "Region length differs from metadata")
            for span in self.split_spans(text):
                start, end = base + span.char_start, base + span.char_end
                metadata = deepcopy(original)
                metadata.update(char_start=start, char_end=end, start_index=span.char_start,
                                region_char_start=base, token_count=span.token_count,
                                tokenizer_id=self.tokens.tokenizer_id,
                                method_metadata={"content_token_count": span.content_token_count,
                                                 "special_token_count": self.tokens.special_tokens,
                                                 "region_token_start": span.region_token_start,
                                                 "region_token_end": span.region_token_end,
                                                 "boundary_backoffs": span.boundary_backoffs})
                if "pages" in metadata:
                    metadata["pages"] = [p for p in metadata["pages"] if p["char_start"] < end and start < p["char_end"]]
                    metadata["page_numbers"] = [p["page_number"] for p in metadata["pages"]]
                    metadata["quality_flags"] = sorted({f for p in metadata["pages"] for f in p["quality_flags"]})
                chunks.append(Document(page_content=text[span.char_start:span.char_end], metadata=metadata))
        return chunks
