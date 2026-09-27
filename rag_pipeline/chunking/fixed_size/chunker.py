"""A LangChain text splitter that preserves exact source slices and token limits."""

from copy import deepcopy
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter

from ...common.load_corpus import require
from ...common.tokenization import ResearchTokenizer
from ...common.token_windows import TokenSpan, split_token_windows


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
        return split_token_windows(text, self.tokens, self._chunk_size)

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
