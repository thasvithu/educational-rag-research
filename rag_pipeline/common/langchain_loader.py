"""Load validated educational text using LangChain's familiar loader interface."""

from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from .config import ROOT, load_config, within
from .load_corpus import load_corpus


class PreparedCorpusLoader(BaseLoader):
    """Return one LangChain Document per continuous, eligible text region.

    A PDF can have several regions when excluded or blank pages interrupt it.
    Keeping them separate prevents a later splitter from bridging missing pages.
    Use load() for a list or lazy_load() to iterate over the validated regions.
    The complete corpus is validated first, before any region is yielded.
    """

    def __init__(self, config_path: Path = ROOT / "rag_pipeline/config.json", *, root: Path = ROOT):
        self.root = root.resolve()
        self.config = load_config(Path(config_path), self.root)
        self.corpus = None

    def lazy_load(self) -> Iterator[Document]:
        # Revalidate on each load: never reuse a stale snapshot after input edits.
        self.corpus = load_corpus(within(self.root, self.config.corpus_dir), self.config)
        for source in self.corpus.documents:
            for region in source.regions:
                pages = [p for p in source.pages if p.page_number in region.page_numbers]
                yield Document(
                    id=f"{source.document_id}:{region.region_id}",
                    page_content=source.text[region.char_start:region.char_end],
                    metadata={
                        "document_id": source.document_id,
                        "source": source.source,
                        "member": source.member,
                        "corpus_hash": self.corpus.corpus_hash,
                        "clean_sha256": source.clean_sha256,
                        "region_id": region.region_id,
                        "char_start": region.char_start,
                        "char_end": region.char_end,
                        "page_numbers": list(region.page_numbers),
                        "pages": [asdict(p) for p in pages],
                        "quality_flags": sorted({flag for p in pages for flag in p.quality_flags}),
                        "offset_convention": "document-level Python characters; end exclusive",
                    },
                )

    def load_and_split(self, text_splitter=None):
        # BaseLoader's generic splitter copies metadata without updating our spans.
        # Phase-specific chunkers will adapt LangChain splitters where appropriate.
        raise ValueError("Use load(), then a research chunker that records exact source offsets; "
                         "generic load_and_split() would leave region metadata on smaller chunks.")
