"""A short, offline example for the team: python -m rag_pipeline.examples.inspect_langchain."""

from rag_pipeline.common.langchain_loader import PreparedCorpusLoader


def main():
    # The inherited LangChain load() method calls our validated lazy_load().
    loader = PreparedCorpusLoader()
    documents = loader.load()

    print(f"Source PDFs: {len(loader.corpus.documents)}")
    print(f"Continuous text regions: {len(documents)}")
    print("A region is input to a chunker, not a finished chunk.")

    # Each member gets one visible example without printing the whole corpus.
    for member in sorted({document.metadata["member"] for document in documents}):
        document = next(document for document in documents if document.metadata["member"] == member)
        print(f"\n{member}: {document.metadata['source']}")
        print(f"Physical pages: {document.metadata['page_numbers']}")
        print(f"Document offset: {document.metadata['char_start']}")
        print(document.page_content[:250])


if __name__ == "__main__":
    main()
