"""Exact FAISS indexes with LangChain Documents and explicit JSON source maps."""

import json

import faiss
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
import numpy as np

from ..common.load_corpus import file_hash, require
from ..common.run_manifest import write_json
from ..embedding.encode_chunks import atomic_jsonl


class PrecomputedQueries(Embeddings):
    """Protect the index from accidentally invoking a different query encoder.

    The shared runner computes queries once with Jina's LangChain adapter, saves
    their identity, then uses similarity_search_with_score_by_vector below.
    """
    def embed_documents(self, texts):
        raise ValueError("Use the verified shared embedding stage")

    def embed_query(self, text):
        raise ValueError("Supply a verified precomputed query vector")


def check_matrix(vectors, count, dimension):
    require(vectors.dtype == np.dtype("float32") and vectors.shape == (count, dimension), "Invalid index vector shape/type")
    require(count > 0 and np.isfinite(vectors).all(), "Empty/non-finite index vectors")
    require(np.allclose(np.linalg.norm(vectors, axis=1), 1, rtol=0, atol=1e-4), "Index vectors must be normalized")


def wrap_index(index, chunks):
    require(index.ntotal == len(chunks), "Index/document row count differs")
    ids = [row["chunk_id"] for row in chunks]
    require(len(set(ids)) == len(ids), "Duplicate index chunk IDs")
    docs = {row["chunk_id"]: Document(page_content=row["text"], metadata={k: v for k, v in row.items() if k != "text"})
            for row in chunks}
    return FAISS(embedding_function=PrecomputedQueries(), index=index, docstore=InMemoryDocstore(docs),
                 index_to_docstore_id=dict(enumerate(ids)), distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)


def build_index(directory, chunks, vectors, *, corpus_hash, embedding_hash, input_hashes):
    """Store supplied vectors unchanged; the index performs exhaustive search."""
    check_matrix(vectors, len(chunks), vectors.shape[1])
    require(len({r["method"] for r in chunks}) == 1, "Mixed methods in index")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors))
    faiss.write_index(index, str(directory / "index.faiss"))
    atomic_jsonl(directory / "index_rows.jsonl", [{"row_index": i, "chunk_id": r["chunk_id"]} for i, r in enumerate(chunks)])
    atomic_jsonl(directory / "chunks.jsonl", chunks)
    manifest = {"schema_version": 1, "method": chunks[0]["method"], "rows": len(chunks),
                "dimension": vectors.shape[1], "corpus_hash": corpus_hash, "embedding_hash": embedding_hash,
                "index_type": "IndexFlatIP", "similarity": "normalized_inner_product",
                "input_sha256": input_hashes,
                "artifact_sha256": {name: file_hash(directory / name) for name in ("index.faiss", "index_rows.jsonl", "chunks.jsonl")}}
    write_json(directory / "index_manifest.json", manifest)
    # Always read the exported files back before retrieval begins.
    store, reloaded = load_index(directory)
    require(reloaded == chunks and np.array_equal(store.index.reconstruct_n(0, len(chunks)), vectors),
            "Index reload changed vectors or documents")
    return store


def load_index(directory):
    manifest = json.loads((directory / "index_manifest.json").read_text())
    require(manifest["schema_version"] == 1 and manifest["index_type"] == "IndexFlatIP", "Unsupported index format")
    require(set(manifest["artifact_sha256"]) == {"index.faiss", "index_rows.jsonl", "chunks.jsonl"}, "Incomplete index manifest")
    for name, expected in manifest["artifact_sha256"].items():
        require(file_hash(directory / name) == expected, f"Index artifact changed: {name}")
    chunks = [json.loads(line) for line in (directory / "chunks.jsonl").read_text().splitlines()]
    mapping = [json.loads(line) for line in (directory / "index_rows.jsonl").read_text().splitlines()]
    require(mapping == [{"row_index": i, "chunk_id": r["chunk_id"]} for i, r in enumerate(chunks)], "Index row map differs")
    require(all(r["method"] == manifest["method"] and r["corpus_hash"] == manifest["corpus_hash"] for r in chunks),
            "Index mixes methods/corpora")
    index = faiss.read_index(str(directory / "index.faiss"))
    require(type(index) is faiss.IndexFlatIP and index.metric_type == faiss.METRIC_INNER_PRODUCT, "Wrong FAISS metric/type")
    require(index.ntotal == manifest["rows"] == len(chunks) and index.d == manifest["dimension"], "Index dimensions differ")
    check_matrix(index.reconstruct_n(0, index.ntotal), index.ntotal, index.d)
    return wrap_index(index, chunks), chunks
