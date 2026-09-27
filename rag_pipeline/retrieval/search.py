"""One ranking policy for all methods, with global deterministic tie handling."""

import numpy as np

from ..common.load_corpus import require
from ..embedding.vector_cache import check_vector


def search(store, query_vector, k=10):
    require(type(k) is int and k > 0, "K must be a positive integer")
    query = check_vector(query_vector, store.index.d)
    # At this corpus size, retrieving all exact scores is cheap. Asking only for
    # K would let FAISS choose an arbitrary subset of ties at the K boundary.
    matches = store.similarity_search_with_score_by_vector(query.tolist(), k=store.index.ntotal)
    require(all(np.isfinite(score) for _, score in matches), "Non-finite retrieval score")
    matches.sort(key=lambda pair: (-float(pair[1]), pair[0].metadata["chunk_id"]))
    return [{"rank": rank, "score": float(score), "text": doc.page_content, **doc.metadata}
            for rank, (doc, score) in enumerate(matches[:k], 1)]
