"""Exact ranking, source mapping and context-budget checks with synthetic data."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import faiss
import numpy as np

from rag_pipeline.common.load_corpus import file_hash
from rag_pipeline.common.provenance import Document, Page, eligible_regions
from rag_pipeline.retrieval.assemble_context import assemble_context, subtract_intervals
from rag_pipeline.retrieval.build_index import build_index, load_index
from rag_pipeline.retrieval.diagnostics import covered, diagnose
from rag_pipeline.retrieval.inputs import compatible_identity, load_settings
from rag_pipeline.retrieval.search import search
from rag_pipeline.tests import test_fixed_size as fixtures


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        fixtures.FixedSizeTests.setUp(self)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        text = "echo rule cafe echo rule cafe"
        pages = (Page(1, "1", 0, 15, True, (), (), False), Page(2, "2", 15, len(text), True, (), (), False))
        self.doc = Document("a"*64, "member/handbook_2024.pdf", "member", text, "b"*64,
                            "unused", "c"*64, pages, eligible_regions(pages))
        self.documents = {self.doc.document_id: self.doc}
        self.rows = [{"chunk_id": name, "method": "fixed_size", "corpus_hash": "d"*64,
                      "document_id": self.doc.document_id, "source": self.doc.source, "member": "member",
                      "char_start": start, "char_end": end, "text": text[start:end], "page_numbers": [1]}
                     for name, start, end in (("z", 0, 5), ("a", 5, 10), ("b", 10, 15))]
        self.vectors = np.asarray([[1, 0], [1, 0], [0, 1]], dtype="float32")
        faiss.omp_set_num_threads(1)

    def build(self, rows=None, vectors=None):
        return build_index(self.root, rows or self.rows, vectors if vectors is not None else self.vectors,
                           corpus_hash="d"*64, embedding_hash="e"*64, input_hashes={})

    def hit(self, start, end, rank=1, cid=None):
        return {"rank": rank, "chunk_id": cid or str(rank), "document_id": self.doc.document_id,
                "char_start": start, "char_end": end, "text": self.doc.text[start:end]}

    def test_global_ties_at_k_boundary_and_reload_keep_original_source(self):
        store = self.build()
        self.assertEqual([r["chunk_id"] for r in search(store, [1, 0], 1)], ["a"])
        self.assertEqual([r["chunk_id"] for r in search(store, [1, 0], 10)], ["a", "z", "b"])
        self.assertEqual(search(store, [0, 1], 1)[0]["text"], "cafe ")
        loaded, rows = load_index(self.root)
        self.assertEqual(rows, self.rows)
        self.assertEqual(search(loaded, [1, 0], 3), search(store, [1, 0], 3))
        np.testing.assert_array_equal(loaded.index.reconstruct_n(0, 3), self.vectors)

    def test_negative_similarities_and_known_nearest_vector(self):
        vectors = np.array([[-1, 0], [0, -1], [0.6, 0.8]], dtype="float32")
        hits = search(self.build(vectors=vectors), [1, 0], 3)
        self.assertEqual([h["chunk_id"] for h in hits], ["b", "a", "z"])
        np.testing.assert_allclose([h["score"] for h in hits], [.6, 0, -1], atol=1e-7)

    def test_bad_vectors_duplicate_ids_and_mixed_methods_rejected(self):
        for vectors in (self.vectors.astype(float), self.vectors*2, np.zeros((3, 2), dtype="float32"),
                        np.full((3, 2), np.nan, dtype="float32")):
            with self.assertRaises(ValueError): self.build(vectors=vectors)
        duplicate = deepcopy(self.rows); duplicate[1]["chunk_id"] = duplicate[0]["chunk_id"]
        with self.assertRaisesRegex(ValueError, "Duplicate"): self.build(rows=duplicate)
        mixed = deepcopy(self.rows); mixed[1]["method"] = "late"
        with self.assertRaisesRegex(ValueError, "Mixed"): self.build(rows=mixed)

    def test_mapping_corruption_and_wrong_index_metric_rejected(self):
        self.build()
        path = self.root / "index_rows.jsonl"
        path.write_text(path.read_text().replace('"z"', '"unknown"'))
        with self.assertRaisesRegex(ValueError, "changed"): load_index(self.root)
        manifest_path = self.root / "index_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifact_sha256"]["index_rows.jsonl"] = file_hash(path)
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "row map"): load_index(self.root)
        self.build()
        wrong = faiss.IndexFlatL2(2); wrong.add(self.vectors)
        faiss.write_index(wrong, str(self.root / "index.faiss"))
        manifest = json.loads(manifest_path.read_text())
        manifest["artifact_sha256"]["index.faiss"] = file_hash(self.root / "index.faiss")
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "metric/type"): load_index(self.root)

    def test_invalid_query_vectors_k_and_implicit_encoder_are_rejected(self):
        store = self.build()
        for query in ([0, 0], [2, 0], [1, 0, 0], [float('nan'), 0]):
            with self.assertRaises(ValueError): search(store, query)
        for k in (0, -1, True, 1.5):
            with self.assertRaises(ValueError): search(store, [1, 0], k)
        with self.assertRaisesRegex(ValueError, "precomputed"): store.similarity_search("question")

    def test_subtract_nested_overlapping_intervals_and_multiple_pieces(self):
        self.assertEqual(subtract_intervals(0, 20, [(4, 8), (6, 10), (12, 15), (12, 13)]), [(0, 4), (10, 12), (15, 20)])
        self.assertEqual(subtract_intervals(5, 8, [(0, 20)]), [])
        self.assertEqual(subtract_intervals(5, 8, [(0, 5), (8, 10)]), [(5, 8)])

    def test_overlap_removed_without_rejoining_or_changing_text(self):
        context = assemble_context([self.hit(5, 10), self.hit(0, 15, 2)], self.documents, self.tokens, 3)
        self.assertEqual([(p["char_start"], p["char_end"]) for p in context["pieces"]], [(5, 10), (0, 5), (10, 15)])
        self.assertEqual(context["evidence_content_tokens"], 3)
        self.assertEqual(context["decisions"][1]["overlap_characters_removed"], 5)
        for piece in context["pieces"]:
            self.assertEqual(piece["text"], self.doc.text[piece["char_start"]:piece["char_end"]])
        self.assertGreater(context["rendered_context_content_tokens"], context["evidence_content_tokens"])

    def test_budget_skip_tries_next_and_does_not_reserve_skipped_source(self):
        context = assemble_context([self.hit(0, 15), self.hit(0, 5, 2), self.hit(5, 10, 3)], self.documents, self.tokens, 2)
        self.assertEqual([d["status"] for d in context["decisions"]], ["skipped_budget", "included", "included"])
        self.assertEqual(context["evidence_content_tokens"], 2)
        self.assertEqual([p["chunk_id"] for p in context["pieces"]], ["2", "3"])
        self.assertEqual(assemble_context([self.hit(0, 5)], self.documents, self.tokens, 0)["pieces"], [])

    def test_cross_page_mapping_duplicates_and_repeated_text(self):
        hits = [self.hit(10, 20), self.hit(10, 20, 2), self.hit(0, 5, 3), self.hit(15, 20, 4)]
        context = assemble_context(hits, self.documents, self.tokens, 10)
        self.assertEqual(context["pieces"][0]["page_numbers"], [1, 2])
        self.assertEqual([d["status"] for d in context["decisions"]],
                         ["included", "skipped_no_new_nonwhitespace_text", "included", "skipped_no_new_nonwhitespace_text"])
        altered = self.hit(0, 5); altered["text"] = "wrong"
        with self.assertRaisesRegex(ValueError, "differs"): assemble_context([altered], self.documents, self.tokens)

    def test_evidence_requires_full_nonwhitespace_union_in_correct_document(self):
        alternative = {"document_id": self.doc.document_id, "char_start": 0, "char_end": 10}
        self.assertFalse(covered(alternative, [self.hit(0, 5)], self.documents))
        self.assertTrue(covered(alternative, [self.hit(0, 4), self.hit(5, 9)], self.documents))
        wrong = self.hit(0, 10); wrong["document_id"] = "wrong version"
        self.assertFalse(covered(alternative, [wrong], self.documents))
        question = {"question_id": "q", "answerability": "answerable", "review": {"status": "pending"},
                    "evidence_units": [{"unit_id": "u", "alternatives": [alternative]}]}
        result = diagnose(question, [self.hit(0, 5), self.hit(5, 10, 2)], {"pieces": [self.hit(0, 5)]}, self.documents, [1, 3])
        unit = result["units"][0]
        self.assertEqual(unit["covered_at_k"], {"1": False, "3": True})
        self.assertIsNone(unit["first_single_chunk_rank"])
        self.assertFalse(unit["covered_in_context"])
        question["answerability"] = "ambiguous"
        self.assertEqual(diagnose(question, [], {"pieces": []}, self.documents, [1])["units"], [])

    def test_model_compatibility_allows_batch_difference_but_not_pooling_or_revision(self):
        current = {"purpose": "independent_chunk_embedding", "batch_policy": "batch", "dimension": 2,
                   "embedding": {"batch_size": 1, "pooling": "content_mean_l2", "revision": "pinned"}}
        saved = deepcopy(current); saved["embedding"]["batch_size"] = 8
        compatible_identity(current, saved, "fixed_size")
        saved["purpose"] = "late_contextual_chunk_embedding"
        compatible_identity(current, saved, "late")
        for key in ("pooling", "revision"):
            wrong = deepcopy(saved); wrong["embedding"][key] = "different"
            with self.assertRaisesRegex(ValueError, "identity differs"): compatible_identity(current, wrong, "late")


if __name__ == "__main__":
    unittest.main()
