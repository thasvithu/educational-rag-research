"""Context ownership, pooling, Unicode offsets and recovery without GPU access."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from rag_pipeline.chunking.late.cache import WindowCache
from rag_pipeline.chunking.late.chunker import plan_windows, pool_spans, token_membership
from rag_pipeline.embedding.model_adapter import content_mean
from rag_pipeline.tests import test_fixed_size as fixtures


class LateTests(unittest.TestCase):
    def setUp(self):
        fixtures.FixedSizeTests.setUp(self)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "windows.sqlite3"

    def fixture(self, text="echo " * 12):
        spans = self.splitter.split_spans(text)
        rows = [{"document_id": "doc", "region_id": "region", "chunk_id": str(i),
                 "char_start": s.char_start, "char_end": s.char_end} for i, s in enumerate(spans)]
        return rows, {"doc": SimpleNamespace(text=text)}

    def test_windows_cover_exact_baseline_once_with_halos_and_tail(self):
        rows, docs = self.fixture()
        windows = plan_windows(rows, docs, self.tokens, core_chunks=1, halo_chunks=1, max_input_tokens=10)
        self.assertEqual([cid for w in windows for cid in w["owned_chunk_ids"]], [r["chunk_id"] for r in rows])
        self.assertTrue(all(w["input_token_count"] <= 10 for w in windows))
        self.assertTrue(any(w["budget_adjustments"] for w in windows))
        self.assertTrue(any(len(w["early_chunk_ids"]) > len(w["owned_chunk_ids"]) for w in windows))
        self.assertEqual(windows[-1]["char_end"], len(docs["doc"].text))
        self.assertEqual(windows, plan_windows(rows, docs, self.tokens, core_chunks=1, halo_chunks=1, max_input_tokens=10))

    def test_gaps_and_documents_are_never_bridged(self):
        rows, docs = self.fixture()
        rows[2]["region_id"] = rows[3]["region_id"] = "after_gap"
        rows[3]["document_id"] = "other"
        docs["other"] = docs["doc"]
        windows = plan_windows(rows, docs, self.tokens, max_input_tokens=10)
        self.assertEqual([(w["document_id"], w["region_id"]) for w in windows],
                         [("doc", "region"), ("doc", "after_gap"), ("other", "after_gap")])

    def test_invalid_limits_and_duplicate_ownership_fail(self):
        rows, docs = self.fixture()
        for kwargs in ({"core_chunks": 0}, {"halo_chunks": -1}, {"core_chunks": True},
                       {"max_input_tokens": 11}, {"max_input_tokens": 3}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                plan_windows(rows, docs, self.tokens, **kwargs)
        rows[1]["chunk_id"] = rows[0]["chunk_id"]
        with self.assertRaises(ValueError): plan_windows(rows, docs, self.tokens, max_input_tokens=10)

    def test_unicode_repeated_offsets_and_cross_boundary_rejection(self):
        self.assertEqual(token_membership([(0, 1), (0, 1), (2, 4)], [(0, 2), (2, 4)]), [[0, 1], [2]])
        with self.assertRaisesRegex(ValueError, "crosses"):
            token_membership([(0, 3)], [(0, 2)])
        with self.assertRaisesRegex(ValueError, "no contextual"):
            token_membership([(0, 0)], [(0, 1)])
        rows, docs = self.fixture("  Café தமிழ் unbelievable echo rule echo ")
        plan_windows(rows, docs, self.tokens, core_chunks=1, max_input_tokens=10)

    def test_pool_excludes_padding_specials_and_neighbor_values(self):
        encoded = {"offset_mapping": torch.tensor([[[0, 0], [0, 4], [5, 9], [0, 0], [0, 4]]]),
                   "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
                   "special_tokens_mask": torch.tensor([[1, 0, 0, 1, 0]])}
        hidden = torch.tensor([[[100., 100.], [1., 0.], [0., 2.], [100., 100.], [100., 100.]]])
        vectors, groups = pool_spans(hidden, encoded, [(0, 5), (5, 9)])
        np.testing.assert_array_equal(vectors, [[1., 0.], [0., 1.]])
        self.assertEqual(groups, [[1], [2]])
        whole, _ = pool_spans(hidden, encoded, [(0, 9)])
        np.testing.assert_allclose(whole, content_mean(hidden, encoded).numpy(), atol=1e-7)

    def cache(self, factory, only=False, identity=None):
        result = WindowCache(self.path, identity or {"dimension": 2, "model": "fake"}, factory, cache_only=only)
        self.addCleanup(result.close)
        return result

    def test_cache_context_identity_corruption_and_atomic_recovery(self):
        tokens = self.tokens
        class FakeModel:
            device = "cpu"
            def encode_tokens(self, texts):
                encoded = tokens.tokenizer(texts, return_tensors="pt", return_offsets_mapping=True,
                                           return_special_tokens_mask=True)
                hidden = torch.ones((*encoded["input_ids"].shape, 2))
                return encoded, hidden
            def embed_documents(self, texts):
                encoded, hidden = self.encode_tokens(texts)
                return content_mean(hidden, encoded).tolist()
        rows, docs = self.fixture("echo rule echo ")
        window = plan_windows(rows, docs, tokens, max_input_tokens=10)[0]
        text = docs["doc"].text
        cold = self.cache(FakeModel)
        vectors, key = cold.encode(text, window)
        self.assertTrue(cold.parity_probe["passed"])
        self.assertNotEqual(key, cold.key("rule " + text, window["spans"]))
        self.assertNotEqual(key, cold.key(text, [(0, 5)]))
        def never(): raise AssertionError("cached repeat must not load the model")
        warm = self.cache(never, only=True)
        np.testing.assert_array_equal(warm.encode(text, window)[0], vectors)
        changed = self.cache(never, only=True, identity={"dimension": 2, "model": "different"})
        with self.assertRaisesRegex(ValueError, "cache miss"): changed.encode(text, window)
        # An interrupted inference does not commit a partial window.
        class Failing(FakeModel):
            def encode_tokens(self, texts): raise RuntimeError("interrupted")
        failing = self.cache(Failing)
        with self.assertRaisesRegex(RuntimeError, "interrupted"): failing.encode("rule echo rule ", window)
        self.assertEqual(failing.connection.execute("SELECT count(*) FROM windows").fetchone()[0], 1)
        recovered = self.cache(FakeModel)
        recovered.encode("rule echo rule ", window)
        self.assertEqual(recovered.misses, 1)
        warm.connection.execute("UPDATE windows SET vectors=? WHERE key=?", (b"broken", key))
        warm.connection.commit()
        with self.assertRaisesRegex(ValueError, "checksum"): warm.encode(text, window)


if __name__ == "__main__":
    unittest.main()
