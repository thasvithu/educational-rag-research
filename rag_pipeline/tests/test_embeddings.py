"""Vector integrity and recovery with controlled LangChain embeddings; no GPU needed."""

from copy import deepcopy
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from langchain_core.embeddings import Embeddings
import numpy as np

from rag_pipeline.common.chunk_schema import make_chunk
from rag_pipeline.common.provenance import Document, Page, eligible_regions
from rag_pipeline.embedding.encode_chunks import encode_variant, exclusive_run, validate_outputs
from rag_pipeline.embedding.inputs import validate_rows
from rag_pipeline.embedding.vector_cache import ChunkEmbeddings
from rag_pipeline.tests import test_fixed_size as fixtures


class FakeModel(Embeddings):
    def __init__(self, fail_at=None):
        self.calls, self.fail_at = [], fail_at

    def embed_documents(self, texts):
        if self.fail_at is not None and len(self.calls) == self.fail_at:
            raise RuntimeError("simulated interruption")
        self.calls.append(list(texts))
        return [[0., 1.] if "rule" in text else [1., 0.] for text in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        fixtures.FixedSizeTests.setUp(self)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.identity = {"dimension": 2, "embedding": {"batch_size": 2, "device": "cpu"}, "pooling": "content_mean_l2"}
        text = "echo rule echo "
        pages = (Page(1, "1", 0, len(text), True, (), (), False),)
        self.doc = Document("a"*64, "member/synthetic.pdf", "member", text, "b"*64, "unused", "c"*64,
                            pages, eligible_regions(pages))
        self.rows = [json.loads(json.dumps(make_chunk(self.doc, start=a, end=b, method="fixed_size",
                      corpus_hash="d"*64, config_hash="e"*64, run_id="synthetic", chunk_index=i,
                      token_count=self.tokens.count(text[a:b]), tokenizer_id=self.tokens.tokenizer_id,
                      method_metadata={"content_token_count": self.tokens.count(text[a:b], special_tokens=False)}).to_dict()))
                     for i, (a, b) in enumerate(((0, 5), (5, 10), (10, 15)))]

    def cache(self, factory=FakeModel, *, identity=None, only=False):
        result = ChunkEmbeddings(self.root / "cache.sqlite3", identity or self.identity, self.tokens, factory, cache_only=only)
        self.addCleanup(result.close)
        return result

    def test_batches_deduplicate_preserve_order_and_query_policy(self):
        model = FakeModel()
        cache = self.cache(lambda: model)
        self.assertEqual(cache.embed_documents(["echo", "rule", "echo"]), [[1, 0], [0, 1], [1, 0]])
        self.assertEqual(model.calls, [["echo", "rule"]])
        self.assertEqual((cache.hits, cache.misses), (1, 2))
        self.assertEqual(cache.embed_query("rule"), [0, 1])
        self.assertEqual(len(model.calls), 1)

    def test_length_checks_before_any_inference_even_with_cache(self):
        model = FakeModel(); cache = self.cache(lambda: model)
        with self.assertRaisesRegex(ValueError, "truncation forbidden"):
            cache.embed_documents(["echo", "echo " * 9])
        self.assertEqual(model.calls, [])
        cache.embed_documents(["echo " * 8])  # Eight content tokens + two specials = limit.
        self.tokens.max_input_tokens = 9
        with self.assertRaises(ValueError): cache.embed_documents(["echo " * 8])
        for text in ("   ", "\uf0a1"):
            with self.assertRaises(ValueError): cache.embed_documents([text])

    def test_cache_identity_and_exact_whitespace(self):
        first = self.cache()
        first.embed_documents(["echo"])
        self.assertNotEqual(first.key("echo"), first.key("echo "))
        changed = self.cache(identity=self.identity | {"pooling": "different"}, only=True)
        self.assertNotEqual(first.key("echo"), changed.key("echo"))
        with self.assertRaisesRegex(ValueError, "cache miss"): changed.embed_documents(["echo"])

    def test_wrong_dimensions_nonfinite_zero_and_wrong_count_rejected(self):
        class Bad(FakeModel):
            def embed_documents(self, texts): return self.result
        for values in ([[1, 0, 0]], [[float("nan"), 1]], [[0, 0]], []):
            with self.subTest(values=values):
                model = Bad(); model.result = values
                cache = self.cache(lambda: model)
                with self.assertRaises(ValueError): cache.embed_documents(["echo"])
                self.assertEqual(cache.connection.execute("SELECT count(*) FROM vectors").fetchone()[0], 0)

    def test_cache_corruption_fails_without_reencoding(self):
        cache = self.cache(); cache.embed_documents(["echo"])
        cache.connection.execute("UPDATE vectors SET vector=?", (b"broken",)); cache.connection.commit()
        with self.assertRaisesRegex(ValueError, "checksum"): cache.embed_documents(["echo"])

    def test_interruption_keeps_committed_batches_and_order(self):
        identity = self.identity | {"embedding": {"batch_size": 1, "device": "cpu"}}
        first = self.cache(lambda: FakeModel(fail_at=1), identity=identity)
        variant = self.root / "variant"; variant.mkdir()
        with self.assertRaisesRegex(RuntimeError, "interruption"): encode_variant(variant, self.rows, first)
        self.assertEqual(first.connection.execute("SELECT count(*) FROM vectors").fetchone()[0], 1)
        resumed = self.cache(identity=identity)
        stats = encode_variant(variant, self.rows, resumed)
        self.assertEqual((stats["cache_hits"], stats["cache_misses"]), (2, 1))
        self.assertEqual(np.load(variant / "embeddings.npy").tolist(), [[1, 0], [0, 1], [1, 0]])
        mapping = [json.loads(s) for s in (variant / "embedding_rows.jsonl").read_text().splitlines()]
        self.assertEqual([r["chunk_id"] for r in mapping], [r["chunk_id"] for r in self.rows])
        self.assertEqual(len(set(r["chunk_id"] for r in mapping)), 3)
        cold = {name: (variant / name).read_bytes() for name in ("embeddings.npy", "embedding_rows.jsonl", "chunks.jsonl")}
        def never(): raise AssertionError("Warm repeat must not load a model")
        warm = self.cache(never, identity=identity, only=True)
        encode_variant(variant, self.rows, warm)
        self.assertTrue(all((variant / name).read_bytes() == data for name, data in cold.items()))

    def test_saved_mapping_reorder_and_vector_damage_detected(self):
        variant = self.root / "variant"; variant.mkdir()
        cache = self.cache(); encode_variant(variant, self.rows, cache)
        path = variant / "embedding_rows.jsonl"
        original = path.read_text(); rows = original.splitlines(); rows.reverse()
        path.write_text("\n".join(rows)+"\n")
        with self.assertRaisesRegex(ValueError, "order/identity"): validate_outputs(variant, self.rows, 2, cache.identity_hash)
        path.write_text(original)
        vectors = np.load(variant / "embeddings.npy"); vectors[0] = [0, 1]; np.save(variant / "embeddings.npy", vectors)
        with self.assertRaisesRegex(ValueError, "checksum"): validate_outputs(variant, self.rows, 2, cache.identity_hash)

    def test_source_validation_rejects_duplicate_stale_and_truncated_rows(self):
        class Corpus:
            documents = [self.doc]
            corpus_hash = "d"*64
        validate_rows(self.rows, Corpus, self.tokens, "fixed_size", "e"*64, 3)
        cases = []
        duplicate = deepcopy(self.rows); duplicate[1] = duplicate[0]; cases.append(duplicate)
        stale = deepcopy(self.rows); stale[0]["text"] = "rule "; cases.append(stale)
        stale_count = deepcopy(self.rows); stale_count[0]["token_count"] = 2; cases.append(stale_count)
        cases.append(self.rows[:-1])
        for rows in cases:
            with self.assertRaises(ValueError): validate_rows(rows, Corpus, self.tokens, "fixed_size", "e"*64, 3)

    def test_concurrent_run_lock_rejected(self):
        with exclusive_run(self.root):
            with self.assertRaisesRegex(ValueError, "Another embedding"):
                with exclusive_run(self.root): pass
        with exclusive_run(self.root): pass

    def test_stage_resume_preserves_completed_methods_and_rejects_changes(self):
        from rag_pipeline.embedding import encode_chunks as runner
        from rag_pipeline.common.run_manifest import write_json
        inputs = {method: deepcopy(self.rows) for method in runner.METHODS}
        for method, rows in inputs.items():
            for row in rows: row["method"] = method
        inputs["sliding_window"][0]["text"] = "cafe "
        config = SimpleNamespace(output_root="data/outputs/rag", corpus_dir="final_data")
        corpus = SimpleNamespace(corpus_hash="d"*64, snapshot={"files": {}})
        def create(root, config, run_id):
            run = root / config.output_root / run_id
            run.mkdir(parents=True)
            manifest = {"status": "running", "run_id": run_id}
            write_json(run / "stage_manifest.json", manifest)
            return run, manifest
        with ExitStack() as stack:
            mocks = {"ROOT": self.root, "load_config": lambda p: config, "load_tokenizer": lambda p: self.tokens,
                     "embedding_identity": lambda *args: self.identity, "load_corpus": lambda *args: corpus,
                     "load_inputs": lambda *args, **kwargs: (inputs, {}), "create_run": create, "environment": lambda root: {}}
            for name, value in mocks.items(): stack.enter_context(patch.object(runner, name, value))
            stack.enter_context(patch.object(runner.subprocess, "run", return_value=SimpleNamespace(returncode=0)))
            stack.enter_context(patch.object(runner.subprocess, "check_output", return_value=""))
            factory = stack.enter_context(patch.object(runner, "model_factory", return_value=lambda: FakeModel(fail_at=1)))
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                runner.run_embeddings("resume_test")
            run = self.root / config.output_root / "resume_test"
            first_manifest = run / "fixed_size/variant_001/stage_manifest.json"
            preserved = first_manifest.read_bytes()
            self.assertEqual(json.loads((run / "stage_manifest.json").read_text())["status"], "failed")
            changed = self.identity | {"precision": "different"}
            with patch.object(runner, "embedding_identity", return_value=changed), self.assertRaisesRegex(ValueError, "changed"):
                runner.run_embeddings("resume_test", resume=True)
            factory.return_value = FakeModel
            runner.run_embeddings("resume_test", resume=True)
            self.assertEqual(first_manifest.read_bytes(), preserved)
            manifest = json.loads((run / "stage_manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual([a["status"] for a in manifest["attempts"]], ["failed", "complete"])
            with self.assertRaisesRegex(ValueError, "immutable"):
                runner.run_embeddings("resume_test", resume=True)


if __name__ == "__main__":
    unittest.main()
