"""Semantic decisions and cache integrity, using controlled vectors without a GPU."""

import json
from pathlib import Path
import tempfile
import unittest

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_pipeline.chunking.semantic.cache import CachedBoundaryEmbeddings
from rag_pipeline.chunking.semantic.chunker import SemanticSplitter, load_settings
from rag_pipeline.chunking.semantic.units import segment_units
from rag_pipeline.tests import test_fixed_size as fixtures


class TopicEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[0.0, 1.0] if "Student" in text else [1.0, 0.0] for text in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class SemanticTests(unittest.TestCase):
    def setUp(self):
        fixtures.FixedSizeTests.setUp(self)
        self.tokens.max_input_tokens = 100

    def semantic_splitter(self, cap=20, minimum=1, percentile=50):
        return SemanticSplitter(self.tokens, TopicEmbeddings(), chunk_size=cap, min_chunk_size=minimum, percentile=percentile)

    def check(self, splitter, text):
        chunks = splitter.create_documents([text], [{"char_start": 100, "char_end": 100+len(text)}])
        self.assertEqual("".join(c.page_content for c in chunks), text)
        cursor = 100
        for chunk in chunks:
            self.assertEqual(chunk.metadata["char_start"], cursor)
            self.assertEqual(chunk.page_content, text[cursor-100:chunk.metadata["char_end"]-100])
            self.assertEqual(chunk.metadata["token_count"], self.tokens.count(chunk.page_content))
            self.assertLessEqual(chunk.metadata["method_metadata"]["content_token_count"], splitter._chunk_size)
            cursor = chunk.metadata["char_end"]
        return chunks, splitter.last_diagnostics[0]

    def test_known_topic_change_selected_related_sentences_retained(self):
        chunks, report = self.check(self.semantic_splitter(), "Echo echo. Echo echo. Student rule. Student rule.")
        self.assertEqual(len(chunks), 2)
        self.assertEqual([b["cosine_distance"] for b in report["boundaries"]], [0, 1, 0])
        self.assertEqual([b["selected"] for b in report["boundaries"]], [False, True, False])
        self.assertEqual(chunks[0].metadata["method_metadata"]["end_reason"], "semantic")

    def test_minimum_suppresses_topic_cut(self):
        chunks, report = self.check(self.semantic_splitter(minimum=9), "Echo echo. Echo echo. Student rule. Student rule.")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(report["boundaries"][1]["decision"], "minimum_size_suppressed")

    def test_hard_cap_overrides_minimum_and_similarity(self):
        chunks, report = self.check(self.semantic_splitter(cap=5, minimum=4), "Echo echo. Echo echo. Echo echo.")
        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(c.metadata["method_metadata"]["below_minimum_target"] for c in chunks))
        self.assertEqual([b["decision"] for b in report["boundaries"]], ["max_tokens", "max_tokens"])

    def test_equal_threshold_ties_do_not_manufacture_topic_changes(self):
        chunks, report = self.check(self.semantic_splitter(), "Echo. Echo. Echo. Echo.")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(report["threshold"], 0)
        self.assertFalse(any(b["candidate"] for b in report["boundaries"]))

    def test_short_regions_skip_embedding_but_still_obey_cap(self):
        class NeverEmbed(TopicEmbeddings):
            def embed_documents(self, texts):
                raise AssertionError("Short region should not be embedded")
        splitter = SemanticSplitter(self.tokens, NeverEmbed(), chunk_size=5, min_chunk_size=4)
        chunks, report = self.check(splitter, "Echo echo. Student rule.")
        self.assertEqual(len(chunks), 2)
        self.assertIsNone(report["threshold"])

    def test_long_units_are_split_before_embedding_without_losing_tail(self):
        chunks, report = self.check(self.semantic_splitter(cap=5, minimum=2), "echo " * 22)
        self.assertTrue(all(u["forced_token_split"] for u in report["units"]))
        self.assertEqual(chunks[-1].metadata["method_metadata"]["content_token_count"], 2)

    def test_unicode_repeated_text_and_tokenless_bullets_preserved(self):
        text = "\uf0a1\n\nCafé தமிழ். Echo echo.\n\n\uf0a1\n\nEcho echo.\n"
        chunks, _ = self.check(self.semantic_splitter(cap=5, minimum=2), text)
        self.assertEqual(len({c.metadata["char_start"] for c in chunks}), len(chunks))

    def test_abbreviations_decimals_course_codes_lists_and_tables(self):
        text = "Dr. Smith gives SCS 1301 3.5 credits. Students study.\n\n• Echo rule\n  echo\n• Student rule\n\nCourse   Credits\nSCS1301   3.5\nSCS1302   2\n"
        units = segment_units(text, self.tokens, 40)
        self.assertEqual("".join(text[u.start:u.end] for u in units), text)
        self.assertEqual(text[units[0].start:units[0].end], "Dr. Smith gives SCS 1301 3.5 credits. ")
        self.assertIn("list_item", {u.kind for u in units})
        self.assertIn("table_row", {u.kind for u in units})

    def test_page_metadata_and_region_gaps(self):
        metadata = {"char_start": 0, "char_end": 17, "pages": [
            {"page_number": 1, "char_start": 0, "char_end": 5, "quality_flags": []},
            {"page_number": 2, "char_start": 5, "char_end": 17, "quality_flags": ["flag"]}]}
        chunks = self.semantic_splitter().split_documents([Document(page_content="Echo. Echo. Echo.", metadata=metadata),
                                                 Document(page_content="Student.", metadata={"char_start": 100})])
        self.assertEqual(chunks[0].metadata["page_numbers"], [1, 2])
        self.assertEqual(chunks[1].metadata["char_start"], 100)
        self.assertEqual(metadata["char_end"], 17)

    def test_settings_and_invalid_vectors_fail(self):
        settings = json.loads(Path("rag_pipeline/configs/semantic.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for change in ({"percentile": 100}, {"min_chunk_size": 300}, {"overlap": 1}, {"chunk_size": True}):
                path.write_text(json.dumps(settings | change))
                with self.assertRaises(ValueError): load_settings(path)
        class BadVectors(TopicEmbeddings):
            def embed_documents(self, texts): return [[0, 0]] * len(texts)
        with self.assertRaisesRegex(ValueError, "Zero boundary"):
            SemanticSplitter(self.tokens, BadVectors(), chunk_size=20, min_chunk_size=1).split_text("Echo. Echo. Echo.")


class CacheTests(unittest.TestCase):
    def test_persistent_cache_keys_and_corruption(self):
        identity = {"dimension": 2, "embedding": {"device": "cpu"}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.sqlite3"
            cache = CachedBoundaryEmbeddings(path, identity, "config-a", TopicEmbeddings)
            expected = cache.embed_documents(["Echo.", "Student.", "Echo."])
            self.assertEqual((cache.hits, cache.misses), (1, 2))
            first_key = cache.key("Echo.")
            self.assertNotEqual(first_key, cache.key(" Echo."))
            cache.close()
            def never(): raise AssertionError("A warm cache should not load the model")
            cache = CachedBoundaryEmbeddings(path, identity, "config-a", never)
            self.assertEqual(cache.embed_documents(["Echo.", "Student.", "Echo."]), expected)
            self.assertEqual((cache.hits, cache.misses), (3, 0))
            cache.connection.execute("UPDATE vectors SET vector = ? WHERE key = ?", (b"corrupt", first_key))
            cache.connection.commit()
            with self.assertRaisesRegex(ValueError, "checksum"):
                cache.embed_documents(["Echo."])
            cache.close()
            changed = CachedBoundaryEmbeddings(path, identity, "config-b", TopicEmbeddings)
            self.assertNotEqual(first_key, changed.key("Echo."))
            changed.embed_documents(["Echo."])
            self.assertEqual(changed.misses, 1)
            changed.close()


if __name__ == "__main__":
    unittest.main()
