"""Sliding windows must add new evidence while preserving exact source spans."""

import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from rag_pipeline.chunking.fixed_size.chunker import FixedSizeSplitter
from rag_pipeline.chunking.sliding_window.chunker import SlidingWindowSplitter, load_settings
from rag_pipeline.common.chunk_schema import make_chunk
from rag_pipeline.common.chunk_run import run_token_chunker
from rag_pipeline.common.load_corpus import file_hash
from rag_pipeline.tests import test_fixed_size as fixed_fixtures
from rag_pipeline.tests import test_foundation as foundation_fixtures


class SlidingWindowTests(unittest.TestCase):
    def setUp(self):
        fixed_fixtures.FixedSizeTests.setUp(self)

    def assert_coverage(self, text, size, overlap):
        splitter = SlidingWindowSplitter(self.tokens, size, overlap)
        spans = splitter.split_spans(text)
        covered, previous = 0, None
        reconstructed = ""
        for span in spans:
            self.assertLessEqual(span.char_start, covered)
            self.assertGreater(span.char_end, covered)
            if previous:
                self.assertGreater(span.char_start, previous.char_start)
                shared = previous.region_token_end - span.region_token_start
                self.assertTrue(0 <= shared <= overlap)
            reconstructed += text[covered:span.char_end]
            piece = text[span.char_start:span.char_end]
            self.assertEqual(span.token_count, self.tokens.count(piece))
            self.assertTrue(0 < span.content_token_count <= size)
            covered, previous = span.char_end, span
        self.assertEqual(covered, len(text))
        self.assertEqual(reconstructed, text)
        return spans

    def test_regular_stride_and_no_redundant_tail(self):
        spans = self.assert_coverage("echo " * 8, 4, 2)
        self.assertEqual([(s.region_token_start, s.region_token_end) for s in spans], [(0, 4), (2, 6), (4, 8)])
        self.assertEqual([s.content_token_count for s in self.assert_coverage("echo " * 7, 4, 2)], [4, 4, 3])

    def test_short_and_exact_length_inputs_produce_one_chunk(self):
        for length in (1, 3, 4):
            self.assertEqual(len(self.assert_coverage("echo " * length, 4, 3)), 1)

    def test_zero_overlap_matches_fixed_including_unicode(self):
        for text in ("echo " * 17, "  Café\tதமிழ் x²\n ECHO rule.  ", "unbelievable " * 5):
            for size in (2, 3, 8):
                self.assertEqual(SlidingWindowSplitter(self.tokens, size, 0).split_spans(text),
                                 FixedSizeSplitter(self.tokens, size).split_spans(text))

    def test_all_valid_small_overlaps_cover_mixed_text(self):
        rng = random.Random(42)
        text = "  " + " ".join(rng.choices(["echo", "unbelievable", "Café", "தமிழ்", "80%", "x²", "rule\n"], k=45)) + "\t "
        for size in range(2, 9):
            for overlap in range(size):
                with self.subTest(size=size, overlap=overlap):
                    self.assert_coverage(text, size, overlap)

    def test_metadata_marks_shared_text_and_real_source_positions(self):
        text = "echo " * 8
        chunks = SlidingWindowSplitter(self.tokens, 4, 2).create_documents(
            [text], [{"char_start": 100, "char_end": 140}])
        self.assertEqual([c.metadata["char_start"] for c in chunks], [100, 110, 120])
        self.assertEqual(chunks[0].page_content, chunks[1].page_content)
        for i, chunk in enumerate(chunks):
            meta = chunk.metadata["method_metadata"]
            self.assertEqual(meta["overlap_with_previous_region_tokens"], 2 if i else 0)
            self.assertEqual(meta["overlap_with_previous_characters"], 10 if i else 0)
            self.assertEqual(meta["nominal_stride_tokens"], 2)

    def test_regions_and_page_references_remain_correct(self):
        metadata = {"char_start": 10, "char_end": 30, "pages": [
            {"page_number": 1, "char_start": 10, "char_end": 20, "quality_flags": ["a"]},
            {"page_number": 2, "char_start": 20, "char_end": 30, "quality_flags": ["b"]}]}
        regions = [Document(page_content="echo " * 4, metadata=metadata),
                   Document(page_content="rule", metadata={"char_start": 80, "char_end": 84})]
        chunks = SlidingWindowSplitter(self.tokens, 3, 1).transform_documents(regions)
        self.assertEqual(chunks[0].metadata["page_numbers"], [1, 2])
        self.assertEqual(chunks[1].metadata["page_numbers"], [2])
        self.assertEqual(chunks[-1].metadata["char_start"], 80)
        self.assertEqual(chunks[-1].metadata["method_metadata"]["overlap_with_previous_region_tokens"], 0)
        self.assertEqual(metadata["char_end"], 30)

    def test_invalid_settings_and_empty_inputs(self):
        for overlap in (-1, 4, 5, True, 1.5):
            with self.assertRaises(ValueError):
                SlidingWindowSplitter(self.tokens, 4, overlap)
        splitter = SlidingWindowSplitter(self.tokens, 4, 1)
        self.assertEqual(splitter.split_text(" \n\t"), [])
        with self.assertRaisesRegex(ValueError, "no encodable"):
            splitter.split_text("\x00")
        settings = json.loads(Path("rag_pipeline/configs/sliding_window.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            for change in ({"overlap": 256}, {"overlap": True}, {"chunk_size": 0}, {"other": 1}):
                path.write_text(json.dumps(settings | change))
                with self.assertRaises(ValueError):
                    load_settings(path)

    def test_zero_overlap_ids_still_identify_method(self):
        fixture = foundation_fixtures.FoundationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        corpus = fixture.load()
        doc = corpus.documents[0]
        region = doc.regions[0]
        args = dict(start=region.char_start, end=region.char_end, corpus_hash=corpus.corpus_hash,
                    config_hash="f" * 64, run_id="test", chunk_index=0)
        fixed = make_chunk(doc, method="fixed_size", **args)
        sliding = make_chunk(doc, method="sliding_window", **args)
        self.assertEqual(fixed.text, sliding.text)
        self.assertNotEqual(fixed.chunk_id, sliding.chunk_id)


class SlidingEdgeTests(unittest.TestCase):
    def test_shared_character_groups_reduce_overlap_safely(self):
        class Tokens:
            max_input_tokens, special_tokens = 10, 2
            def offsets(self, text):
                return [(i, i + 1) for i in range(len(text)) for _ in range(2)]
            def count(self, text, special_tokens=True):
                return len(text) * 2 + (2 if special_tokens else 0)
        spans = SlidingWindowSplitter(Tokens(), 3, 2).split_spans("ééé")
        self.assertEqual([(s.char_start, s.char_end) for s in spans], [(0, 1), (1, 2), (2, 3)])

    def test_retokenization_cannot_produce_contained_or_stuck_window(self):
        class Tokens:
            max_input_tokens, special_tokens = 10, 2
            def offsets(self, text):
                return [(i, i + 1) for i in range(len(text))]
            def count(self, text, special_tokens=True):
                return (3 if text == "bc" else len(text)) + (2 if special_tokens else 0)
        spans = SlidingWindowSplitter(Tokens(), 2, 1).split_spans("abcd")
        self.assertEqual([(s.char_start, s.char_end) for s in spans], [(0, 2), (2, 4)])


class ArtifactTests(unittest.TestCase):
    def test_failed_run_keeps_manifest_and_cannot_be_overwritten(self):
        fixture = foundation_fixtures.FoundationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        import subprocess
        subprocess.run(["git", "init", "-q", str(fixture.root)], check=True)
        (fixture.root / ".gitignore").write_text("/data/\n")
        config_path = fixture.root / "config.json"
        config_path.write_text(json.dumps(fixture.config.to_dict()))
        model_path = fixture.root / "model.json"
        model_path.write_text("{}")
        settings = load_settings(Path("rag_pipeline/configs/sliding_window.json"))
        class Tokens:
            def identity(self): return {}
        with patch("rag_pipeline.common.chunk_run.ROOT", fixture.root), \
             patch("rag_pipeline.common.chunk_run.environment", return_value={}), \
             patch("rag_pipeline.common.chunk_run.PreparedCorpusLoader", side_effect=ValueError("test extraction failure")):
            args = dict(config_path=config_path, model_config=model_path, settings=settings, splitter=None, tokens=Tokens())
            with self.assertRaisesRegex(ValueError, "test extraction"):
                run_token_chunker("failed", **args)
            run = fixture.root / "data/outputs/rag/failed"
            manifest = json.loads((run / "stage_manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            nested = "sliding_window/variant_001/stage_manifest.json"
            self.assertEqual(manifest["artifact_sha256"][nested], file_hash(run / nested))
            with self.assertRaises(FileExistsError):
                run_token_chunker("failed", **args)


if __name__ == "__main__":
    unittest.main()
