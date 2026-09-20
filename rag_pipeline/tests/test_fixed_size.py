"""Fixed-size boundary tests with a tiny local WordPiece vocabulary."""

import json
from pathlib import Path
import tempfile
import unittest

from langchain_core.documents import Document
from transformers import BertTokenizerFast

from rag_pipeline.chunking.fixed_size.chunker import FixedSizeSplitter, load_settings
from rag_pipeline.common.chunk_schema import make_chunk
from rag_pipeline.common.provenance import Document as Source, Page, eligible_regions
from rag_pipeline.common.tokenization import ResearchTokenizer


class FixedSizeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        vocab = Path(temporary.name) / "vocab.txt"
        vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
                                   "echo", "rule", "cafe", ".", ",", "a", "b", "c", "d",
                                   "un", "##bel", "##iev", "##able", "student", "80", "%"]))
        self.tokens = ResearchTokenizer(BertTokenizerFast(vocab_file=str(vocab), do_lower_case=True),
                                        "synthetic-wordpiece-v1", 10, {})
        self.splitter = FixedSizeSplitter(self.tokens, 3)

    def assert_partition(self, text):
        spans = self.splitter.split_spans(text)
        self.assertEqual("".join(text[s.char_start:s.char_end] for s in spans), text)
        self.assertEqual(spans[0].char_start, 0)
        self.assertEqual(spans[-1].char_end, len(text))
        for left, right in zip(spans, spans[1:]):
            self.assertEqual(left.char_end, right.char_start)
        for span in spans:
            piece = text[span.char_start:span.char_end]
            self.assertLessEqual(self.tokens.count(piece, special_tokens=False), 3)
            self.assertEqual(span.token_count, self.tokens.count(piece))
        return spans

    def test_exact_windows_and_short_tail(self):
        spans = self.assert_partition("echo " * 8)
        self.assertEqual([s.content_token_count for s in spans], [3, 3, 2])

    def test_case_unicode_whitespace_and_control_glyphs_retained(self):
        for text in ("  Café\tதமிழ் x²\n\nECHO rule.  ", "echo\x00 rule cafe\n echo", "unbelievable echo rule"):
            with self.subTest(text=text):
                self.assert_partition(text)

    def test_whitespace_ownership(self):
        text = "  echo\t echo\n echo   rule  "
        pieces = self.splitter.split_text(text)
        self.assertEqual(pieces, ["  echo\t echo\n echo   ", "rule  "])

    def test_repeated_text_has_distinct_offsets(self):
        text = "echo " * 9
        chunks = self.splitter.create_documents([text], [{"char_start": 100, "char_end": 100 + len(text)}])
        self.assertEqual([c.metadata["char_start"] for c in chunks], [100, 115, 130])
        self.assertEqual([c.metadata["start_index"] for c in chunks], [0, 15, 30])
        self.assertEqual(chunks[0].page_content, chunks[1].page_content)

    def test_cross_page_mapping_updates_and_input_metadata_unchanged(self):
        metadata = {"char_start": 10, "char_end": 29, "pages": [
            {"page_number": 1, "char_start": 10, "char_end": 19, "quality_flags": ["first"]},
            {"page_number": 2, "char_start": 19, "char_end": 29, "quality_flags": ["second"]}]}
        chunks = self.splitter.split_documents([Document(page_content="echo echo echo rule", metadata=metadata)])
        self.assertEqual(chunks[0].metadata["page_numbers"], [1, 2])
        self.assertEqual(chunks[1].metadata["page_numbers"], [2])
        self.assertEqual(chunks[1].metadata["quality_flags"], ["second"])
        self.assertEqual(metadata["char_end"], 29)

    def test_separate_regions_remain_separate(self):
        documents = [Document(page_content="echo ", metadata={"char_start": 0, "char_end": 5}),
                     Document(page_content="rule", metadata={"char_start": 20, "char_end": 24})]
        chunks = self.splitter.transform_documents(documents)
        self.assertEqual([(d.metadata["char_start"], d.metadata["char_end"]) for d in chunks], [(0, 5), (20, 24)])

    def test_empty_and_tokenless_text(self):
        self.assertEqual(self.splitter.split_text(" \n\t"), [])
        with self.assertRaisesRegex(ValueError, "no encodable"):
            self.splitter.split_text("\x00")

    def test_invalid_budgets_and_mismatched_metadata(self):
        for budget in (0, -1, True, 9):
            with self.assertRaises(ValueError):
                FixedSizeSplitter(self.tokens, budget)
        with self.assertRaises(ValueError):
            self.splitter.create_documents(["echo"], [])
        with self.assertRaises(ValueError):
            self.splitter.create_documents(["echo"], [{"char_start": 2, "char_end": 99}])

    def test_stable_chunk_ids_across_execution_ids(self):
        text = "echo " * 8
        pages = (Page(1, "1", 0, len(text), True, (), (), False),)
        source = Source("a"*64, "synthetic.pdf", "member", text, "b"*64, "unused", "c"*64, pages, eligible_regions(pages))
        outputs = []
        for run_id in ("run_one", "run_two"):
            outputs.append([make_chunk(source, start=s.char_start, end=s.char_end, method="fixed_size",
                                       corpus_hash="d"*64, config_hash="e"*64, run_id=run_id, chunk_index=i,
                                       token_count=s.token_count, tokenizer_id=self.tokens.tokenizer_id).chunk_id
                            for i, s in enumerate(self.splitter.split_spans(text))])
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(len(outputs[0]), len(set(outputs[0])))

    def test_settings_reject_overlap_and_unknown_policy(self):
        settings = json.loads(Path("rag_pipeline/configs/fixed_size.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for key, value in (("overlap", 1), ("extra", True), ("chunk_size", True)):
                bad = settings | {key: value}
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    load_settings(path)


class OffsetEdgeTests(unittest.TestCase):
    def test_retokenization_overflow_backs_off_without_dropping_text(self):
        class Tokens:
            max_input_tokens, special_tokens = 10, 2
            def offsets(self, text):
                return [(i, i+1) for i in range(len(text))]
            def count(self, text, special_tokens=True):
                return (3 if text == "ab" else len(text)) + (2 if special_tokens else 0)
        splitter = FixedSizeSplitter(Tokens(), 2)
        self.assertEqual(splitter.split_text("abcd"), ["a", "bc", "d"])
        self.assertEqual(splitter.split_spans("abcd")[0].boundary_backoffs, 1)

    def test_subtokens_sharing_a_character_cannot_be_split(self):
        class Tokens:
            max_input_tokens, special_tokens = 10, 2
            def offsets(self, text):
                return [(i, i+1) for i in range(len(text)) for _ in range(2)]
            def count(self, text, special_tokens=True):
                return len(text)*2 + (2 if special_tokens else 0)
        self.assertEqual(FixedSizeSplitter(Tokens(), 3).split_text("éé"), ["é", "é"])
        with self.assertRaisesRegex(ValueError, "indivisible"):
            FixedSizeSplitter(Tokens(), 1).split_text("é")


if __name__ == "__main__":
    unittest.main()
