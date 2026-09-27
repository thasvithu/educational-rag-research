"""Exact boundaries, heading uncertainty and structural fallback behavior."""

import json
from pathlib import Path
import tempfile
import unittest

from langchain_core.documents import Document

from rag_pipeline.chunking.structure_aware.chunker import StructureAwareSplitter, load_settings
from rag_pipeline.chunking.structure_aware.headings import resolve_headings
from rag_pipeline.chunking.structure_aware.units import sentence_units
from rag_pipeline.common.provenance import Document as Source, Heading, Page, eligible_regions
from rag_pipeline.tests import test_fixed_size as fixtures


def source(text, headings=()):
    pages = (Page(1, "1", 0, len(text), True, (), tuple(headings), False),)
    return Source("a"*64, "synthetic.pdf", "member", text, "b"*64, "unused", "c"*64, pages, eligible_regions(pages))


class HeadingTests(unittest.TestCase):
    def test_font_outline_and_prepared_hints_deduplicate_with_levels(self):
        text = "1 Introduction\nSome ordinary prose.\n1.1 Eligibility\nMore ordinary prose.\n"
        h = Heading("1 Introduction", 0, 14, None, "pdf_font_candidate", True)
        layout = {"pdf_toc": [[1, "1 Introduction", 1]], "pages": [{"page_number": 1,
            "heading_candidates": [{"text": "1 Introduction", "font_size": 20},
                                   {"text": "1.1 Eligibility", "font_size": 10},
                                   {"text": "Some ordinary prose.", "font_size": 10}], "detected_tables": []}]}
        headings, decisions, summary = resolve_headings(source(text, [h]), layout)
        self.assertEqual([x["level"] for x in headings], [1, 2])
        self.assertEqual(headings[0]["level_rule"], "pdf_outline")
        self.assertEqual(summary["status_counts"]["duplicate"], 2)
        self.assertIn("sentence_punctuation", summary["rejection_reasons"])

    def test_ambiguous_font_hints_and_false_headings_are_rejected(self):
        text = "TITLE\nTITLE\nThis is a bold sentence.\nCourse   Credits\n"
        layout = {"pdf_toc": [[1, "Absent title", 1]], "pages": [{"page_number": 1,
            "heading_candidates": [{"text": x, "font_size": 12} for x in text.splitlines()], "detected_tables": []}]}
        headings, _, summary = resolve_headings(source(text), layout)
        self.assertEqual(headings, [])
        self.assertIn("ambiguous_line", summary["rejection_reasons"])
        self.assertIn("column_like_line", summary["rejection_reasons"])

    def test_exact_prepared_offset_disambiguates_repeated_text(self):
        text = "TITLE\nTITLE\n"
        h = Heading("TITLE", 6, 11, None, "pdf_font_candidate", True)
        headings, _, _ = resolve_headings(source(text, [h]), {"pdf_toc": [], "pages": []})
        self.assertEqual(headings[0]["char_start"], 6)
        self.assertEqual(headings[0]["level_rule"], "unknown_level_as_sibling_2")

    def test_excluded_outline_and_partial_bold_line_never_become_heading(self):
        doc = source("Ordinary TITLE sentence\n", [Heading("TITLE", 9, 14, 2, "extractor_markdown", True)])
        headings, _, summary = resolve_headings(doc, {"pdf_toc": [[1, "Missing", 2]], "pages": []})
        self.assertFalse(headings)
        self.assertIn("no_complete_source_line", summary["rejection_reasons"])
        self.assertIn("excluded_blank_or_invalid_page", summary["rejection_reasons"])

    def test_numbered_bibliography_line_is_not_a_section(self):
        text = "2. Jane Smith, Internet and Web Programming, 2020\n"
        hint = Heading(text.strip(), 0, len(text)-1, 4, "extractor_markdown", True)
        headings, _, summary = resolve_headings(source(text, [hint]), {"pdf_toc": [], "pages": []})
        self.assertFalse(headings)
        self.assertIn("enumerated_line_without_heading_support", summary["rejection_reasons"])


class StructureTests(unittest.TestCase):
    def setUp(self):
        fixtures.FixedSizeTests.setUp(self)
        self.tokens.max_input_tokens = 100

    def check_partition(self, chunks, text, base=0, cap=20):
        self.assertEqual("".join(c.page_content for c in chunks), text)
        cursor = base
        for chunk in chunks:
            m = chunk.metadata
            self.assertEqual(m["char_start"], cursor)
            self.assertEqual(chunk.page_content, text[m["char_start"]-base:m["char_end"]-base])
            self.assertEqual(m["token_count"], self.tokens.count(chunk.page_content))
            self.assertLessEqual(m["method_metadata"]["content_token_count"], cap)
            cursor = m["char_end"]

    def test_hierarchy_cross_page_section_and_exact_text(self):
        text = "1 Rule\necho echo\n\n1.1 Student\nrule echo\n\n2 Other\necho rule\n"
        heads = []
        for title, level in [("1 Rule", 1), ("1.1 Student", 2), ("2 Other", 1)]:
            start = text.index(title)
            heads.append({"text": title, "char_start": 100+start, "char_end": 100+start+len(title), "level": level})
        metadata = {"char_start": 100, "char_end": 100+len(text), "resolved_headings": heads,
                    "pages": [{"page_number": 1, "char_start": 100, "char_end": 132, "quality_flags": []},
                              {"page_number": 2, "char_start": 132, "char_end": 100+len(text), "quality_flags": []}]}
        chunks = StructureAwareSplitter(self.tokens, 20).split_documents([Document(page_content=text, metadata=metadata)])
        self.check_partition(chunks, text, 100)
        self.assertEqual([c.metadata["section_path"] for c in chunks], [["1 Rule"], ["1 Rule", "1.1 Student"], ["2 Other"]])
        self.assertEqual(chunks[1].metadata["page_numbers"], [1, 2])

    def test_heading_path_resets_at_a_region_gap(self):
        headings = [{"text": "TITLE", "char_start": 0, "char_end": 5, "level": 1}]
        regions = [Document(page_content="TITLE\necho", metadata={"char_start": 0, "resolved_headings": headings}),
                   Document(page_content="rule echo", metadata={"char_start": 100, "resolved_headings": headings})]
        chunks = StructureAwareSplitter(self.tokens, 20).split_documents(regions)
        self.assertEqual(chunks[-1].metadata["section_path"], [])

    def test_adjacent_parent_child_headings_are_packed_together(self):
        text = "TITLE\nCHILD\necho rule\n"
        headings = [{"text": "TITLE", "char_start": 0, "char_end": 5, "level": 1},
                    {"text": "CHILD", "char_start": 6, "char_end": 11, "level": 2}]
        chunks = StructureAwareSplitter(self.tokens, 20).create_documents([text], [{"resolved_headings": headings}])
        self.check_partition(chunks, text)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["section_path"], ["TITLE", "CHILD"])

    def test_repeated_continuation_heading_keeps_section(self):
        text = "TITLE\necho\nTITLE\nrule\n"
        heads = [{"text": "TITLE", "char_start": i, "char_end": i+5, "level": 2} for i in (0, 11)]
        chunks = StructureAwareSplitter(self.tokens, 20).create_documents([text], [{"resolved_headings": heads}])
        self.check_partition(chunks, text)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["method_metadata"]["repeated_heading_continuations_in_region"], 1)

    def test_no_headings_paragraph_packing_unicode_and_tail(self):
        text = "  Café தமிழ்\n\necho rule\n\necho " * 3
        chunks = StructureAwareSplitter(self.tokens, 5).create_documents([text])
        self.check_partition(chunks, text, cap=5)
        self.assertTrue(all(c.metadata["method_metadata"]["missing_heading"] for c in chunks))

    def test_rows_and_list_items_kept_whole_when_they_fit(self):
        for text in ("\n\nstudent   80\nrule      80\necho      80\n",
                     "• echo rule\n  echo\n• student rule\n• echo echo\n"):
            chunks = StructureAwareSplitter(self.tokens, 5).create_documents([text])
            self.check_partition(chunks, text, cap=5)
            self.assertTrue(all(e["fallback"] == "table_rows" for c in chunks
                                for e in c.metadata["method_metadata"]["split_events"]))

    def test_fitting_table_moves_whole_to_next_chunk(self):
        text = "echo echo echo echo\n\nstudent   80\nrule   80\necho   80\n"
        chunks = StructureAwareSplitter(self.tokens, 7).create_documents([text])
        self.check_partition(chunks, text, cap=7)
        self.assertEqual(len(chunks), 2)
        self.assertIn("student   80\nrule   80\necho   80", chunks[1].page_content)

    def test_oversized_list_item_records_unavoidable_split(self):
        text = "• " + "echo " * 20
        chunks = StructureAwareSplitter(self.tokens, 5).create_documents([text])
        self.check_partition(chunks, text, cap=5)
        self.assertTrue(any(e["kind"] == "list_item" and e["fallback"] == "token"
                            for c in chunks for e in c.metadata["method_metadata"]["split_events"]))

    def test_sentence_fallback_preserves_decimal_and_abbreviation(self):
        text = "Dr. Smith gives 3.5 credits. Students study. Other rules apply."
        units = sentence_units(text)
        self.assertEqual(text[units[0].start:units[0].end], "Dr. Smith gives 3.5 credits. ")
        chunks = StructureAwareSplitter(self.tokens, 10).create_documents([text])
        self.check_partition(chunks, text, cap=10)
        self.assertTrue(any("sentence" in c.metadata["method_metadata"]["fallback_types"] for c in chunks))

    def test_tokenless_private_use_bullets_stay_with_neighboring_text(self):
        for text in ("\uf0a1\n\necho rule\n", "echo rule\n\n\uf0a1\n\necho\n", "echo\n\n\uf0a1\n"):
            chunks = StructureAwareSplitter(self.tokens, 5).create_documents([text])
            self.check_partition(chunks, text, cap=5)

    def test_empty_and_invalid_settings(self):
        splitter = StructureAwareSplitter(self.tokens, 20)
        self.assertEqual(splitter.split_text("\n\t"), [])
        with self.assertRaisesRegex(ValueError, "no encodable"):
            splitter.split_text("\x00")
        for size in (0, True, 99):
            with self.assertRaises(ValueError):
                StructureAwareSplitter(self.tokens, size)
        settings = json.loads(Path("rag_pipeline/configs/structure_aware.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for change in ({"overlap": 1}, {"unknown_level": 3}, {"chunk_size": True}, {"extra": 1}):
                path.write_text(json.dumps(settings | change))
                with self.assertRaises(ValueError):
                    load_settings(path)


if __name__ == "__main__":
    unittest.main()
