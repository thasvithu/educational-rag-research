"""Regression tests for evidence preservation, provenance and corpus generation."""

import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import pymupdf

SCRIPT = Path(__file__).resolve().parents[1] / "clean_pdfs.py"
spec = importlib.util.spec_from_file_location("clean_pdfs", SCRIPT)
cleaner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cleaner
spec.loader.exec_module(cleaner)
sys.path.insert(0, str(SCRIPT.parent))
from validate_corpus import validate


class TextPolicyTests(unittest.TestCase):
    def test_unicode_and_scientific_meaning_survive(self):
        text = "GPA ≥ 3.5; x² + H₂O; ∑ 𝑃; தமிழ் සිංහල; a I x\nco-operate"
        self.assertEqual(cleaner.clean_text(text), text)

    def test_safe_normalization_is_idempotent(self):
        text = "\ufeffﬁnal\u00a0data\x00  \r\n\r\n\r\n\uf0b7 item\nquali\u00ad\nfication"
        expected = "final data\n\n• item\nqualification"
        result = cleaner.clean_text(text)
        self.assertEqual(result, expected)
        self.assertEqual(cleaner.clean_text(result), result)

    def test_table_spacing_hard_hyphens_and_joiners_survive(self):
        text = "Code     Credits\nICT1234  3\nstate-of-the-art\nexam-\nination\nக\u200dக\u200cக"
        self.assertEqual(cleaner.clean_text(text), text)

    def test_missing_number_and_math_symbol_reject_extraction(self):
        config = cleaner.CleaningConfig()
        self.assertFalse(cleaner.acceptable(cleaner.preservation_metrics("GPA ≥ 3.5", "GPA 3.5"), config))
        self.assertFalse(cleaner.acceptable(cleaner.preservation_metrics("GPA ≥ 3.5", "GPA ≥ 3.0"), config))
        self.assertTrue(cleaner.acceptable(cleaner.preservation_metrics("GPA ≥ 3.5", "**GPA** ≥ 3.5"), config))

    def test_negation_loss_is_rejected_even_with_high_overall_coverage(self):
        reference = "A student is not eligible. " + "scholarship " * 1000
        candidate = reference.replace("not ", "")
        metrics = cleaner.preservation_metrics(reference, candidate)
        self.assertGreater(metrics["token_coverage"], 0.99)
        self.assertFalse(cleaner.acceptable(metrics, cleaner.CleaningConfig()))

    def test_page_spans_include_empty_pages_without_synthetic_headings(self):
        pages = [{"clean_text": "தமிழ்"}, {"clean_text": ""}, {"clean_text": "GPA ≥ 3.5"}]
        text, spans = cleaner.assemble_document(pages)
        for p, s in zip(pages, spans):
            self.assertEqual(text[s["char_start"]:s["char_end"]], p["clean_text"])
        self.assertNotIn("Page", text)


class MarginTests(unittest.TestCase):
    def test_only_repeated_margin_lines_and_edge_page_numbers_removed(self):
        snapshots = []
        for i in range(1, 6):
            snapshots.append({"page_number": i, "height": 800, "source_lines": [
                {"text": "Faculty handbook", "bbox": [10, 20, 200, 30]},
                {"text": "Required course", "bbox": [10, 200, 200, 220]},
                {"text": "3", "bbox": [10, 400, 20, 420]},
                {"text": str(i), "bbox": [290, 775, 305, 790]},
            ]})
        result = cleaner.find_margin_removals(snapshots, cleaner.CleaningConfig())
        self.assertEqual([r["text"] for r in result[1]], ["1"])
        self.assertEqual([r["text"] for r in result[2]], ["Faculty handbook", "2"])
        self.assertFalse(any(r["text"] == "Required course" for rows in result.values() for r in rows))

    def test_single_page_repeated_text_is_not_running_header(self):
        page = {"page_number": 1, "height": 800, "source_lines": [
            {"text": "Important rule", "bbox": [10, 20, 200, 30]}
        ] * 5}
        self.assertEqual(cleaner.find_margin_removals([page], cleaner.CleaningConfig()), {})


@unittest.skipUnless(shutil.which("pdftotext"), "Poppler required for integration tests")
class CorpusIntegrationTests(unittest.TestCase):
    def test_source_unchanged_offsets_deduplication_and_repeatability(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "data" / "member"
            source.mkdir(parents=True)
            pdf = source / "handbook.pdf"
            with pymupdf.open() as doc:
                for i in range(4):
                    page = doc.new_page(width=600, height=800)
                    page.insert_text((40, 35), "Faculty handbook", fontsize=10)
                    page.insert_text((40, 150), "Scholarship Requirements", fontsize=16)
                    page.insert_text((40, 200), "Students must maintain a GPA of 3.5. Mathematics is required.")
                    page.insert_text((295, 780), str(i + 1))
                doc.save(pdf)
            original_hash = cleaner.sha256_file(pdf)
            output = root / "final"
            row = cleaner.clean_pdf(pdf, output, root / "data")
            self.assertEqual(cleaner.sha256_file(pdf), original_hash)
            text = (output / row["text_path"]).read_text()
            detail = json.loads((output / row["metadata_path"]).read_text())
            for page in detail["pages"]:
                self.assertEqual(text[page["char_start"]:page["char_end"]], page["clean_text"])
                self.assertIn("3.5", page["clean_text"])
                self.assertIn("Scholarship", page["clean_text"])
            self.assertEqual(row["removed_margin_lines"], 7)
            row2 = cleaner.clean_pdf(pdf, output, root / "data")
            self.assertEqual(row["clean_sha256"], row2["clean_sha256"])
            duplicate = source / "duplicate.pdf"
            shutil.copyfile(pdf, duplicate)
            duplicate_row = cleaner.clean_pdf(duplicate, output, root / "data")
            rows = [row, duplicate_row]
            summary = cleaner.build_indexes(output, rows, cleaner.CleaningConfig(), root / "data")
            self.assertEqual(summary["documents_in_corpus"], 1)
            self.assertEqual(duplicate_row["duplicate_of"], row["source"])
            self.assertTrue(validate(output, root / "data")["passed"])
            with (output / row["text_path"]).open("a", encoding="utf-8") as stream:
                stream.write("Tampered output")
            self.assertFalse(validate(output, root / "data")["passed"])

    def test_blank_pdf_is_retained_for_audit_but_not_indexed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            member = root / "data" / "member"
            member.mkdir(parents=True)
            pdf = member / "blank.pdf"
            with pymupdf.open() as doc:
                doc.new_page()
                doc.new_page()
                doc.save(pdf)
            out = root / "final"
            row = cleaner.clean_pdf(pdf, out, root / "data")
            summary = cleaner.build_indexes(out, [row], cleaner.CleaningConfig(), root / "data")
            self.assertEqual(summary["documents_in_corpus"], 0)
            self.assertEqual(summary["pages_processed"], 2)
            self.assertEqual(row["exclusion_reason"], "no_extractable_text")


if __name__ == "__main__":
    unittest.main()
