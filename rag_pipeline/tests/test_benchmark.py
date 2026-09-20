import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rag_pipeline.common.load_corpus import Corpus, digest_bytes
from rag_pipeline.common.provenance import Document, Page, eligible_regions
from rag_pipeline.evaluation.validate_benchmark import annotation_hash, validate_directory, validate_records


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        text = "Rule தமிழ்.\n\n\n\nOther rule."
        pages = (Page(1, "i", 0, 11, True, (), (), False),
                 Page(2, "ii", 13, 13, False, (), (), True, "bad_font"),
                 Page(3, "iii", 15, len(text), True, (), (), False))
        doc = Document("a"*64, "member/synthetic.pdf", "member", text,
                       digest_bytes(text.encode()), "unused.json", "b"*64, pages, eligible_regions(pages))
        self.corpus = Corpus((doc,), "c"*64, {})
        self.manifest = {"schema_version": 1, "corpus_hash": self.corpus.corpus_hash,
                         "target_counts": {"development": 1, "test": 0}, "protocol_status": "approved",
                         "document_groups": {doc.document_id: "source_family"}}
        self.row = {"schema_version": 1, "question_id": "q1", "corpus_hash": self.corpus.corpus_hash,
                    "question": "What is the rule?", "reference_answer": "Rule தமிழ்.", "scope": "Synthetic example",
                    "group_id": "rule", "author": "draft_author", "language": "en", "category": "definition",
                    "answerability": "answerable", "split": "development", "rubric": ["State the rule"],
                    "evidence_units": [{"unit_id": "e1", "claim": "The rule", "alternatives": [{
                        "document_id": doc.document_id, "source": doc.source, "source_sha256": doc.document_id,
                        "clean_sha256": doc.clean_sha256, "char_start": 0, "char_end": 11,
                        "text": text[:11], "page_numbers": [1]}]}],
                    "review": {"status": "pending"}}

    def validate(self, rows=None):
        return validate_records(rows or [self.row], self.corpus, self.manifest)

    def approve(self):
        self.row["review"] = {"status": "approved", "reviewer": "independent_human", "reviewer_kind": "human",
                              "pdf_verified": True, "notes": "Synthetic test approval", "reviewed_content_hash": annotation_hash(self.row)}

    def test_valid_draft_is_not_freeze_ready(self):
        result = self.validate()
        self.assertTrue(result["valid"])
        self.assertFalse(result["freeze_ready"])

    def test_approval_gate_and_stale_review(self):
        self.approve()
        self.assertTrue(self.validate()["freeze_ready"])
        self.row["reference_answer"] = "Changed"
        self.assertIn("Review is stale", str(self.validate()["errors"]))

    def test_assistant_cannot_supply_human_approval(self):
        self.approve()
        self.row["review"]["reviewer_kind"] = "assistant"
        self.assertFalse(self.validate()["valid"])

    def test_wrong_evidence_text_page_and_hash(self):
        original = copy.deepcopy(self.row)
        for key, value in (("text", "Other"), ("page_numbers", [3]), ("clean_sha256", "0"*64)):
            with self.subTest(key=key):
                self.row = copy.deepcopy(original)
                self.row["evidence_units"][0]["alternatives"][0][key] = value
                self.assertFalse(self.validate()["valid"])

    def test_gap_crossing_rejected(self):
        span = self.row["evidence_units"][0]["alternatives"][0]
        span.update(char_end=20, text=self.corpus.documents[0].text[:20], page_numbers=[1, 3])
        self.assertIn("excluded/blank", str(self.validate()["errors"]))

    def test_source_family_split_leakage(self):
        other = copy.deepcopy(self.row)
        other.update(question_id="q2", group_id="different_question", split="test")
        self.assertIn("family leaks", str(self.validate([self.row, other])["errors"]))

    def test_exposed_pilot_cannot_become_test_after_removal(self):
        self.manifest["reserved_development_groups"] = ["source_family"]
        self.row["split"] = "test"
        self.assertIn("Exposed source", str(self.validate()["errors"]))

    def test_duplicate_question_rejected(self):
        self.assertIn("duplicate", str(self.validate([self.row, self.row])["errors"]))

    def test_answerable_needs_evidence_and_unanswerable_needs_rationale(self):
        self.row["evidence_units"] = []
        self.assertFalse(self.validate()["valid"])
        self.row["answerability"] = "unanswerable"
        self.assertFalse(self.validate()["valid"])
        self.row["rationale"] = "The synthetic corpus does not supply this fact."
        self.assertTrue(self.validate()["valid"])

    def test_protocol_and_target_gates(self):
        self.approve()
        self.manifest["protocol_status"] = "draft"
        self.manifest["target_counts"]["test"] = 80
        self.assertTrue(self.validate()["valid"])
        self.assertEqual(len(self.validate()["blockers"]), 2)

    def test_frozen_snapshot_detects_protocol_and_pdf_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            (data / "member").mkdir(parents=True)
            source = data / "member/synthetic.pdf"
            source.write_bytes(b"synthetic source identity; no PDF parsing in this test")
            sha = digest_bytes(source.read_bytes())
            doc = replace(self.corpus.documents[0], document_id=sha)
            corpus = replace(self.corpus, documents=(doc,))
            span = self.row["evidence_units"][0]["alternatives"][0]
            span.update(document_id=sha, source_sha256=sha)
            self.manifest.update(document_groups={sha: "source_family"}, files={"development": "development.jsonl"})
            self.approve()
            docs = root / "rag_pipeline/docs"
            docs.mkdir(parents=True)
            for name in ("experiment_protocol.md", "annotation_guide.md", "model_feasibility.md"):
                (docs / name).write_text("Synthetic protocol")
            benchmark = data / "benchmark"
            benchmark.mkdir()
            (benchmark / "benchmark_manifest.json").write_text(json.dumps(self.manifest))
            (benchmark / "development.jsonl").write_text(json.dumps(self.row) + "\n")
            with patch("rag_pipeline.evaluation.validate_benchmark.ROOT", root):
                result = validate_directory(benchmark, corpus)
                self.assertTrue(result["freeze_ready"], result)
                (benchmark / "frozen_snapshot.json").write_text(json.dumps(result))
                self.assertTrue(validate_directory(benchmark, corpus)["valid"])
                (docs / "experiment_protocol.md").write_text("Changed rules")
                self.assertIn("protocol_hashes", str(validate_directory(benchmark, corpus)["errors"]))
                source.write_bytes(b"Changed original source")
                self.assertIn("Original PDF changed", str(validate_directory(benchmark, corpus)["errors"]))


if __name__ == "__main__":
    unittest.main()
