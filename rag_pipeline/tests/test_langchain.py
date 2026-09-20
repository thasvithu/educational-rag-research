"""Interface checks using synthetic source text; no model download or API call."""

from dataclasses import asdict
import json
import os
import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_pipeline.common.langchain_loader import PreparedCorpusLoader
from rag_pipeline.embedding.model_adapter import JinaEmbeddings
from rag_pipeline.generation.model_adapter import create_chat_model
from rag_pipeline.tests import test_foundation


class LangChainLoaderTests(unittest.TestCase):
    def setUp(self):
        # Reuse the existing synthetic PDF-metadata fixture: Unicode, repeated
        # text, a blank page, and an excluded page. No original PDFs are needed.
        self.fixture = test_foundation.FoundationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.config_path = self.fixture.root / "config.json"
        self.config_path.write_text(json.dumps(asdict(self.fixture.config)))
        self.loader = PreparedCorpusLoader(self.config_path, root=self.fixture.root)

    def test_load_returns_exact_regions_without_bridging_gaps(self):
        documents = self.loader.load()
        source = self.loader.corpus.documents[0]
        self.assertEqual(len(documents), 3)
        self.assertEqual([d.metadata["page_numbers"] for d in documents], [[1, 2], [4], [6]])
        for document in documents:
            self.assertIsInstance(document, Document)
            start, end = document.metadata["char_start"], document.metadata["char_end"]
            self.assertEqual(document.page_content, source.text[start:end])
            self.assertNotIn("NEVER INDEX", document.page_content)
            self.assertEqual(document.metadata["corpus_hash"], self.loader.corpus.corpus_hash)

    def test_metadata_is_serializable_and_does_not_mutate_source(self):
        document = self.loader.load()[0]
        json.dumps(document.metadata)
        self.assertEqual(document.metadata["pages"][0]["headings"][0]["char_start"], 0)
        document.metadata["pages"][0]["headings"][0]["text"] = "Changed"
        self.assertEqual(self.loader.corpus.documents[0].pages[0].headings[0].text, "Title")

    def test_reloading_checks_for_changed_inputs(self):
        self.loader.load()
        (self.fixture.corpus_dir / "member/example.txt").write_text("Changed")
        with self.assertRaises(ValueError):
            self.loader.load()

    def test_generic_splitter_cannot_keep_stale_region_offsets(self):
        with self.assertRaisesRegex(ValueError, "exact source offsets"):
            self.loader.load_and_split()


class LangChainModelTests(unittest.TestCase):
    def test_cuda_unavailable_does_not_silently_switch_devices(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "CUDA is unavailable"):
                JinaEmbeddings("unused_bundle", device="cuda")

    def test_overlong_input_fails_before_model_inference(self):
        import torch
        adapter = object.__new__(JinaEmbeddings)
        self.assertIsInstance(adapter, Embeddings)
        adapter.device = "cpu"
        adapter.max_input_tokens = 8
        adapter.tokenizer = Mock(return_value={"input_ids": torch.ones((1, 9), dtype=torch.long)})
        adapter.model = Mock()
        with self.assertRaisesRegex(ValueError, "truncation forbidden"):
            adapter.encode_tokens(["A long input"])
        adapter.model.assert_not_called()
        self.assertIs(adapter.tokenizer.call_args.kwargs["truncation"], False)

    def test_groq_requires_an_explicit_model(self):
        with self.assertRaisesRegex(ValueError, "Choose an available"):
            create_chat_model()

    def test_groq_client_construction_is_offline(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "offline-test-key"}):
            with patch("httpx.Client.send", side_effect=AssertionError("Network forbidden")):
                client = create_chat_model("offline-model-fixture")
                self.assertEqual(client.model_name, "offline-model-fixture")
                self.assertEqual(client.temperature, 1e-8)
                self.assertEqual(client.max_tokens, 512)


if __name__ == "__main__":
    unittest.main()
