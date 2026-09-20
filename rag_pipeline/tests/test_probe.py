"""Small numerical checks; no downloads or model loading."""

from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None

from rag_pipeline.embedding.probe import content_mean, verify_bundle


class BundleTests(unittest.TestCase):
    def test_tampered_bundle_rejected_before_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model").mkdir()
            (root / "model/config.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_bundle(root)


@unittest.skipIf(torch is None, "Install requirements-phase2.txt for numerical pooling tests")
class PoolingTests(unittest.TestCase):
    def test_special_and_padded_values_cannot_influence_content_vector(self):
        hidden = torch.tensor([[[900., 900.], [1., 0.], [0., 1.], [-900., 900.], [900., -900.]]])
        masks = {"attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
                 "special_tokens_mask": torch.tensor([[1, 0, 0, 1, 0]])}
        result = content_mean(hidden, masks)
        torch.testing.assert_close(result, torch.tensor([[2**-0.5, 2**-0.5]]))

    def test_empty_content_rejected(self):
        with self.assertRaisesRegex(ValueError, "without content"):
            content_mean(torch.ones((1, 2, 3)), {"attention_mask": torch.ones((1, 2)),
                                               "special_tokens_mask": torch.ones((1, 2))})


if __name__ == "__main__":
    unittest.main()
