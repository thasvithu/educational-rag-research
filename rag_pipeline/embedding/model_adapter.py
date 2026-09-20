"""Pinned local Jina model behind LangChain's Embeddings interface.

LangChain handles the common API. These helpers preserve our identical
content-only pooling and length checks for both early and late experiments.
"""

import importlib.util
import json
from pathlib import Path
import sys
import types

from langchain_core.embeddings import Embeddings

from ..common.config import ROOT, within
from ..common.load_corpus import file_hash, require


def verify_bundle(bundle: Path) -> dict:
    expected = json.loads((Path(__file__).with_name("probe_bundle.json")).read_text())
    for name, sha in expected["sha256"].items():
        require(file_hash(within(bundle, name)) == sha, f"Model bundle hash mismatch: {name}")
    return expected


def load_local_model(bundle, window=2048):
    """Import only the two pinned, hash-checked implementation files."""
    verify_bundle(bundle)
    import torch
    from transformers import BertTokenizerFast

    name = "_educational_rag_pinned_jina"
    package = types.ModuleType(name)
    package.__path__ = [str(bundle / "implementation")]
    sys.modules[name] = package
    loaded = {}
    for short in ("configuration_bert", "modeling_bert"):
        spec = importlib.util.spec_from_file_location(f"{name}.{short}", bundle / "implementation" / f"{short}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        loaded[short] = module
    config = loaded["configuration_bert"].JinaBertConfig.from_pretrained(bundle / "model", local_files_only=True)
    # ALiBi uses relative positions, not a learned position table. This controls
    # eager buffer allocation. Enforce the same cap before every forward pass.
    config.max_position_embeddings = window
    model, info = loaded["modeling_bert"].JinaBertModel.from_pretrained(
        bundle / "model", config=config, local_files_only=True,
        use_safetensors=True, dtype=torch.float32, output_loading_info=True)
    require(not info["missing_keys"] and not info["mismatched_keys"] and not info["error_msgs"],
            f"Incomplete model load: {info}")
    tokenizer = BertTokenizerFast.from_pretrained(bundle / "model", local_files_only=True)
    return model.cpu().eval(), tokenizer, info


def content_mean(hidden, encoded):
    """Same content-only pooling for queries, independent chunks, and late spans.

    This intentionally excludes CLS/SEP, unlike the stock all-attended-token
    pooling. Record it as an experimental adaptation, never as stock encode().
    """
    import torch
    mask = encoded["attention_mask"].bool() & ~encoded["special_tokens_mask"].bool()
    require(bool(mask.any(dim=1).all()), "Cannot pool text without content tokens")
    vectors = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
    return torch.nn.functional.normalize(vectors, p=2, dim=1)


class JinaEmbeddings(Embeddings):
    """Local Jina embeddings usable by LangChain retrievers and vector stores.

    Example: vectors = embeddings.embed_documents(["First passage", "Second passage"])
    Start with batch_size=1 on the 4 GB GPU. CUDA errors are reported, not silently
    retried on CPU; an explicit CPU run must have its device recorded separately.
    """

    def __init__(self, bundle: Path, *, device: str = "cuda", batch_size: int = 1,
                 max_input_tokens: int = 2048):
        import torch
        require(device in {"cpu", "cuda"}, "Device must be cpu or cuda")
        require(type(batch_size) is int and batch_size > 0, "Batch size must be positive")
        require(type(max_input_tokens) is int and 3 <= max_input_tokens <= 2048,
                "Input cap must be between 3 and the tested limit of 2048")
        if device == "cuda":
            require(torch.cuda.is_available(), "CUDA is unavailable in this environment. "
                    "Run in a GPU-enabled terminal, or explicitly select device='cpu'.")
        else:
            torch.set_num_threads(4)
        self.device = device
        self.batch_size = batch_size
        self.max_input_tokens = max_input_tokens
        model, self.tokenizer, self.loading_info = load_local_model(Path(bundle), max_input_tokens)
        self.model = model.to(device).eval()

    @classmethod
    def from_config(cls, path: Path = ROOT / "rag_pipeline/configs/models.json", *, device=None):
        settings = json.loads(Path(path).read_text(encoding="utf-8"))
        require(settings.get("schema_version") == 1, "Unsupported model configuration")
        settings = settings["embedding"]
        pinned = json.loads(Path(__file__).with_name("probe_bundle.json").read_text())
        require(settings["model_id"] == "jinaai/jina-embeddings-v2-small-en", "Unsupported embedding model")
        require(settings["model_revision"] == pinned["repos"][settings["model_id"]], "Wrong model revision")
        require(settings["implementation_revision"] == pinned["repos"]["jinaai/jina-bert-implementation"],
                "Wrong model implementation revision")
        require(settings["dtype"] == "float32" and settings["pooling"] == "content_mean_l2",
                "The tested adapter requires FP32 and content_mean_l2 pooling")
        return cls(within(ROOT, settings["bundle_dir"]), device=device or settings["device"],
                   batch_size=settings["batch_size"], max_input_tokens=settings["max_input_tokens"])

    def encode_tokens(self, texts: list[str]):
        """Return contextual token vectors and alignment data for a small batch.

        This is the low-level operation late chunking will reuse in Phase 8.
        It computes no chunk boundaries and performs no pooling itself.
        Offsets remain on CPU; model inputs and pooling masks use the selected device.
        """
        import torch
        require(isinstance(texts, list) and bool(texts) and
                all(isinstance(text, str) and text.strip() for text in texts),
                "Provide a nonempty list of nonblank text strings")
        encoded = self.tokenizer(texts, padding=True, truncation=False, return_tensors="pt",
                                 return_offsets_mapping=True, return_special_tokens_mask=True)
        require(encoded["input_ids"].shape[1] <= self.max_input_tokens,
                f"Input exceeds {self.max_input_tokens} tokens including special tokens; truncation forbidden")
        for key in ("input_ids", "attention_mask", "token_type_ids", "special_tokens_mask"):
            encoded[key] = encoded[key].to(self.device)
        with torch.inference_mode():
            hidden = self.model(**{key: encoded[key] for key in
                                  ("input_ids", "attention_mask", "token_type_ids")}).last_hidden_state
        return encoded, hidden

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Encode independent chunks, returning ordinary lists for LangChain."""
        import torch
        require(isinstance(texts, list), "texts must be a list")
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            encoded, hidden = self.encode_tokens(texts[start:start + self.batch_size])
            with torch.inference_mode():
                batch_vectors = content_mean(hidden, encoded)
            require(bool(torch.isfinite(batch_vectors).all()), "Embedding contains non-finite values")
            vectors.extend(batch_vectors.cpu().tolist())
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Use exactly the same model and pooling for a search question."""
        return self.embed_documents([text])[0]

