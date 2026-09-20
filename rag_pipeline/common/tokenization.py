"""Pinned Jina token counts and character offsets, without loading model weights."""

from dataclasses import dataclass
import json
from pathlib import Path

from transformers import BertTokenizerFast

from .config import ROOT, within
from .load_corpus import digest_json, file_hash, require


@dataclass
class ResearchTokenizer:
    tokenizer: object
    tokenizer_id: str
    max_input_tokens: int
    file_hashes: dict

    @property
    def special_tokens(self):
        return self.tokenizer.num_special_tokens_to_add(pair=False)

    def count(self, text: str, *, special_tokens: bool = True) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=special_tokens,
                                         truncation=False, verbose=False))

    def offsets(self, text: str) -> list[tuple[int, int]]:
        encoded = self.tokenizer(text, add_special_tokens=False, truncation=False,
                                 return_offsets_mapping=True, verbose=False)
        offsets = encoded["offset_mapping"]
        previous = -1
        for start, end in offsets:
            require(0 <= start < end <= len(text) and start >= previous,
                    "Tokenizer returned invalid/nonmonotonic character offsets")
            previous = start
        return offsets

    def identity(self):
        return {"tokenizer_id": self.tokenizer_id, "file_hashes": self.file_hashes,
                "max_input_tokens": self.max_input_tokens, "special_tokens": self.special_tokens}


def load_tokenizer(config_path: Path = ROOT / "rag_pipeline/configs/models.json") -> ResearchTokenizer:
    settings = json.loads(Path(config_path).read_text(encoding="utf-8"))
    require(settings.get("schema_version") == 1, "Unsupported model settings")
    settings = settings["embedding"]
    pins = json.loads((ROOT / "rag_pipeline/embedding/probe_bundle.json").read_text())
    model_id = "jinaai/jina-embeddings-v2-small-en"
    require(settings["model_id"] == model_id and settings["model_revision"] == pins["repos"][model_id],
            "Fixed-size experiment requires the selected pinned Jina tokenizer")
    limit = settings["max_input_tokens"]
    require(type(limit) is int and 3 <= limit <= 2048, "Input limit exceeds the tested configuration")
    bundle = within(ROOT, settings["bundle_dir"])
    hashes = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"):
        relative = "model/" + name
        hashes[name] = file_hash(within(bundle, relative))
        require(hashes[name] == pins["sha256"][relative], f"Tokenizer hash mismatch: {name}")
    tokenizer = BertTokenizerFast.from_pretrained(bundle / "model", local_files_only=True)
    require(tokenizer.is_fast, "Character offsets require a fast tokenizer")
    result = ResearchTokenizer(tokenizer, f"{model_id}@{settings['model_revision']}", limit, hashes)
    require(result.special_tokens == 2, "Unexpected Jina special-token count")
    return result


def experiment_config(chunk_settings: dict, tokenizer: ResearchTokenizer, gap_policy: str) -> dict:
    """Only settings that affect chunk records enter their configuration identity."""
    effective = {"chunking": chunk_settings, "tokenization": tokenizer.identity(), "gap_policy": gap_policy}
    return effective | {"config_hash": digest_json(effective)}
