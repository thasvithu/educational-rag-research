"""Run sliding-window chunking through the shared artifact writer."""

from pathlib import Path

from ...common.chunk_run import run_token_chunker
from ...common.config import ROOT
from ...common.tokenization import load_tokenizer
from .chunker import SlidingWindowSplitter, load_settings


def run_sliding_window(run_id: str, *, config_path: Path = ROOT / "rag_pipeline/config.json",
                       method_config: Path = ROOT / "rag_pipeline/configs/sliding_window.json",
                       model_config: Path = ROOT / "rag_pipeline/configs/models.json",
                       sample: bool = False) -> Path:
    settings = load_settings(method_config)
    tokens = load_tokenizer(model_config)
    return run_token_chunker(run_id, config_path=config_path, model_config=model_config,
                             settings=settings, tokens=tokens,
                             splitter=SlidingWindowSplitter(tokens, settings["chunk_size"], settings["overlap"]),
                             sample=sample)
