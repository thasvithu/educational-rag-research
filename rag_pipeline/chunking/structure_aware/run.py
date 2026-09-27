"""Run the structure-aware method through the shared research artifact writer."""

from pathlib import Path

from ...common.chunk_run import run_token_chunker
from ...common.config import ROOT
from ...common.tokenization import load_tokenizer
from .chunker import StructureAwareSplitter, load_settings


def run_structure_aware(run_id: str, *, config_path: Path = ROOT / "rag_pipeline/config.json",
                        method_config: Path = ROOT / "rag_pipeline/configs/structure_aware.json",
                        model_config: Path = ROOT / "rag_pipeline/configs/models.json", sample=False):
    settings = load_settings(method_config)
    tokens = load_tokenizer(model_config)
    return run_token_chunker(run_id, config_path=config_path, model_config=model_config,
                             settings=settings, tokens=tokens,
                             splitter=StructureAwareSplitter(tokens, settings["chunk_size"]), sample=sample)
