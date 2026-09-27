"""Run semantic boundary detection with the pinned local Jina model."""

from pathlib import Path
import subprocess

from ...common.chunk_run import run_token_chunker
from ...common.config import ROOT, load_config
from ...common.load_corpus import require
from ...common.tokenization import experiment_config, load_tokenizer
from .cache import CachedBoundaryEmbeddings, boundary_identity, model_factory
from .chunker import SemanticSplitter, load_settings


def run_semantic(run_id, *, config_path: Path = ROOT / "rag_pipeline/config.json",
                 method_config: Path = ROOT / "rag_pipeline/configs/semantic.json",
                 model_config: Path = ROOT / "rag_pipeline/configs/models.json",
                 sample=False, device=None, cache_only=False):
    settings = load_settings(method_config)
    tokens = load_tokenizer(model_config)
    identity = boundary_identity(model_config, device)
    settings["boundary_embedding"] = identity
    config = load_config(config_path)
    effective = experiment_config(settings, tokens, config.gap_policy)
    cache = ROOT / "data/outputs/rag/semantic_cache" / (effective["config_hash"] + ".sqlite3")
    relative = cache.relative_to(ROOT).as_posix()
    require(subprocess.run(["git", "check-ignore", "--no-index", "-q", relative], cwd=ROOT).returncode == 0, "Cache must be ignored")
    require(not subprocess.check_output(["git", "ls-files", "--", relative], cwd=ROOT, text=True).strip(), "Cache must not be tracked")
    factory = model_factory(model_config, identity["embedding"]["device"])
    if cache_only:
        def factory():
            raise ValueError("Boundary cache miss in --cache-only mode; rerun with the model enabled")
    embeddings = CachedBoundaryEmbeddings(cache, identity, effective["config_hash"], factory)
    splitter = SemanticSplitter(tokens, embeddings, chunk_size=settings["chunk_size"],
                                min_chunk_size=settings["min_chunk_size"], percentile=settings["percentile"])
    splitter.cache_path = relative
    splitter.cache_only = cache_only
    try:
        return run_token_chunker(run_id, config_path=config_path, model_config=model_config,
                                 settings=settings, splitter=splitter, tokens=tokens, sample=sample)
    finally:
        embeddings.close()
