"""Offline CPU/CUDA probe of the LangChain Jina adapter. Does not embed the corpus."""

import argparse
import json
import os
from pathlib import Path
import resource
import time

from ..common.config import ROOT, within
from ..common.load_corpus import require
from ..common.run_manifest import environment, write_json
from .model_adapter import JinaEmbeddings, content_mean, verify_bundle


def probe(bundle: Path, output: Path, device: str = "cpu"):
    require(not output.exists(), "Probe output already exists; choose a new output directory")
    output.mkdir(parents=True)
    write_json(output / "environment.json", environment(ROOT))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(42)
    if device == "cuda":
        require(torch.cuda.is_available(), "CUDA is unavailable in this environment")
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    embeddings = JinaEmbeddings(bundle, device=device)
    tokenizer, loading = embeddings.tokenizer, embeddings.loading_info
    if device == "cuda":
        torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started

    def encode(text):
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        data, hidden = embeddings.encode_tokens([text])
        with torch.inference_mode():
            vector = content_mean(hidden, data)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t
        require(bool(torch.isfinite(hidden).all()) and bool(torch.isfinite(vector).all()), "Non-finite output")
        return data, hidden, vector, elapsed

    rows = []
    for content_tokens in (128, 256, 512, 1024, 2046):
        text = "education " * content_tokens
        data, hidden, vector, seconds = encode(text)
        require(data["input_ids"].shape[1] == content_tokens + 2, "Unexpected special-token accounting")
        rows.append({"content_tokens": content_tokens, "input_tokens": hidden.shape[1],
                     "shape": list(hidden.shape), "seconds": seconds, "vector_norm": vector.norm().item()})
        print(json.dumps(rows[-1]), flush=True)

    text = "A credit represents study time. Students must attend at least 80% of classes."
    data, hidden, early_whole, _ = encode(text)
    offsets = data["offset_mapping"][0].tolist()
    indices = [i for i, (a, b) in enumerate(offsets) if b > a]
    pooled_whole = torch.nn.functional.normalize(hidden[0, indices].mean(0), dim=0)
    difference = (early_whole[0] - pooled_whole).abs().max().item()
    require(difference < 1e-6, "Single-window early/late pooling mismatch")
    start = text.index("Students")
    indices = [i for i, (a, b) in enumerate(offsets) if a >= start and b > a]
    contextual = torch.nn.functional.normalize(hidden[0, indices].mean(0), dim=0)
    _, _, independent, _ = encode(text[start:])
    unicode_text = "Café தமிழ் x²: 80% attendance."
    unicode_offsets = tokenizer(unicode_text, return_offsets_mapping=True)["offset_mapping"]
    require(all(0 <= a <= b <= len(unicode_text) for a, b in unicode_offsets), "Invalid Unicode character offsets")
    rejected = False
    try:
        encode("education " * 2047)
    except ValueError:
        rejected = True
    require(rejected, "Overflow was not rejected")
    # Exercise the actual LangChain public API, including unequal-length padding.
    examples = ["A credit represents study time.", "Students must attend at least 80% of classes to sit the examination."]
    independent_vectors = torch.tensor(embeddings.embed_documents(examples))
    embeddings.batch_size = 2
    batch_vectors = torch.tensor(embeddings.embed_documents(examples))
    query = torch.tensor(embeddings.embed_query(examples[0]))
    padding_difference = (batch_vectors - independent_vectors).abs().max().item()
    require(padding_difference < 1e-5, "Padding changed content embeddings beyond tolerance")
    require(torch.allclose(query, independent_vectors[0], atol=1e-6), "Query/document encoding differs")
    result = {"status": "passed", "scope": "LangChain adapter capability smoke test, not retrieval quality or full-corpus throughput",
              "bundle": verify_bundle(bundle), "loading_info": loading, "device": device, "dtype": "float32",
              "threads": 4, "seed": 42, "advertised_model_limit": 8192, "tested_input_cap": 2048,
              "config_override": {"max_position_embeddings": 2048}, "special_tokens": 2,
              "embedding_dimension": 512, "pooling": "mean_content_tokens_then_L2; identical query/early/late policy",
              "query_prompt": "", "document_prompt": "", "load_seconds": load_seconds,
              "timings": rows, "single_window_max_abs_difference": difference,
              "example_early_late_cosine": float(independent[0] @ contextual),
              "unicode_offsets": unicode_offsets, "unicode_text": unicode_text,
              "langchain_api_checks": {"embedding_count": len(batch_vectors), "padding_max_abs_difference": padding_difference,
                                       "query_document_match": True},
              "overflow_rejected": rejected, "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "paid_calls": 0}
    if device == "cuda":
        result["gpu"] = {"name": torch.cuda.get_device_name(0),
                         "total_memory_mib": torch.cuda.get_device_properties(0).total_memory / 1024**2,
                         "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                         "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2}
    write_json(output / "probe_result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    # Restrict generated artifacts to the ignored output tree.
    output = args.output.resolve()
    require(output.is_relative_to(ROOT / "data/outputs/rag"), "Output must be under data/outputs/rag")
    probe(args.bundle.resolve(), output, args.device)


if __name__ == "__main__":
    main()
