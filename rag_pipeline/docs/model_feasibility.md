# Local embedding feasibility — 20 September 2026

## Decision

Use **Jina embeddings v2 small English for the five-method study**, as selected
by Vithusan. Local CPU and CUDA capability probes passed. Retrieval quality is still unmeasured;
confirm it with reviewed development questions before freezing configurations.
This phase did not build corpus embeddings, indexes, or any of the five chunkers.

**GPU follow-up:** outside the sandbox, NVIDIA reports an RTX 3050 Laptop GPU
with 4,096 MiB VRAM and PyTorch successfully runs a tiny CUDA calculation. The
original CPU-only probe remains valid, but did not establish GPU unavailability.
The subsequent LangChain CUDA model probe also passed at 2,048 tokens: about
667 MiB peak allocated memory, 798 MiB reserved, batch size 1 and FP32.

The official model card describes a small English encoder with ALiBi and an
8,192-token limit; it requires Jina's implementation and recommends mean pooling
and normalization. [Official model card](https://huggingface.co/jinaai/jina-embeddings-v2-small-en)
Late chunking computes contextual token representations before pooling chunk spans.
[Late chunking paper](https://arxiv.org/abs/2409.04701)

| Item | Tested configuration |
|---|---|
| Model | `jinaai/jina-embeddings-v2-small-en` |
| Model/tokenizer revision | `44e7d1d6caec8c883c2d4b207588504d519788d0` |
| Custom implementation | `jinaai/jina-bert-implementation` |
| Implementation revision | `f3ec4cf7de7e561007f27c9efc7148b0bd713f81` |
| Weights | Safetensors, 65,388,544 bytes; SHA256 in `embedding/probe_bundle.json` |
| Runtime | Python 3.11.16, torch 2.5.1, transformers 4.57.6; CPU FP32, four threads |
| Vector dimension | 512 |
| Tested maximum | 2,048 total tokens = 2,046 content + CLS/SEP |
| Prompts | None for query or document |
| Experiment pooling | Content-token mean, then L2 normalization; exclude special tokens and padding |
| Paid calls | Zero |

## Pooling and memory choices

The stock implementation's mean pooling uses all attended tokens, including
special tokens. The research probe deliberately applies **content-only mean
pooling to both early and late paths**. This preserves the planned exclusion of
special tokens while keeping the comparison matched. Queries must use the same
adaptation. Do not describe its vectors as identical to stock `model.encode()`.
No projection is applied; use `last_hidden_state`, not BERT's separate pooler output.

The pinned encoder eagerly constructs its ALiBi tensor using
`max_position_embeddings`. The probe overrides this allocation size to 2,048,
and separately rejects longer inputs before inference. It changes no learned
weights. Later adapters must retain the explicit guard: the implementation can
otherwise rebuild a larger buffer dynamically. All five methods share this runtime
configuration. Longer handbooks require eligible-region context windows.

The two downloaded custom Python files had their imports and loading/pooling/
ALiBi paths inspected. Their hashes and repository revisions are pinned; loading
is local and offline after download. This is not a claim of a comprehensive security
audit. No unpinned `trust_remote_code` import is used by the probe.

## Initial CPU measurements

Output: `data/outputs/rag/phase2_probe_20260920/probe_result.json`.

| Total input tokens | Forward + content pooling seconds |
|---|---:|
| 130 | 0.048 |
| 258 | 0.040 |
| 514 | 0.094 |
| 1,026 | 0.305 |
| 2,048 | 1.095 |

These are single CPU forwards on repeated-token synthetic inputs, not averaged
benchmarks, full-corpus estimates, or a comparison of retrieval quality. Model
loading took 7.16 seconds. Process peak RSS was approximately **1,402 MiB**, including
Python/imports and temporary allocations. Peak RSS is not model-only memory.

- No missing, unexpected, mismatched, or error loading keys.
- Finite normalized vectors and token arrays of shape `[1, input_tokens, 512]`.
- Exact-window early/late pooling maximum absolute difference: **0.0**.
- Contextual versus independent encoding of a sentence produced cosine **0.9449**;
  this demonstrates different representations, not better retrieval.
- Fast-tokenizer character offsets available, including a Unicode fixture. This
  does not establish Tamil embedding quality or complete tokenizer coverage of every glyph.
- A 2,049-token input was rejected before inference; no truncation.
- The advertised 8,192-token limit, production-scale batching, window seams, full-corpus
  speed and retrieval quality remain untested. Phase 7/8 must test them as relevant.

## LangChain adapter follow-up

`phase2_langchain_cuda_20260920/probe_result.json` records a successful GPU probe
using the actual `JinaEmbeddings` LangChain adapter. At 2,048 tokens, one synthetic
tokenization/transfer/forward/pooling operation took 0.089 seconds after earlier
shorter inputs had warmed up the device. This is not a throughput benchmark.
Padded two-text batches matched independent calls within 4.5e-8 maximum absolute
difference; query/document encoding matched. Full-window early/late pooling
differed by less than 1.5e-8. Overflow was rejected before inference.

The explicit CPU path also passed in `phase2_langchain_cpu_20260920/`. Both use
the same weights, FP32 and content pooling. Long corpus runs remain for later
phases. Model loading/pooling helpers now live in `embedding/model_adapter.py`;
the probe calls that adapter rather than maintaining a second embedding path.

## Reproduce

From the repository root, use the existing `.venv` or install the pinned Phase 2
dependencies with uv. The resolved package environment is saved with each probe;
the requirements file pins the model runtime but is not a full transitive lock.
The current run used the existing environment; a separate clean Phase 2 install
has not yet been tested.

```bash
uv pip install --python .venv/bin/python -r requirements-phase2.txt
.venv/bin/python -m rag_pipeline.embedding.download_probe --bundle data/outputs/rag/model_probe_20260920
.venv/bin/python -m rag_pipeline.embedding.probe --bundle data/outputs/rag/model_probe_20260920 --output data/outputs/rag/phase2_probe_repeat --device cuda
```

The download retrieves fixed revisions and verifies every expected SHA256. It
does not execute code. The probe verifies hashes again before local imports.
Choose a new output path per run. On this machine the already-downloaded bundle
is `data/outputs/rag/model_probe_20260920`; it can be reused offline.

For later answer generation, **GroqCloud free tier** is selected with zero paid
budget. The exact available model is still to be chosen; this probe does not
measure generation speed or quality.
