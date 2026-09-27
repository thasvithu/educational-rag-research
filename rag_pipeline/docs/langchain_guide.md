# LangChain setup: a short guide for the team

## What changed before Phase 3

The pipeline now uses LangChain's `BaseLoader`, `Document`, `Embeddings` and
`ChatGroq` interfaces. Model settings live in `rag_pipeline/configs/models.json`.
The selected embedding model is **jinaai/jina-embeddings-v2-small-en**, pinned to
the tested revision. Local CUDA FP32 with batch size 1 is the default.

Phase 3 now adds a LangChain `TextSplitter` for the fixed-size baseline; see the
[Phase 3 guide and examples](phase3_completion.md). Phase 4 adds a compatible
`SlidingWindowSplitter`; see the [overlap guide](phase4_completion.md). Phase 5 adds
`StructureAwareSplitter` with validated heading metadata and recorded fallbacks;
see the [structure guide](phase5_completion.md). Phase 6 adds `SemanticSplitter`
and a cached LangChain `Embeddings` wrapper around the same Jina adapter; see the
[semantic guide](phase6_completion.md). It embeds source units to choose boundaries.
Phase 7 uses `ChunkEmbeddings` with a tested batch size of eight to encode final
chunk text through that shared
LangChain adapter and save vectors with explicit source row maps; see the
[embedding guide](phase7_completion.md). Phase 8 reuses `JinaEmbeddings.encode_tokens()`
to encode longer passages, then pools exact fixed-size spans after encoding; see
the [late-chunking guide](phase8_completion.md). This small helper preserves token
alignment that a normal independent `embed_documents()` call cannot supply.
Phase 9 uses LangChain's `FAISS`, `Document` and `InMemoryDocstore` interfaces,
with the shared Jina query encoder and explicit tie/context rules; see the
[retrieval guide](phase9_completion.md). Phase 10 adds grounded answers through ChatPromptTemplate and ChatGroq.

## 1. Load the prepared corpus

```python
from rag_pipeline.common.langchain_loader import PreparedCorpusLoader

loader = PreparedCorpusLoader()
documents = loader.load()

print(documents[0].page_content)         # Exact prepared text
print(documents[0].metadata["source"])   # Original PDF filename
print(documents[0].metadata["page_numbers"])
```

This uses the same `load()` convention as `PyPDFLoader` and `TextLoader`. Our
loader reads the already-cleaned JSONL and detailed metadata, checks integrity,
and returns **169 continuous text regions from 60 PDFs**. A PDF interrupted by an
excluded/blank page produces multiple regions. Regions are not finished chunks.

All metadata offsets refer to the full prepared document, including headings
inside `metadata["pages"]`. If a splitter gives a region-local offset of 100,
its document offset is `metadata["char_start"] + 100`. Preserve that relationship
through chunking; do not locate repeated text using its first occurrence.

`lazy_load()` yields regions after the complete corpus has been validated; it
does not promise constant-memory corpus validation. The Phase 1 validation
helpers remain because stock loaders do not know our exclusion and hash rules.

`load_and_split()` is intentionally unavailable: its generic implementation would
copy region offsets onto smaller chunks without updating them. The method-specific
chunkers will use LangChain facilities where they preserve our token/span rules.

Run a short example or save a complete region summary:

```bash
.venv/bin/python -m rag_pipeline.examples.inspect_langchain
.venv/bin/python -m rag_pipeline inspect-langchain --run-id my_langchain_review
```

Choose a new run ID. The report is saved under `data/outputs/rag/<run-id>/`.

## 2. Use the LangChain embedding interface

```python
from rag_pipeline.embedding.model_adapter import JinaEmbeddings

embeddings = JinaEmbeddings.from_config()  # CUDA, from models.json
vectors = embeddings.embed_documents(["A credit represents study time."])
question_vector = embeddings.embed_query("What does one credit represent?")
```

The class implements LangChain's standard `Embeddings` API and can be passed to
its vector-store integrations later. It uses the pinned local Jina bundle,
normalizes 512-dimensional vectors and rejects inputs above 2,048 total tokens.
Content-only pooling is identical for queries and documents. It intentionally
differs from stock pooling that includes special tokens; keep this declared
adaptation identical in the late path as well.

The small `encode_tokens()` helper exposes token vectors and exact offsets for
late chunking. A generic sentence-embedding wrapper cannot expose that operation
while automatically preserving our chosen pooling rules, so this part is explicit.

Your RTX 3050 works outside the restricted agent sandbox. Run GPU commands in a
normal GPU-enabled terminal. The CUDA probe passed at 2,048 tokens with about
**667 MiB peak PyTorch allocation / 798 MiB reservation**. These measurements
exclude other applications and are not a full-corpus memory/runtime guarantee.

An explicit CPU fallback is available:

```python
embeddings = JinaEmbeddings.from_config(device="cpu")
```

The adapter never silently changes device after a CUDA error. Record device and
precision consistently for the five methods, particularly when comparing timing.

## 3. Prepare Groq through LangChain

Phase 10 uses `qwen/qwen3.8-27b` on the configured Groq free account. All five
methods use the same model and instructions. `ChatPromptTemplate` builds the
system/user messages; `GroqAnswerClient` wraps `ChatGroq` with saved provider
metadata. The original `create_chat_model()` factory remains an offline teaching
example; the experiment runner adds explicit retries, quota accounting and caching.

Set `GROQ_API_KEY` in the local environment or ignored `.env`. Never put it in
code or experiment files. The runner reads only this key and disables external
tracing. Preparation, local tests and cached answer replay need no key.

```bash
.venv/bin/python -m rag_pipeline generate-answers --run-id answers_prepare_new --dry-run
.venv/bin/python -m rag_pipeline generate-answers --run-id answers_pilot_new
```

The first command prepares all 100 requests without contacting Groq. The second
selects three questions across all five methods (15 answers). Matching requests
reuse cached answers. The [Phase 10 handover](phase10_completion.md) explains
all-question runs, quota limits, explicit resume, new trials and the output files.

The free-tier configuration records the zero-paid-budget policy; it cannot verify
an account's billing tier. There is no paid provider fallback. Low temperature
and a fixed seed do not guarantee identical fresh hosted responses.

## Installation and checks

```bash
uv pip install --python .venv/bin/python -r requirements-phase2.txt
.venv/bin/python -m rag_pipeline.embedding.download_probe --bundle data/outputs/rag/model_probe_20260920
.venv/bin/python -m unittest discover -s rag_pipeline/tests -v
```

This retains the project's LangChain 0.3 family. It pins `langchain-core==0.3.63`,
`langchain-groq==0.2.3`, `groq==0.13.1` and `langchain-text-splitters==0.3.8`;
compatibility was checked in the existing
environment. The core package contains the loader/document/embedding interfaces.
The full requirements retain LangChain and Community integrations for later use.

The download is needed once for the embedding example/probe; it verifies pinned
files and reuses existing matching files. Tests do not require downloaded weights.
No API calls are made by these tests. Full tests need Phase 2 dependencies; the
original minimal Phase 1 environment can run `rag_pipeline.tests.test_foundation`.

## The remaining phases

Use the 20 pilot questions for development through answer generation. Team review
and the larger benchmark remain necessary before final evaluation in Phase 11.
We will reuse LangChain splitters, vector stores, retrieval and prompt facilities
where they meet the experiment contract. Each method stays in its own module.

References: [LangChain loaders](https://docs.langchain.com/oss/python/integrations/document_loaders)
and [ChatGroq](https://docs.langchain.com/oss/python/integrations/chat/groq).
