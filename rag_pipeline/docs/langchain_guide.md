# LangChain setup: a short guide for the team

## What changed before Phase 3

The pipeline now uses LangChain's `BaseLoader`, `Document`, `Embeddings` and
`ChatGroq` interfaces. Model settings live in `rag_pipeline/configs/models.json`.
The selected embedding model is **jinaai/jina-embeddings-v2-small-en**, pinned to
the tested revision. Local CUDA FP32 with batch size 1 is the default.

These are shared building blocks. Chunking begins in Phase 3; corpus embedding,
retrieval and answer-generation experiments remain in their planned phases.

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

GroqCloud free tier is selected; the exact available free model will be set before
generation. `model_id` stays `null` until then. No API key is needed for loading,
chunking, local embeddings or the offline tests.

At the generation stage, set `GROQ_API_KEY` in your local environment and select
a model available to your free-tier account. `create_chat_model(model_id=...)`
returns a LangChain `ChatGroq` client. Creating it sends no request; invoking it
does. Do not paste the key into code, Git, or benchmark files.

The free-tier configuration records our zero-paid-budget policy. It cannot prove
that an account is on the free tier: a paid Groq account using the same model may
be billed. No paid provider fallback is implemented.

The pinned integration changes a zero temperature to `1e-8`; configuration now
records that actual value. It does not guarantee identical hosted responses.
Basic retries come from LangChain; experiment caching and rate-limit scheduling
will be implemented with the answer-generation stage.

## Installation and checks

```bash
uv pip install --python .venv/bin/python -r requirements-phase2.txt
.venv/bin/python -m rag_pipeline.embedding.download_probe --bundle data/outputs/rag/model_probe_20260920
.venv/bin/python -m unittest discover -s rag_pipeline/tests -v
```

This retains the project's LangChain 0.3 family. It pins `langchain-core==0.3.63`,
`langchain-groq==0.2.3` and `groq==0.13.1`; compatibility was checked in the existing
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
