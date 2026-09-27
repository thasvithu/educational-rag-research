# Phase 9 — Indexing and retrieval

Completed on **27 September 2026**. All five full-corpus indexes and the 20-question development pilot passed their checks.

## Results and files to review

| Check | Result |
|---|---|
| Indexed vectors | **53,977** across five separate indexes; each searches all **60 documents**. |
| Per method | Fixed 8,627; sliding 11,419; structure 13,377; semantic 11,927; late 8,627. |
| Pilot execution | **20 shared query vectors, 100 searches and 100 assembled contexts**. |
| Saved evidence budget | Every context stays within **1,536 Jina content tokens**, excluding citation wrappers. |
| Repeat | Queries, indexes, rankings, contexts and diagnostics are **byte-identical**. |
| Numerical audit | Maximum FAISS/direct-float64 score difference **1.62e-7**, below the **2e-6** tolerance. |
| Tests | **116 pipeline + 17 cleaning = 133 passed**. |
| Inputs | **122 prepared files** and **26 source/benchmark artifacts** unchanged. |
| Local resources | About **105.42 MiB** total index storage; **302 MiB** peak reserved query GPU memory; **2.21 GiB** peak process memory. |
| Timings | Initial run **72.2 s**, including **57.6 s** input validation; cached repeat **69.4 s**. These are observed development timings. |
| Human review | **0 of 20 pilot questions approved**; final benchmark work remains deferred. |

Start with these local artifacts:

- [Pilot inspection report](../../data/outputs/rag/phase9_retrieval_pilot_20260927/inspection_report.md).
- [Independent audit and repeat check](../../data/outputs/rag/phase9_retrieval_pilot_20260927_audit.json).
- [Execution/resource summary](../../data/outputs/rag/phase9_retrieval_pilot_20260927/retrieval_summary.json).

The main output folder is `data/outputs/rag/phase9_retrieval_pilot_20260927/`.
The repeat is `data/outputs/rag/phase9_retrieval_repeat_20260927/`.

Two useful examples in the report:

- **pilot_001:** The requested UCSC credit definition is the first structure-aware result. The first fixed-size result is a related rule from another university. Source/version qualification matters.
- **pilot_004:** Fixed and sliding retrieval find the reference at rank 10, but the budgeted context omits it. Structure-aware retrieval finds it at rank 2 and retains it. This separates retrieval success from context-selection loss.

These are examples from unreviewed development labels, not a final ranking. The
raw traces are ready for team inspection and later failure analysis.

## What this stage does

We now give each chunking method its own searchable collection. For a question,
the shared Jina model creates one query vector. Each collection finds the chunks
whose vectors are most similar to that query. We save both the ranked results
and the exact evidence that fits into the context budget.

```text
Question → one shared query vector → five separate indexes
         → each method's top 10 → remove repeated source spans
         → keep evidence within the shared budget → saved context for Phase 10
```

No answer-generation model is called during this phase. The 20 pilot questions
are development drafts with source evidence; human annotation review is still
pending. Their diagnostic results must not become final research claims.

## Shared settings

| Choice | Implementation |
|---|---|
| Index | Separate FAISS `IndexFlatIP` for each of the five methods. |
| Vectors | Existing normalized FP32, 512-dimensional vectors, preserved unchanged. |
| Query encoder | Same pinned local Jina model, content-only mean and L2 normalization, CUDA batch one. |
| Search scope | All 60 documents for each method; no gold-source, page or university filter. |
| Ranking | Descending inner-product score; exactly equal scores use ascending chunk ID. |
| K values | 1, 3, 5, 10, saved as prefixes of the same ranking. |
| Search threads | One FAISS thread, recorded in configuration. |
| Context | Top 10 candidates in rank order; 1,536 Jina content tokens of evidence. |
| Overlap | Subtract previously included character spans from the same document. |
| Oversize candidate | Skip all of its new pieces, record the reason, and try the next candidate. |

FAISS documents `IndexFlatIP` as exact inner-product search, usable for cosine
search with normalized vectors. We use LangChain's FAISS integration and its
`Document`/docstore interfaces. Scores are raw inner products of unit vectors,
not LangChain's transformed relevance scores and not probabilities.
[FAISS index documentation](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes),
[LangChain FAISS integration source](https://github.com/langchain-ai/langchain-community/blob/main/libs/community/langchain_community/vectorstores/faiss.py).

The search helper requests all exact scores before sorting and selecting K.
This is manageable for our largest index of 13,377 chunks and resolves ties even
at the K cutoff. It is an intentional research implementation choice; a much
larger corpus would need a more efficient tie-handling design.

## Keeping the comparison fair

Before indexing, the runner verifies completed source manifests, vector
checksums, row maps, corpus hashes, exact text/page spans and full corpus coverage.
It checks query/document model compatibility and the matched fixed/late spans.
Late vectors have a different encoding-purpose identity, but share the same
model, tokenizer, pooling and vector dimensions as the independent methods.

Only `question_id` and the question text/hash go into query records. Reference
answers, evidence spans and scope labels never enter the search interface.
Labels are read afterward to explain whether reference evidence was found.

Each index is saved, reloaded, and checked against its original vectors and
chunks before search begins. Index rows map explicitly to chunk IDs through
JSONL. Loading uses FAISS plus JSON records; it does not require a pickle docstore.

## Raw retrieval versus assembled context

Raw retrieval always preserves the top 10 complete chunks, scores and source
metadata. Context assembly then chooses what can actually be supplied to the LLM.

For example, if sliding-window chunks cover characters `[0,100)` and `[70,170)`,
the context can include `[0,100)` followed by `[100,170)`. The repeated `[70,100)`
is counted once. If removing overlaps creates several separate pieces, each
piece keeps its own source span. We do not pretend that joined pieces form one
original passage. Identical wording in a different document is not removed.

The budget sums the Jina content-token counts of the exact included pieces.
Citation wrappers and separators are additional text. The saved record includes
wrapper counts and the token count of the full rendered context. **Phase 10 must
still check the selected generator's tokenizer and actual context capacity.**

Every candidate has an explicit decision: included, skipped because no new
non-whitespace text remains, or skipped because the new evidence exceeds the
budget. Budget-skipped candidates do not reserve source spans. No candidate is
silently shortened just to fill the remaining budget.

## Pilot inspection

The report shows all 20 questions in an evidence-coverage table and gives
detailed top-three previews for the same first 10 questions across all methods.
The complete top 10 and exact assembled evidence are in JSONL.

Each atomic reference unit counts as covered only when all non-whitespace
characters of at least one valid alternative source span are present. Several
retrieved chunks may jointly supply a unit. The first complete single-chunk rank
is logged separately. Sharing a source page alone does not count as success.

Ambiguous and unanswerable draft questions have no evidence-recall judgment.
The report exposes retrieval misses and evidence lost during context assembly;
it does not establish their causes or select a winning method. Reviewed
development/test questions and final quantitative evaluation remain Phase 11.

## Code and commands

| File | Role |
|---|---|
| `retrieval/inputs.py` | Validate five source sets and model compatibility. |
| `retrieval/build_index.py` | Build, save and reload LangChain/FAISS stores. |
| `retrieval/search.py` | Common exact ranking and tie rule. |
| `retrieval/assemble_context.py` | Source-span deduplication, budget and citations. |
| `retrieval/diagnostics.py` | Pilot-only evidence inspection after retrieval. |
| `retrieval/run.py` | Query encoding, five-method execution and manifests. |
| `retrieval/audit.py` | Saved-output checks and independent dot-product/evidence checks. |
| `configs/retrieval.json` | Method inputs, pilot location and shared settings. |
| `tests/test_retrieval.py` | Synthetic ranking, corruption, overlap and evidence tests. |

Dependencies already exist in the working environment. A teammate can install
the Phase 9 environment using the existing pinned versions:

```bash
uv pip install --python .venv/bin/python -r requirements-phase9.txt
.venv/bin/python -m rag_pipeline retrieve-pilot --run-id my_retrieval_pilot
.venv/bin/python -m rag_pipeline retrieve-pilot --run-id my_retrieval_repeat --cache-only
.venv/bin/python -m rag_pipeline.retrieval.audit --run-id my_retrieval_pilot --repeat-id my_retrieval_repeat
```

Use a GPU-enabled terminal for fresh query encoding. Cached repeats need no
model inference or GPU access. Change source paths in a copy of the retrieval
configuration and pass `--retrieval-config` if local run names differ.

A new run ID is mandatory. If interrupted, use a new ID with the same settings:
committed query vectors are reused and inexpensive indexes/results are rebuilt.
The incomplete run remains visible. Completed outputs are never overwritten.

For a small local exploration in Python, the same search helper is available:

```python
from pathlib import Path
from rag_pipeline.embedding.model_adapter import JinaEmbeddings
from rag_pipeline.retrieval.build_index import load_index
from rag_pipeline.retrieval.search import search

store, chunks = load_index(Path("data/outputs/rag/my_retrieval_pilot/fixed_size/variant_001"))
encoder = JinaEmbeddings.from_config()
query = encoder.embed_query("According to the UCSC 2024 handbook, how many lecture hours equal one credit?")
hits = search(store, query, k=5)
print(hits[0]["source"], hits[0]["page_numbers"], hits[0]["text"])
```

This exploratory snippet prints results only. Use the pilot command for recorded
research comparisons with full input validation and manifests. The wrapped
store rejects implicit text-query encoding so it cannot silently select another
embedding model or bypass the shared ranking rule.

## Output layout

Root output: `data/outputs/rag/<run-id>/`.

- `queries.jsonl`, `query_embeddings.npy`, `query_embedding_rows.jsonl`:
  the shared questions and their vectors, identities and checksums.
- `query_embedding_config.json`: model settings and separate query-cache path.
- `pilot_validation.json`: annotation validity and outstanding review blockers.
- `run_config.json`, `environment.json`, `corpus_snapshot.json`, `stage_manifest.json`:
  reproducibility and artifact hashes.
- `retrieval_summary.json`, `timings.json`, `inspection_report.md`:
  execution counts, resource measurements and human-readable examples.

Each `<method>/variant_001/` contains:

- `index.faiss`, `index_rows.jsonl`, `index_manifest.json`, `chunks.jsonl`.
- `retrieval.jsonl`: complete ranked chunks and K-specific IDs.
- `generation_contexts.jsonl`: exact included pieces, citation wrappers and budget decisions.
- `pilot_diagnostics.jsonl`: reference-evidence checks, explicitly labelled pilot only.
- `stage_manifest.json`: completed method artifact hashes.

The independent audit is written beside the run. Data, query caches, indexes and
reports remain under ignored local storage; code and documentation can be reviewed
in Git. No paid API or answer-generation call is involved.
