# Research Explorer — Streamlit UI

An inspection interface for the completed educational RAG experiments. It reads
saved outputs and lets the team trace a PDF passage or question through the five
methods. Opening a screen does not run an embedding model or call Groq.

## Start the app

From the repository root:

```bash
uv pip install --python .venv/bin/python -r requirements-ui.txt
.venv/bin/streamlit run research_gui/app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false
```

Open **http://127.0.0.1:8501** in your browser. Keep the terminal running; stop it
with Ctrl+C. If port 8501 is busy, choose another port and use that address.

The UI requires the local `final_data/`, original PDFs under `data/`, and saved
research outputs under `data/outputs/rag/`. These ignored data files are not
distributed by Git. Missing PDFs produce a preview message; incompatible or
changed artifacts produce an error instead of a silently mixed comparison.

## A ten-minute team walkthrough

1. **Overview:** start with `phase9_retrieval_pilot_20260927`. Explain the same
   corpus, five separate collections, and the 15 generated answers out of 100
   prepared prompts. A cached-repeat retrieval run has no linked generation run
   unless one was explicitly created for it.
2. **Documents:** choose `aysha/UCSC_HB_2024.pdf`, physical page **71**. Compare
   the original page image with the prepared text. Physical and printed page
   numbers can differ.
3. **Chunks:** inspect fixed-size and sliding-window chunks on that page. Colours
   show membership; orange shows source text shared by multiple chunks. Select
   a chunk to read its complete text, including text on neighbouring pages.
4. **Compare methods:** compare fixed-size with structure-aware on the same
   source page. The comparison aligns by character offsets, not chunk numbers.
5. **Chunks / Semantic:** inspect boundary distance, threshold and the actual
   split decision. Size-forced cuts are distinct from semantic cuts.
6. **Chunks / Late:** see the longer encoder window with the exact target chunk
   highlighted. Its surrounding text informs the vector but is not added to the
   indexed chunk text. The displayed cosine is not an accuracy score.
7. **Embeddings:** inspect the selected chunk's 512 values and row-to-source map.
   This reads the saved NumPy array; it does not load Jina or use the GPU.
8. **Retrieval & context:** choose `pilot_001`. Compare the five first results,
   inspect the chosen method's full ranked chunks and its packed evidence.
   Try `pilot_004` to inspect evidence skipped by the context budget.
9. **Answers & review:** inspect `pilot_001`, `pilot_004` and `pilot_019`. Expand
   citations to see the supplied passage, or open its original PDF page. Expand
   the exact messages to see what the generator actually received.
10. Select `pilot_002`: the UI correctly says **Not generated**. An absent answer
    is not presented as an abstention or API failure. Write a reviewer observation
    and download it for discussion with the team.

Document and method selectors remain in the sidebar. The question selection
persists when following a source link and returning to the question screens.
Use **How this stage works** for input/operation/output explanations and an
optional display of the relevant Python implementation.

## What the seven screens show

| Screen | Main evidence |
|---|---|
| Overview | Discovered runs, corpus counts, method sizes and generation coverage. |
| Documents | PDF preview, exact prepared text, exclusions, flags, headings and eligible regions. |
| Chunks | Complete chunks, neighbours, overlaps, heading decisions, semantic scores and late windows. |
| Compare methods | Two methods aligned to one physical page, plus document-wide size statistics. |
| Embeddings | Saved vector values, dimensions, norm, row mapping and recorded encoder settings. |
| Retrieval & context | Top-K ranked chunks, included source pieces and packing decisions. |
| Answers & review | Five-method answer table, actual prompts, usage, citation sources and draft review-note exports. |

Changing K changes the displayed ranking only; it does not rerun retrieval or
repack the saved top-10 context. Jina evidence counts and Qwen prompt counts are
labelled separately. Content counts exclude special tokens; encoder counts include
them. Reference answers are shown only in a separate human-review section.

## How the code is organised

- `app.py`: navigation, shared selectors, overview and error presentation.
- `artifacts.py`: cached loading, per-file checksums, recorded lineage and PDF/vector access.
- `components.py`: escaped source highlights, source navigation, chunk cards and explanations.
- `document_views.py`: document, chunk, comparison and embedding screens.
- `question_views.py`: retrieval, answer, citation and review screens.
- `tests/test_explorer.py`: corruption/path checks and real-artifact navigation tests.

The UI reuses the strict corpus loader. A selected Phase 9 run supplies the exact
Phase 7/8 input paths and hashes. Original chunking diagnostics are located through
the embedding run's saved input hashes. Answer runs must reference the selected
retrieval manifest. Files displayed by the UI are checked against their completed
stage manifests; this is not a rerun of every independent phase audit.

Caches are keyed by file path, modification time, size and expected checksum.
**Refresh saved files** clears the in-memory caches. NumPy arrays are opened with
read-only memory mapping and pickle disabled. Source text is HTML-escaped before
highlighting. The app does not read `.env`, credentials or model weights.

## Scope and limitations

- This release explores existing runs. New questions, pipeline execution and
  background job controls are a separate extension; there are no placeholder
  execution buttons that pretend to run them.
- Review notes download as JSON with the request/context identity. They do not
  change source data, completed outputs or benchmark approvals.
- Grouped citation labels such as `[S1, S2]` are recognised by the UI. This extends
  inspection only; the original saved Phase 10 diagnostics remain unchanged.
  Label membership does not establish that the cited passage supports a claim.
- PDF previews show whole pages. Highlighted intervals are exact in prepared
  text; this release does not claim exact text-to-PDF rectangle alignment.
- Intermediate encoder token arrays discarded by earlier phases are unavailable.
  Saved window ownership and pooled vectors remain inspectable.
- Current pilot annotations are labelled as current local records and may have
  changed since a run. They are never inserted into a saved generator prompt.
- Counts, plots and selected examples do not establish a winning method. Reviewed
  benchmark evaluation and systematic failure analysis still belong to Phases 11–12.

## Checks

```bash
.venv/bin/python -m unittest discover -s research_gui/tests -v
.venv/bin/python -m unittest discover -s rag_pipeline/tests -q
.venv/bin/python -m unittest discover -s 02_data_cleaning/tests -q
```

Real-app tests use Streamlit's AppTest and are skipped if the local completed
research outputs are absent. They exercise all seven screens and five methods,
excluded pages, source navigation, question-state preservation, and questions with
no generated answer. A provider-call stub rejects HTTP requests during the screen
test. No benchmark or generation work is executed by these tests.

Framework references: [Streamlit app testing](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest)
and [caching](https://docs.streamlit.io/develop/concepts/architecture/caching).
