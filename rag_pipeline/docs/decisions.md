# Implementation decisions

## Phase 1 — 19 September 2026

| Decision | Reason and consequence |
|---|---|
| JSONL prepared text is the input contract | It matches TXT and carries physical page provenance. Detailed JSON is validated and can supply extra structural metadata. |
| Inputs remain read-only | The legacy validator now accepts an external report path; pipeline reports go under ignored experiment outputs. |
| Excluded and blank prepared pages are barriers | A blank extraction may hide an image or missing passage. Every method will use the same conservative region boundaries. |
| Ordinary nonblank page breaks are allowed within a region | Page layout alone should not force every method to restart. |
| Source text is never normalized in the loader | Exact slicing preserves character offsets and repeated occurrences. |
| Loaded headings use document offsets | The conversion is done once in the loader while original page-local metadata remains untouched. |
| Token fields are nullable in the Phase 1 schema | Model/tokenizer selection belongs to Phase 2. Later chunkers must provide measured counts and tokenizer identity. |
| Minimal runtime has four pinned dependencies | Phase 1 can run without embedding/generation packages. The full requirements file includes the minimal file. |
| New run IDs are mandatory | A previous successful or failed inspection cannot be silently overwritten. |
| Both Git revision and source hashes are recorded | Experimental code may be uncommitted; a commit ID alone would not identify the code actually run. |

## Observed corpus boundaries

- 60 documents; 6,215 physical page records.
- 44 explicit exclusions, retained in metadata but absent from retrievable text.
- 172 included pages have empty/whitespace-only prepared text.
- 169 continuous eligible regions, each containing one or more nonblank pages.

These regions constrain future chunks; they do not modify `final_data` and are
not themselves chunking outputs. The gap policy forms part of the corpus identity.
Changing it later requires a recorded configuration/protocol change and a consistent
rerun for every method.

## Phase 2 decisions still pending

- Embedding model, tokenizer, input limits, and matched early/late feasibility.
- Benchmark composition, reviewed reference evidence, splits, and primary metrics.
- Chunk-size/overlap/semantic parameter grids and compute/API budgets.
