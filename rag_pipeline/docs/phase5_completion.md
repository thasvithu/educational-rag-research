# Phase 5 — Structure-aware chunking

Completed on 21 September 2026. This is the third chunking method in our comparison.
It uses document headings and paragraph/list/table boundaries, with the same pinned
Jina tokenizer and **256-content-token cap** as the earlier methods.

## The simple idea

Fixed-size chunking fills a token window. Structure-aware chunking first asks:
**“Which section does this text belong to, and which pieces should stay together?”**

1. Find usable heading hints in the saved metadata.
2. Split the source into sections and record their heading paths.
3. Pack neighboring paragraphs within each section while they fit.
4. Keep recognizable list items and fitting table blocks together.
5. Split oversized material using table rows, sentences, and finally token windows.

There is **no overlap**. Chunks are exact source substrings; section titles are
metadata and are never inserted into the text. Normal page breaks can occur inside
a section or chunk. Blank/excluded pages remain barriers and reset heading context.

## Heading decisions

The cleaner saved many heading candidates, including some bold sentences. A
candidate must match an actual complete source line before it can be used.

| Hint | Treatment |
|---|---|
| Prepared heading | Validate its existing document-level offsets and line text. |
| Font candidate | Match a unique normalized line on its physical page; use font size as supporting evidence. |
| PDF outline entry | Match the title on its destination page; reject unmatched, ambiguous or excluded destinations. |

We reject common prose, bullet items, column-like lines, contents entries with dot
leaders, contact lines and unsupported numbered list entries. Short heading shape,
capitalization, numbering, inferred levels and font evidence guide the remaining
decisions. A large-font candidate is at least 15% larger than the document's median
**candidate** font size; that median is not a certified body-font measurement.
Duplicate hints for the same span become one heading with multiple source labels.

Outline levels take priority, followed by numbering depth and prepared inferred
levels. An unknown level becomes level 2, treated as a sibling of other level-2
headings. Adjacent title lines are packed together when possible. A repeated current
heading at the same level is treated as a continuation rather than a new section.

These are deterministic heuristics, not perfect document understanding. All
acceptances, rejections and duplicate decisions are saved for review.

## Fallbacks and exact text

- Without a usable heading, the chunk has an empty `section_path` and uses paragraph
  and list/table rules. A page without a new heading may continue an earlier section
  if it is inside the same eligible region.
- A recognizable table block stays whole when it fits. Larger blocks use visible
  row boundaries. Wrapped cells remain uncertain; the code does not reconstruct
  table facts from coordinates.
- Oversized paragraphs or list/row units use conservative sentence boundaries.
  The sentence helper protects common abbreviations, initials and decimal values.
- A unit still above the cap uses the shared exact-offset token splitter.
- Tokenizer-ignored decorative glyphs, such as private-use PDF bullets, attach to
  a neighboring unit without deletion. Entire nonblank regions with no encodable
  text still fail explicitly.

Every accepted chunk is retokenized to check its actual input size. `token_count`
includes Jina's two special tokens, so the maximum is **258 encoder input tokens**.

## Code and commands

| File | Responsibility |
|---|---|
| `chunking/structure_aware/headings.py` | Validate and combine heading hints. |
| `chunking/structure_aware/units.py` | Locate paragraph, list, table and sentence units. |
| `chunking/structure_aware/chunker.py` | LangChain `TextSplitter`, section paths, packing and fallbacks. |
| `chunking/structure_aware/reporting.py` | Heading decisions and fallback reports. |
| `chunking/structure_aware/run.py` | Method entry point using the common artifact writer. |
| `configs/structure_aware.json` | Versioned settings and the 256-token cap. |
| `tests/test_structure_aware.py` | Boundary, hierarchy, table, fallback and glyph checks. |

From the project root, use the existing environment and a **new run ID**:

```bash
.venv/bin/python -m rag_pipeline chunk-structure --run-id my_structure_sample --sample
.venv/bin/python -m rag_pipeline chunk-structure --run-id my_structure_full
```

The sample takes the first document by source name from each member. The command
also supports `--config`, `--method-config` and `--model-config`. No new packages,
embedding inference, GPU access or API calls are required for this phase.

The CLI supplies validated PDF metadata to the LangChain splitter. Calling
`StructureAwareSplitter.split_text()` directly supplies no PDF hints, so it uses
the fallback rules. Use the CLI to reproduce the complete research method.

## Saved artifacts

Use the completed full run:

```text
data/outputs/rag/phase5_structure_full_v2_20260920/
    run_config.json
    corpus_snapshot.json
    environment.json
    stage_manifest.json
    structure_aware/variant_001/
        chunks.jsonl
        chunk_stats.json
        heading_decisions.jsonl
        heading_summary.json
        structure_stats.json
        structure_report.md
        inspection_examples.json
        inspection_report.md
        stage_manifest.json
```

The later embedding stage should read **`chunks.jsonl`**. Each chunk includes its
section path, source heading references, section span, unit kinds, missing-heading
flag and fallback events, alongside the shared source/page/token fields.

`heading_decisions.jsonl` records why each hint was accepted or rejected.
`structure_stats.json` summarizes fallbacks. Its `table_row:table_rows` count means
row units emitted from oversized table blocks; it does **not** mean that those rows
were broken in half. Sentence/token events identify finer splits. A unit may need
both fallbacks, so those counts must not be added as if they were disjoint.

The earlier `phase5_structure_full_20260920` directory records the failed glyph-only
unit case and is retained for diagnosis. The completed `full_v2` run is the usable
artifact. The independent repeat is `phase5_structure_repeat_20260921`.

## Verified results

| Check | Result |
|---|---:|
| Documents / eligible regions | 60 / 169 |
| Structure-aware chunks | **13,377** |
| Content-token range | 1–256 |
| Mean / median content tokens | 163.28 / 188 |
| Unique eligible characters covered | **11,424,026 — 100%** |
| Overlap / empty chunks / overflow chunks | 0 / 0 / 0 |
| Accepted headings / candidate hints | 8,448 / 89,071 |
| Sections | 6,209 |
| Chunks without a heading | 86 |
| Chunks using table-row / sentence / token fallback | 1,372 / 940 / 261 |
| Observed full-run processing time | About 83 seconds |
| Automated checks | **78 pipeline + 17 cleaning tests passed** |

All 60 documents had at least one accepted heading, but some regions lacked usable
headings. There are 2,375 chunks below 64 content tokens. We retain these short
sections rather than merge unrelated sections solely to fill the token budget.

### Size comparison

| Method | Chunks | Mean content tokens | Total content tokens |
|---|---:|---:|---:|
| Fixed-size | 8,627 | 253.18 | 2,184,219 |
| Sliding window | 11,419 | 254.33 | 2,904,243 |
| Structure-aware | 13,377 | 163.28 | 2,184,164 |

Structure-aware chunking creates more, smaller chunks while retaining the same
unique text. Small total-token differences without overlap arise from independent
substring tokenization. More chunks mean more vectors and more per-chunk overhead
later; these counts do not establish better retrieval or answers.

## What to review

- [Admission exception, course table, cross-page section and unheaded lecture example](../../data/outputs/rag/phase5_verification_20260920/review_examples.md)
- [Heading/fallback report](../../data/outputs/rag/phase5_structure_full_v2_20260920/structure_aware/variant_001/structure_report.md)
- [Complete statistics](../../data/outputs/rag/phase5_structure_full_v2_20260920/structure_aware/variant_001/chunk_stats.json)
- [Independent verification and method comparison](../../data/outputs/rag/phase5_verification_20260920/verification.json)

The audit checked every saved substring, source page, token count, section/heading
reference, fallback span, coverage interval and artifact hash. Both full runs have
identical canonical chunk records. All 122 prepared-input hashes match Phase 1.
The [baseline regression check](../../data/outputs/rag/phase5_verification_20260920/baseline_regression.json)
also confirmed that the fixed-size sample's 555 records and sliding sample's 739
records remain identical after the shared writer changes.

The examples also show a **questionable heading hierarchy**: a contents-like label
was inferred as an ancestor of a history section. Matching real source text proves
provenance, not that the inferred parent relationship is correct. Treat heading
paths as diagnostic metadata. They are not added to embeddings in this phase.
Table reading order and missing image content remain preparation limitations.

These reports are local, ignored artifacts. The audit can be repeated with:

```bash
.venv/bin/python data/outputs/rag/phase5_verification_20260920/check_outputs.py
```

**Next: Phase 6 — semantic chunking**, when requested. The larger reviewed benchmark
is still due before Phase 11; no final evaluation scores have been produced.
