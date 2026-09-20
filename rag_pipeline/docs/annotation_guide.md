# Annotating and reviewing the benchmark

## Start here

Open `data/benchmark/pilot_review.md`. It contains **20 drafts with answers and
evidence**, including links to 13 rendered physical PDF pages from seven PDFs.
The 18 answerable drafts reference six questions from each member's documents.
Collection balance here is for instruction; the final benchmark needs balanced
question/document categories, not simply equal numbers per member.

The assistant visually compared these reference passages with the rendered PDFs.
**None has human approval yet.** The two non-answerable cases still need scope
review. These files are ignored by Git; distribute them privately with the corpus.

## A teammate's review

1. Read the question without looking at chunks. Is the university, year or lecture
   sufficiently specified? Does the source answer the actual question?
2. Open the original PDF at the **physical page number**. For example, UCSC PDF
   page 71 is printed page 57. Some UWU physical pages contain two printed pages.
3. Check every number, condition, exception and table column against the draft
   answer and the prepared evidence. Missing or corrupt evidence is diagnostic-only.
4. Check whether each required evidence unit is sufficient and whether another
   valid location should be annotated as an alternative. Do not add irrelevant
   paragraphs just to make a reference span easier to retrieve.
5. Check the rubric: it should award equivalent correct wording but require exact
   thresholds and scope. For `pilot_014`, 8% is of the **practical component**;
   for `pilot_018`, **more than 35** must not become **at least 35**.
6. For unanswerable cases, inspect the relevant source family and explain why the
   fact is absent/underspecified. An unsuccessful keyword search alone is insufficient.
7. Enter your actual name, decision and corrections in `annotation_reviews.csv`.
   Suggested reviewer assignments are suggestions, not completed reviews.
8. Resolve corrections first. Apply the final reviewed wording to `pilot.jsonl`,
   then record the approval against its current annotation hash. Review by a
   second team member is required when a teammate authored/revised the question.

Vithusan can bring the completed review sheet back for reconciliation. The CSV
is a review worksheet; editing it does **not automatically approve JSON records**.
Do not pre-fill a teammate's approval or copy an old hash after changing a question.

## Record contract (schema version 1)

Each JSONL line has question identity, corpus hash, split, language, category,
answerability, question, scope, reference answer, rubric, question-family `group_id`,
author, `evidence_units`, rationale when non-answerable, and `review`.

Each evidence unit has a `unit_id`, a claim, and a list of alternative source spans.
Each span has document/source identity, original PDF SHA256, prepared-text SHA256,
document-level Python character start/end (end exclusive), exact `text`, and
one-based physical `page_numbers`. Each alternative must support the same claim.
For facts distributed across separate passages, use multiple units, not one
invented concatenated span. The validator rejects spans across excluded/blank gaps.

Approved review fields:

```json
{
  "status": "approved",
  "reviewer": "Actual independent teammate name",
  "reviewer_kind": "human",
  "pdf_verified": true,
  "reviewed_content_hash": "SHA256 from annotation_hash(final_record)",
  "notes": "Actual checks, corrections and adjudication outcome"
}
```

`annotation_hash` lives in `rag_pipeline.evaluation.validate_benchmark`. Approval
does not enter that hash, but wording, evidence, rubric, group and split do. Moving
a reviewed pilot record to development therefore requires recording a review of
the updated record, even if its question has not changed.

## Splits and validation

Currently only `pilot.jsonl` exists. Development and test have **zero records**.
After the pilot review, create `development.jsonl` and `test.jsonl`, register them
under `files` in `benchmark_manifest.json`, and extend `document_groups` to all
referenced sources. All versions/near-duplicates of one source family stay on one
side of the development/test split. Similar slides reused across courses also
need grouping. Keep pilot source families in `reserved_development_groups`.

Reviewed pilot questions may become development questions; they never become
test questions. Ambiguous and extraction-dependent cases stay in pilot/diagnostics.
Write test questions from other reviewed families and do not use test scores for tuning.

```bash
.venv/bin/python -m rag_pipeline.evaluation.validate_benchmark
.venv/bin/python -m rag_pipeline.evaluation.validate_benchmark --require-ready
```

The first command validates draft structure/evidence and reports remaining review
work. The second also requires protocol approval and the agreed development/test
counts; **it currently exits nonzero as intended**. After all actual reviews and
protocol decisions are complete:

```bash
.venv/bin/python -m rag_pipeline.evaluation.validate_benchmark --freeze
```

This refuses drafts and exclusively creates `frozen_snapshot.json` with corpus,
manifest, question-file, source PDF and protocol-document hashes. Subsequent validation detects changed frozen
artifacts. It does not make local files physically read-only; back them up and
preserve the snapshot with the run. The validator checks recorded review metadata,
not the truth of someone's claimed review or semantic sufficiency of evidence.
