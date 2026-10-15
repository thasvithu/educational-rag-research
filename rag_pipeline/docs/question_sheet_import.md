# Importing the team question sheet

The CSV importer creates a **draft intake workspace** under
`data/benchmark/imports/`. It keeps the original questions and answers, applies
documented exclusions, and locates candidate evidence in the frozen prepared
corpus. It does not run retrieval/generation, approve questions, select test
records, or modify the existing benchmark manifest.

## Current import: 6 October 2026

Vithusan authorized excluding four named problem records and one copy of each
of three duplicate pairs. The original 160-row CSV remains unchanged.

| Excluded ID | Reason / retained counterpart |
|---|---|
| HALEEMA_033 | Medical-certificate question with unrelated credit evidence |
| HALEEMA_143 | Duration question answered with study mode |
| HALEEMA_146 | Role question answered with a person's name |
| AYSHA_071 | Physical page 220 in a 41-page source |
| HALEEMA_050 | Duplicate of retained HALEEMA_034 |
| HALEEMA_051 | Duplicate of retained HALEEMA_038 |
| HALEEMA_052 | Duplicate of retained HALEEMA_036 |

Output: `data/benchmark/imports/team_153_20261006/`.

- **153 retained drafts**, all with their original IDs.
- **56 records with aligned candidate evidence** (55 answerable and one with an
  unresolved answerability label). Matching evidence does not establish correctness.
- **12 blank-evidence records recovered** by locating the full reference answer.
- **24 page references corrected** from matched source spans.
- **90 source filename cells resolved** using case/spacing cleanup, removal of a
  pasted page note, a unique basename, or the explicit FMAS filename alias.
- **26 records without the importer's intake flags**; **127 have one or more
  unresolved checks**. These checks overlap: 68 concern underspecified question
  scope, 25 require a separate unanswerable/ambiguous label, and 73 answerable
  records lack a selected full matching passage. The 26 are still draft annotations,
  not independently approved or ready for final scoring.

## Open these files

| File | Purpose |
|---|---|
| `normalized_drafts.csv` | All 153 rows with verified filename/page/text adjustments and visible unresolved checks. Start here. |
| `retained_questions.csv` | The same 153 original rows, byte-value-preserving at the CSV field level. |
| `excluded_questions.csv` | The seven removed rows, with the reason for each exclusion. |
| `unresolved_questions.csv` | Only records with further intake checks; contains original cells and suggested source/pages. |
| `draft_records.jsonl` | Machine-readable intake records, source offsets/hashes, candidate locations, actions and original rows. |
| `decisions.json` | Exclusion decisions, exact input hash and explicit filename alias. |
| `import_summary.json` | Counts, PDF identities, code hash and generated artifact hashes. |

## Evidence matching rules

1. Search the supplied supporting passage against the named document.
2. Interpret `same` as the reference answer, as Vithusan requested. When evidence
   is blank, also try the reference answer as a candidate.
3. Permit whitespace normalization and, as a second pass, straight/curly quotation
   mark differences. Preserve exact source characters and offsets in the result.
4. Do not use fuzzy semantic matching, invented wording, or concatenate passages.
5. A very short answer (less than 20 characters or four whitespace-separated
   words) is not automatically accepted as a supporting passage. Numbers/names
   usually require surrounding source context to justify the answer.
6. Reject spans crossing excluded/blank-page barriers. For repeated text, select
   a location only if it is unique or uniquely supported by supplied physical
   pages; otherwise preserve the candidates for review.

The eight retained `same` records did not yield a sufficiently long full source
match under these rules. That does not prove their answers are false: paraphrases,
short names, formatting, or combined passages can prevent an exact match. Their
meaning was interpreted as requested, but exact evidence remains to be selected.

Original Approved cells are preserved in `original_row` and the retained CSV.
The normalized import says Pending because independent reviewer names and
verification records were not supplied. No human reviewer is invented.

## Why these are intake records rather than final benchmark JSONL

`draft_records.jsonl` has `record_type: benchmark_intake_draft` and
`intake_schema_version: 1`. It intentionally differs from the final benchmark
schema. Unknown labels remain null, splits remain unassigned, and matching text
is stored under `selected_evidence_candidate`, not asserted as reviewed atomic
gold evidence. `rubric` and `evidence_units` remain empty until the reference facts
and supporting units are finalized. Do not feed this intake file to the scored
benchmark validator or bypass that validator.

Existing exposed source-family restrictions are recorded through `development_only`.
Other source families still need grouping before any development/test assignment;
false does not mean that a question is eligible for held-out testing.

## Reproduce in a new directory

```bash
.venv/bin/python -m rag_pipeline.evaluation.import_question_sheet \
  --input 'QA_final_2026_10_03 - Sheet1.csv' \
  --decisions data/benchmark/import_decisions_20261006.json \
  --output data/benchmark/imports/team_153_repeat
```

The decisions file pins the CSV checksum. A revised sheet requires revisiting its
exclusion decisions and input hash. Existing output directories cannot be
overwritten. Local CSVs and decision files remain under the existing ignored data
workflow; share them with teammates separately from code.

## Checks

```bash
.venv/bin/python -m unittest rag_pipeline.tests.test_question_sheet_import rag_pipeline.tests.test_benchmark -v
```

The 17 tests cover Unicode offsets, quote/whitespace normalization, repeated text,
excluded gaps, short answers, unresolved labels, independent-review gates and the
existing benchmark protections. The real import also hashes all 41 resolved
original source PDFs and compares their identities to the loaded corpus.

Next: review unresolved content, build explicit answer rubrics and evidence units,
then agree source families and final split/counts. Final scored evaluation remains
subject to the benchmark's normal review and freeze gates.

## Targeted correction sheets — 6 October 2026

A separate export now turns intake flags into plain-English actions. See
[the team instructions](../../../data/benchmark/reviews/team_corrections_20261006/README.md).
Aysha has 77 correction rows and Haleema has 50. Another 26 rows are in the
review-only file. All 153 original question IDs are preserved. Proposed wording
is separate from the editable question; no human approvals or splits are assigned.

To recreate the export in a new directory:

```bash
.venv/bin/python -m rag_pipeline.evaluation.prepare_correction_sheet \
  --output data/benchmark/reviews/team_corrections_new
```

The returned correction files will need a correction-aware import and fresh
evidence validation; do not feed them directly to the evaluation command or
overwrite the original import. The original CSV, imported drafts and benchmark
manifest were verified unchanged.
