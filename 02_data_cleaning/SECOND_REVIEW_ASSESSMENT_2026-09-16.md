# Assessment of the supplied second-model review

## Conclusion

The recommendation to prioritize genuine evidence problems is sensible. The
claim that the supplied loader makes the corpus ready is not supported: the
loader has filtering and provenance bugs, and several page classifications in
the review are contradicted by the PDFs themselves.

This was a review only. The proposed loader was reproduced in memory for checks;
it was not installed into the project or used to rewrite the corpus.

## CSV checks

`final_data/review_queue.csv` contains 4,365 rows. The supplied counts match:

| Flag | Pages |
|---|---:|
| layout_fallback_review | 3,310 |
| table_structure_review | 1,465 |
| substantial_image_content | 899 |
| reading_order_review_required | 473 |
| unmapped_private_use_characters | 349 |
| fragmented_text_or_formula | 234 |
| low_text_page | 148 |
| no_extractable_text | 83 |
| ocr_candidate | 44 |
| margin_removal_skipped_to_preserve_text | 22 |
| contains_tamil_or_sinhala | 4 |

Flags overlap and must not be added as independent page counts. The supplied
review omits the 22 skipped-margin-edit flags from its table. Its count of 128
reading-order flags in `URJT_FSSH_HB_2025pdf.pdf` is correct.

A flag is a review signal, not a diagnosis. `no_extractable_text` means extraction
returned no text, not that the visible PDF page is blank. `layout_fallback_review`
also includes original-text fallback, not only Poppler. `substantial_image_content`
does not distinguish photographs, logos, screenshots, tables or diagrams.

Only 18 of the 44 OCR-candidate pages are physical page 1. This does not establish
how many are covers: internal dividers and photo pages also occur. The CSV's short
preview alone cannot establish that most images are decorative or most fragmented
text is valid mathematics.

## Source-page checks of the review's examples

Physical PDF page numbers are used throughout.

| Source | Page(s) | Verified observation |
|---|---|---|
| `aysha/UOM_FIT_HB_2020.pdf` | 41–42 | Photo collages, not the substantive text pages the review describes. OCR is not required to recover educational rule/course text on these pages. |
| `haleema/UW_Agri-2019.pdf` | 58 | Decorative photographic divider. Its location deep in the PDF does not make it substantive text. Other listed agriculture pages were not all inspected in this follow-up. |
| `vithusan/Operating system_1.pdf` | 85 | An instructional CPU/cache/memory/device diagram, omitted from the review's proposed content-page OCR list. OCR labels alone would not validate the arrows and relationships. |
| `aysha/UOC_FMF_HB_2018_2022.pdf` | 9 | Visually readable prose about university history, vision and mission, despite garbled extracted text. It is not inherently unreadable source material. Recovery or an explicit scope exclusion is needed. |
| `aysha/UCSC_HB_2024.pdf` | 15 | A decorated Introduction divider with shadow/outline title styling, not a two-column prose page. Duplicate extracted words do not establish a column-order defect. |

The earlier visual audit also established that `FOCP Theory Lrecture-01.pdf`,
physical page 15, loses its image-based comparison table. It has
`low_text_page;substantial_image_content;layout_fallback_review`, **not**
`ocr_candidate`, because its title is extractable. Limiting OCR/content review to
the 44 OCR-candidate pages misses this evidence.

## Problems in the supplied loader

1. **Skipped text is not removed.** The code builds `clean_pages`, but then sets
   `text = doc["text"]`, preserving the full original text. It only drops metadata
   entries. A synthetic excluded-page sentinel remained in the output; on the
   real UCSC document, physical page 15 was dropped from metadata while its text
   was not removed by page slicing.
2. **Page offsets become invalid.** Formatting removal and deduplication change
   text lengths, but `char_start`/`char_end` are retained unchanged. For UCSC, 162
   of 163 retained page slices differed from their previous content when using
   the old offsets on transformed text. In a correct implementation, page-level
   transformations and exclusions must be followed by rebuilding the document
   text and recomputing all offsets.
3. **The known garbled pages survive.** UOC FMF pages 9 and 10 contain 1,860 and
   2,344 cleaned characters, respectively. The loader's `< 50` check does not
   exclude them. The separately supplied `is_garbled()` function is never called
   by that loader.
4. **Global word deduplication can change evidence.** The regex transforms
   `Credits: 2 2 3` into `Credits: 2 3`, and changes valid phrases such as
   `had had an examination`. Repeated table values and repeated words require
   positional/source evidence before removal. Deduplication cannot restore column
   reading order or table relationships.
5. **The garbled-text detector does not recognize English.** Alphabetic ASCII
   gibberish `VWXGHQWV PXVW QRW FKHDW` gets a readability ratio of 1.0, while the
   valid course-code/credit sequence `ICT1234 2 ICT1235 2 ICT1236 3` gets 0.0.
   It is unsuitable as an automatic deletion rule, particularly for this dataset.
6. **Heading removal discards structural markers.** Removing every `#` heading
   prefix without preserving verified section boundaries undermines the intended
   structure-aware comparison. Removing artificial whole-page fences can be
   useful, but actual code examples, scientific symbols and source footnotes
   must remain distinguishable.

The OCR example is a useful starting experiment, not an acceptance test. The
document is closed on successful execution but not protected by a context manager
against exceptions. More importantly, OCR output still needs source verification,
especially for course codes, credits, formulas and tables. A fixed page-number
allowlist based only on flags is insufficient.

## Work actually needed

No requirement exists to manually repair all 4,365 flagged pages or OCR every
image. Prioritize evidence relevant to the declared experimental scope:

1. Classify pages/regions using source inspection: retain meaningful text,
   preserve document identity/year metadata, exclude confirmed decorative/blank
   content, and recover image or garbled evidence where it is in scope.
2. Normalize a consistent indexable representation with useful headings and
   explicit table/column relationships. Fix the verified missing-table, formula,
   punctuation and number-formatting cases from the first audit.
3. Apply exclusions to text, not just metadata. Regenerate full documents, page
   spans, manifests and hashes together. Preserve an exclusion/recovery audit trail.
4. Verify the source evidence for the evaluation questions and recheck a fresh
   sample. Freeze one corpus version for every chunking method. If complex visual
   material is excluded, declare that scope and limit the resulting research claims.

After those checks, the accepted corpus can feed chunking and embeddings via the
`text` field in `corpus.jsonl`. The current full corpus can support pipeline
prototyping, but the supplied loader does not establish research readiness.
