# Final text preparation — 16 September 2026

## Result

Processed the full contents of all 60 text files and their page metadata: Aysha 12,
Haleema 24, Vithusan 24; 6,215 physical PDF pages. The output is a plain-text baseline
for the next chunking experiments. This was a complete automated text pass with
targeted inspection, not a manual semantic certification of every PDF page.

| Change | Count |
|---|---:|
| Artificial whole-page backtick wrappers removed | 3,310 |
| Lines with generated emphasis cleaned | 12,603 |
| HTML line-break tags replaced | 8,137 |
| Generated heading prefixes removed | 1,981 |
| Markdown table alignment rows removed | 366 |
| Recorded page-number lines removed | 2,812 |
| Corrupt-font pages quarantined | 44 |

Genuine footnote asterisks, literal code quotes, mathematical symbols, repeated
numeric values, course codes, table separators and fixed-width column spacing
remain. Asterisks still present in a file are not automatically extraction noise.

## Explicit exclusion

`aysha/UOC_FMF_HB_2018_2022.pdf`, physical pages 9–56 except 18, 19, 25 and 26,
contains corrupted embedded-font extraction, such as `7KH8QLYHUVLW` instead of
readable prose. These 44 pages now have empty `clean_text`,
`include_in_chunking: false`, and a recorded exclusion reason. Their original
extraction survives in `excluded_text`, `raw_text`, and the full output backup.
The exclusion applies only to the recorded source hash. The remaining readable
pages of the handbook are retained. All 60 documents remain in the corpus.

This removes unusable extraction from the retrieval input; it does not recover
the missing knowledge. Exclude questions requiring these pages from the current
benchmark, or recover and validate their text before adding such questions.

## Validation

- 17 regression tests passed, including code, Unicode, footnotes, repeated credits,
  word/number boundaries, conservative footer removal and exact heading spans.
- All 60 source PDF hashes and all 6,215 page records passed integrity validation.
- TXT, detailed JSON and JSONL agree; page/heading offsets and text hashes match.
- No remaining triple-backtick fences, HTML break tags or number-only pages.
- A whole-corpus alphanumeric-token comparison found no unexpected word/number
  loss on retained pages after accounting for recorded footer removals and
  formatting tags. This does not establish correct reading order or table meaning.
- Repeating the preparation command validates the existing result without applying
  formatting transformations a second time.

Machine-readable results are in `final_data/preparation_report.json`,
`preparation_content_check.json`, and `validation_report.json`. The full backup
location is recorded in the manifest and preparation report. The preexisting UCSC
wrapper-only edit was reconciled; unrelated content changes would stop the script.

## Next-stage input

Use the prepared `.txt` files or `corpus.jsonl`'s identical `text` field. JSONL is
preferable when retaining source/page references. For structure-aware chunking,
use `pages[].headings` and the detailed JSON/PDF outline; generated Markdown
heading markers have been removed from TXT. Heading offsets are page-local;
add the page's document offset when locating them in full-document text.

Keep this same corpus for every chunking method. Image-only content has not been
OCRed, and table/formula reading order can still be imperfect. Check the source
evidence for evaluation questions. No overall cleaning-accuracy percentage is
claimed by these automated checks.
