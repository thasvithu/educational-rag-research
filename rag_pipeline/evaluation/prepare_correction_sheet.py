"""Export an author's correction queue without changing benchmark annotations."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re

from ..common.config import ROOT, within
from ..common.load_corpus import file_hash, require
from ..common.run_manifest import write_json

SOURCE = ROOT / "data/benchmark/imports/team_153_20261006"
OUTPUT = ROOT / "data/benchmark/reviews/team_corrections_20261006"
EVIDENCE_FLAGS = {"evidence_needs_source_check", "supporting_passage_too_short", "multiple_evidence_locations"}
WORDING_FLAGS = {"question_source_scope_needs_review", "question_time_scope_needs_review"}
ACTIONS = {
    "evidence_needs_source_check": "Copy the exact PDF passage(s) that support every fact in the answer. The supplied wording did not resolve to one source span; this does not prove the answer is wrong.",
    "supporting_passage_too_short": "Copy the full sentence or table row, including its heading/labels. A number or name alone does not explain what the value means.",
    "multiple_evidence_locations": "Choose the passage that supports this question from the candidate physical pages shown. Include surrounding text to identify the intended occurrence.",
    "physical_pages_need_confirmation": "Give the PDF viewer's physical page number, counting the cover as page 1. Keep printed page labels or section notes in Notes.",
    "source_or_scope_missing": "Name the intended PDF(s). For a corpus-wide unanswerable case, write 'whole corpus' in Notes and list the documents/topics checked; do not invent a supporting passage.",
    "question_source_scope_needs_review": "Make the question identify its intended handbook/institution/programme. Check the proposed wording; filename-based scope is a draft, not an approved change.",
    "question_time_scope_needs_review": "Decide whether this asks about the historical handbook or live information. For the handbook, replace 'current/right now' with its stated period; for live information, explain the missing evidence.",
    "choose_unanswerable_or_ambiguous": "Choose one label: unanswerable (clear request, answer absent from corpus), ambiguous (request needs clarification), or answerable if supporting evidence exists.",
    "nonanswerability_rationale_missing": "Explain what is missing or ambiguous and which sources you checked. One handbook omitting an answer does not establish absence across all 60 documents.",
    "reference_response_missing": "Write the expected response: a supported answer, an insufficient-evidence response, or a request for the specific missing clarification.",
    "choose_question_category": "Choose one category: definition, explanation, procedure, eligibility, exception, table_lookup, multi_passage, abstention or ambiguity. Use multi_passage when separate passages are needed.",
}


def task_group(row):
    flags = set(row["unresolved"])
    if not flags:
        return "4 - Review existing annotation"
    if "choose_unanswerable_or_ambiguous" in flags:
        return "2 - Decide answerability"
    if flags & EVIDENCE_FLAGS:
        return "1 - Confirm supporting evidence"
    if flags <= WORDING_FLAGS:
        return "3 - Confirm question wording"
    return "2 - Complete missing information"


def proposed_wording(row):
    """Suggest explicit scope without guessing what abbreviated filenames mean."""
    if not set(row["unresolved"]) & WORDING_FLAGS or not row["scope_source"]:
        return ""
    question = row["question"]
    if row["answerability"] == "answerable":
        question = re.sub(r"\bcurrent\s+", "", question, flags=re.I)
    else:
        # A temporal unanswerable question must not silently become historical.
        return ""
    return f'According to the handbook "{Path(row["scope_source"]).name}", {question[0].lower() + question[1:]}'


def export_row(row):
    flags = row["unresolved"]
    require(set(flags) <= set(ACTIONS), "Unknown intake flag; add a clear action before exporting")
    original = row["original_row"]
    candidate = row["selected_evidence_candidate"]
    pages = sorted({p for c in row["evidence_candidates"] for p in c["page_numbers"]})
    action = " | ".join(f"{i}. {ACTIONS[flag]}" for i, flag in enumerate(flags, 1))
    if not flags:
        action = "No automated intake flag. A different teammate should confirm the question, answer, intended source and evidence. This row is not automatically approved."
    if row["question_id"] == "AYSHA_019":
        action += " Also confirm why the Humanities question uses a file named UOK_FS_HB_2024_2025.pdf; the filename alone cannot establish whether this is correct."
    return {
        "Task group": task_group(row), "Question ID": row["question_id"], "Author": row["author"],
        "Original sheet row": row["sheet_row"], "What to do": action,
        "Proposed question wording (not approved)": proposed_wording(row),
        "Candidate physical pages (not proof of support)": ", ".join(map(str, pages)),
        "Question": row["question"], "Reference answer": row["reference_answer"],
        "Answerability": row["answerability"] or original["Answerability"],
        "Question category": row["category"] or original["Question category"],
        "Exact PDF filename": row["scope_source"] or original["Exact PDF filename"],
        "Physical PDF page(s)": ", ".join(map(str, candidate["page_numbers"])) if candidate else original["Physical PDF page(s)"],
        "Supporting passage(s)": candidate["text"] if candidate else original["Supporting passage(s)"],
        "Notes / Reason if unanswerable": row["rationale"],
        "Author correction notes": "", "Correction status": "Not started" if flags else "Review only",
        "Reviewer": "", "Review decision": "Pending", "Reviewer notes / PDF checked": "",
        "Previously supplied review decision": original["Review decision"],
        "Technical adjustments already recorded": " | ".join(row["automatic_actions"]),
        "Original intake flags": " | ".join(flags),
    }


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(source=SOURCE, output=OUTPUT):
    source, output = Path(source).resolve(), Path(output).resolve()
    require(output.is_relative_to(ROOT / "data/benchmark/reviews"), "Keep review exports under data/benchmark/reviews")
    require(not output.exists(), "Preserve existing exports; use a new output directory")
    summary = json.loads((source / "import_summary.json").read_text())
    for name, expected in summary["artifact_sha256"].items():
        require(file_hash(within(source, name)) == expected, f"Import artifact changed: {name}")
    drafts = [json.loads(line) for line in (source / "draft_records.jsonl").read_text().splitlines() if line.strip()]
    require(len({d["question_id"] for d in drafts}) == len(drafts) == summary["retained_drafts"], "Draft identity/count differs")
    rows = sorted([export_row(d) for d in drafts], key=lambda r: (r["Task group"], r["Author"], r["Original sheet row"]))
    corrections = [r for r in rows if r["Original intake flags"]]
    review_only = [r for r in rows if not r["Original intake flags"]]
    output.mkdir(parents=True)
    write_csv(output / "corrections_all.csv", corrections)
    write_csv(output / "review_only.csv", review_only)
    for author in sorted({r["Author"] for r in rows}):
        name = re.sub(r"[^a-z0-9]+", "_", author.lower()).strip("_")
        require(bool(name), "Author filename is empty")
        write_csv(output / f"{name}_corrections.csv", [r for r in corrections if r["Author"] == author])
    stats = {"status": "correction_queue_only", "import_sha256": file_hash(source / "draft_records.jsonl"),
             "source_import": str(source.relative_to(ROOT)), "retained": len(rows), "corrections": len(corrections),
             "review_only": len(review_only), "task_groups": dict(Counter(r["Task group"] for r in rows)),
             "corrections_by_author": dict(Counter(r["Author"] for r in corrections)),
             "review_only_by_author": dict(Counter(r["Author"] for r in review_only)),
             "note": "Flags request clarification; they are not findings that an answer is wrong. No approvals or splits assigned."}
    lines = ["# Team question correction sheet — 6 October 2026", "",
             "We are preparing the final research benchmark. You do not need to rewrite all questions. Please use your correction file and complete only the actions shown for each question.", "",
             "## Files to use", "", "| File | Rows | Purpose |", "|---|---:|---|",
             f"| [corrections_all.csv](corrections_all.csv) | {len(corrections)} | Combined correction queue. |"]
    for author, count in sorted(stats["corrections_by_author"].items()):
        name = re.sub(r"[^a-z0-9]+", "_", author.lower()).strip("_")
        lines.append(f"| [{name}_corrections.csv]({name}_corrections.csv) | {count} | Questions authored by {author}; this is not assignment by who collected the PDF. |")
    lines += [f"| [review_only.csv](review_only.csv) | {len(review_only)} | Existing annotations with no automated intake flags; named independent review still needed. |", "",
              "## Work in this order", ""]
    lines += [f"- **{group}: {count} questions.**" for group, count in sorted(stats["task_groups"].items())]
    lines += ["", "A flag does not mean the answer is incorrect. Exact text matching can fail for a valid paraphrased answer. We need an exact supporting passage so evidence can be scored fairly across chunking methods.", "",
              "## How to fill the sheet", "",
              "1. Import your CSV into Google Sheets or open it in Excel. Keep the Question ID unchanged. Use either the author files or the combined file as your working copy, to avoid conflicting edits.",
              "2. Read `What to do`. Edit the existing Question, Reference answer, Answerability, category, PDF, page, passage or rationale cells that need correction. Leave correct values unchanged.",
              "3. Proposed question wording is a suggestion. Copy it into Question only after checking the intended source. You may replace the filename with a clear institution/faculty/handbook-year description. Keep changes consistent with the original question's intent.",
              "4. Copy evidence from the original PDF. Use physical viewer page numbers, with the cover as page 1. For several passages, label each PDF/page and paste each passage separately. Include table headings, conditions and exceptions.",
              "5. For unanswerable questions, explain why the requested information is absent and what sources were checked. No gold passage is required when no answer exists. For ambiguous questions, say exactly what needs clarification; these belong in a separate diagnostic group.",
              "6. Add Author correction notes and set Correction status to `Ready for review`. Another teammate checks the PDF and fills Reviewer, Review decision (`Approved` or `Changes requested`) and Reviewer notes / PDF checked.",
              "7. Return the edited CSV/Excel files. We will revalidate the exact text/page mappings and prepare evidence units, rubrics and development/test grouping. These exports are not directly accepted by the scoring command.", "",
              "Use a different reviewer from the question author. Previous Approved cells are preserved for reference; they did not name a reviewer or establish a recorded independent PDF check. Do not approve on behalf of somebody else.", "",
              "## Work handled in code", "",
              "The import already recorded source-path repairs and candidate page/text matches. The Technical adjustments column shows them. After corrected sheets return, the pipeline work includes character offsets, hashes, evidence-unit/rubric drafting, duplicate/source-family grouping and split validation. Teammates do not need to calculate offsets or hashes.", "",
              "Candidate page lists indicate matching text locations, not verified supporting evidence. Questions with only wording flags still need source-scope confirmation. A failed match alone is not an extraction error. Do not delete or relabel questions just to improve scores.", "",
              "The original 160-row CSV, the 153-question import and the pilot benchmark remain unchanged. No test split, final scoring, reviewer approval or new exclusion has been created.", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    stats["artifact_sha256"] = {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()}
    write_json(output / "summary.json", stats)
    for name, expected in summary["artifact_sha256"].items():
        require(file_hash(within(source, name)) == expected, f"Source import changed: {name}")
    print(json.dumps(stats, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    prepare(args.source, args.output)
