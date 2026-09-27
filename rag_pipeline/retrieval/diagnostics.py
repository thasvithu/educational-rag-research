"""Pilot evidence inspection after retrieval; labels never enter the search API."""

from .assemble_context import subtract_intervals


def covered(alternative, passages, documents):
    intervals = [(p["char_start"], p["char_end"]) for p in passages if p["document_id"] == alternative["document_id"]]
    missing = subtract_intervals(alternative["char_start"], alternative["char_end"], intervals)
    text = documents[alternative["document_id"]].text
    return all(not text[a:b].strip() for a, b in missing)


def diagnose(question, hits, context, documents, k_values):
    result = {"question_id": question["question_id"], "answerability": question["answerability"],
              "review_status": question["review"]["status"], "use": "pilot_diagnostic_only", "units": []}
    if question["answerability"] != "answerable":
        result["note"] = "No recall judgment for unanswerable, ambiguous or extraction-diagnostic cases."
        return result
    for unit in question["evidence_units"]:
        alternatives = unit["alternatives"]
        first = next((hit["rank"] for hit in hits if any(covered(a, [hit], documents) for a in alternatives)), None)
        result["units"].append({"unit_id": unit["unit_id"], "first_single_chunk_rank": first,
                                "covered_at_k": {str(k): any(covered(a, hits[:k], documents) for a in alternatives) for k in k_values},
                                "covered_in_context": any(covered(a, context["pieces"], documents) for a in alternatives)})
    return result


def write_report(path, questions, by_method, contexts, diagnostics):
    lines = ["# Phase 9 pilot retrieval inspection", "",
             "These 20 draft questions support development. Their reference labels still require human review; "
             "the following is a diagnostic, not a final benchmark or method ranking.", "",
             "All five methods search all 60 documents with the same query vectors. "
             "No reference document, answer, page or scope label is passed to search.", "",
             "The raw top 10 and budgeted source pieces are saved separately. "
             "A reference passage can be retrieved and then excluded by the context budget.", "",
             "## Evidence checks across the pilot", "",
             "`a/b → c/b` means gold units covered in top 10 → units retained in context. "
             "Coverage uses full non-whitespace source spans, allowing multiple chunks to jointly cover a unit.", "",
             "| Question | " + " | ".join(by_method) + " |", "|---|" + "---|"*len(by_method)]
    for i, question in enumerate(questions):
        cells = []
        for method in by_method:
            units = diagnostics[method][i]["units"]
            cells.append(f"{sum(u['covered_at_k']['10'] for u in units)}/{len(units)} → "
                         f"{sum(u['covered_in_context'] for u in units)}/{len(units)}" if units else "N/A")
        lines.append(f"| {question['question_id']} | " + " | ".join(cells) + " |")
    lines += ["", "## Ten questions to inspect", ""]
    for i, question in enumerate(questions[:10]):
        lines += [f"### {question['question_id']}: {question['question']}", "",
                  f"Category: {question['category']}; review: {question['review']['status']}.", ""]
        for method in by_method:
            hits, context, diagnosis = by_method[method][i]["hits"], contexts[method][i], diagnostics[method][i]
            lines += [f"#### {method}", "", f"Evidence tokens: {context['evidence_content_tokens']}/{context['evidence_budget']}; "
                      f"included pieces: {len(context['pieces'])}; budget-skipped candidates: "
                      f"{sum(d['status'] == 'skipped_budget' for d in context['decisions'])}.", "",
                      "Reference units: " + "; ".join(f"{u['unit_id']}: first complete single chunk rank "
                      f"{u['first_single_chunk_rank']}, top-10 union={u['covered_at_k']['10']}, "
                      f"context={u['covered_in_context']}" for u in diagnosis["units"]), ""]
            for hit in hits[:3]:
                lines += [f"- Rank {hit['rank']}, score {hit['score']:.6f}: {hit['source']}, physical pages "
                          f"{hit['page_numbers']}, chars [{hit['char_start']}, {hit['char_end']}). "
                          f"Chunk `{hit['chunk_id']}`."]
            lines += ["", "Top passage preview (first 500 characters):", "", "~~~~text", hits[0]["text"][:500], "~~~~", ""]
    lines += ["## Interpretation", "", "Investigate misses using the exact reference and retrieved spans. "
              "Similar handbook versions, fragmented evidence, duplicate overlap and context-budget loss are possible causes; "
              "these tables do not establish a cause. Do not tune on future held-out test results. "
              "No LLM answers or final evaluation scores were generated.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
