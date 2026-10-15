"""Evidence metrics use original source spans, never method-specific gold chunks."""

from ..retrieval.diagnostics import covered


def score_retrieval(question, retrieval, context, documents, k_values=(1, 3, 5, 10)):
    """Missing retrieval/context counts as zero for answerable questions.

    Several chunks may jointly cover one evidence alternative for recall. MRR
    requires one chunk to cover a complete alternative on its own. Whitespace
    gaps do not lose evidence; missing letters, punctuation or numbers do.
    """
    result = {
        "retrieval_status": "missing" if retrieval is None else "complete",
        "context_status": "missing" if context is None else "complete",
        "evidence_content_tokens": None if context is None else context["evidence_content_tokens"],
        "units": [], "metrics": {},
    }
    names = [name + str(k) for k in k_values for name in ("recall_at_", "complete_at_", "mrr_at_")]
    names += ["context_recall", "context_complete"]
    result["metrics"] = dict.fromkeys(names)
    if question["answerability"] != "answerable":
        return result
    if not question["evidence_units"]:
        raise ValueError("Answerable questions require validated evidence units")
    hits = [] if retrieval is None else retrieval["hits"]
    pieces = [] if context is None else context["pieces"]
    for unit in question["evidence_units"]:
        alternatives = unit["alternatives"]
        result["units"].append({
            "unit_id": unit["unit_id"],
            "first_single_chunk_rank": next((h["rank"] for h in hits
                if any(covered(a, [h], documents) for a in alternatives)), None),
            "covered_at_k": {str(k): any(covered(a, hits[:k], documents) for a in alternatives) for k in k_values},
            "covered_in_context": any(covered(a, pieces, documents) for a in alternatives),
        })
    units = result["units"]
    for k in k_values:
        count = sum(u["covered_at_k"][str(k)] for u in units)
        ranks = [u["first_single_chunk_rank"] for u in units
                 if u["first_single_chunk_rank"] is not None and u["first_single_chunk_rank"] <= k]
        result["metrics"].update({f"recall_at_{k}": count / len(units),
                                  f"complete_at_{k}": int(count == len(units)),
                                  f"mrr_at_{k}": 1 / min(ranks) if ranks else 0.0})
    count = sum(u["covered_in_context"] for u in units)
    result["metrics"].update(context_recall=count / len(units), context_complete=int(count == len(units)))
    return result
