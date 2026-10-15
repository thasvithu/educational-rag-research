"""Human answer judgments and citation syntax checks; no invented judge scores."""

import re

from ..common.load_corpus import digest_json, require
from .validate_benchmark import annotation_hash

METRICS = ("answer_correctness", "fully_correct", "faithfulness", "citation_support",
           "relevance", "abstention", "unsupported_answer")


def review_identity(question, answer, context, trial_id):
    """Bind judgments to the exact annotation, generated answer and evidence."""
    return digest_json({"annotation_hash": annotation_hash(question), "answer": answer,
                        "context": context, "trial_id": trial_id})


def citation_checks(text, context):
    """Check label existence, including [S1, S2]. Existence is not support."""
    groups = re.findall(r"\[(S\d+(?:\s*,\s*S\d+)*)\]", text)
    labels = sorted({label for group in groups for label in re.findall(r"S\d+", group)})
    available = {p["citation_id"] for p in context["pieces"]}
    return {"cited_labels": labels, "unknown_labels": sorted(set(labels) - available),
            "note": "Label syntax only; a human or judge must check whether each claim is supported."}


def review_template(question, answer, context, trial_id, method):
    return {
        "question_id": question["question_id"], "method": method, "trial_id": trial_id,
        "review_identity": review_identity(question, answer, context, trial_id),
        "status": "pending", "reviewer": "", "reviewer_kind": "human", "notes": "",
        "question": question["question"], "answerability": question["answerability"],
        "reference_answer": question["reference_answer"], "rubric": question["rubric"],
        "evidence_units": question["evidence_units"],
        "answer_text": answer["response"]["text"], "supplied_context": context["pieces"],
        "facts_satisfied": [None for _ in question["rubric"]], "contradiction": None,
        "checkable_claims": None, "supported_claims": None,
        "cited_claims": None, "supported_cited_claims": None, "uncited_claims": None,
        "relevance": None, "abstention": None, "unsupported_answer": None,
    }


def score_answer(question, answer, context, trial_id, review=None):
    result = {"answer_status": "missing" if answer is None else answer["status"],
              "review_status": "not_reviewed", "metrics": dict.fromkeys(METRICS)}
    if answer is None or answer["status"] != "complete":
        result["review_status"] = "no_completed_answer"
        return result
    result["citation_checks"] = citation_checks(answer["response"]["text"], context)
    if review is None or review.get("status") == "pending":
        return result
    try:
        require(review.get("status") == "approved", "Review is not approved")
        require(review.get("review_identity") == review_identity(question, answer, context, trial_id), "Stale answer review")
        require(review.get("reviewer_kind") == "human" and bool(review.get("reviewer", "").strip()), "Named human reviewer required")
        require(bool(review.get("notes", "").strip()), "Review notes required")
        facts = review["facts_satisfied"]
        require(isinstance(facts, list) and len(facts) == len(question["rubric"])
                and all(type(v) is bool for v in facts), "One boolean per rubric fact required")
        for key in ("contradiction", "abstention", "unsupported_answer"):
            require(type(review[key]) is bool, f"Invalid {key}")
        for key in ("checkable_claims", "supported_claims", "cited_claims", "supported_cited_claims", "uncited_claims"):
            require(type(review[key]) is int and review[key] >= 0, f"Invalid {key}")
        require(review["supported_claims"] <= review["checkable_claims"], "Supported claims exceed checkable claims")
        require(review["supported_cited_claims"] <= review["cited_claims"], "Supported citations exceed cited claims")
        require(review["cited_claims"] + review["uncited_claims"] == review["checkable_claims"], "Claim counts disagree")
        require(type(review["relevance"]) is int and review["relevance"] in (0, 1, 2), "Relevance must be 0, 1 or 2")
        metrics = result["metrics"]
        metrics.update(faithfulness=review["supported_claims"] / review["checkable_claims"] if review["checkable_claims"] else None,
                       citation_support=review["supported_cited_claims"] / review["cited_claims"] if review["cited_claims"] else None,
                       relevance=review["relevance"])
        if question["answerability"] == "answerable":
            metrics.update(answer_correctness=0.0 if review["abstention"] else sum(facts) / len(facts),
                           fully_correct=int(all(facts) and not review["contradiction"] and not review["abstention"]))
        elif question["answerability"] == "unanswerable":
            metrics.update(abstention=int(review["abstention"] and not review["unsupported_answer"]),
                           unsupported_answer=int(review["unsupported_answer"]))
        result.update(review_status="approved", reviewer=review["reviewer"], uncited_claims=review["uncited_claims"])
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        result.update(review_status="invalid", review_error=str(exc), metrics=dict.fromkeys(METRICS))
    return result
