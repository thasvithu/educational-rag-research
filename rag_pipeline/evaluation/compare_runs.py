"""Macro averages and paired bootstrap comparisons over related question groups."""

from collections import Counter, defaultdict
from itertools import combinations
import numpy as np


def question_clusters(questions, document_groups):
    """Connect shared question/source families, including multi-source questions."""
    parent = {q["question_id"]: q["question_id"] for q in questions}

    def root(qid):
        while parent[qid] != qid:
            parent[qid] = parent[parent[qid]]
            qid = parent[qid]
        return qid

    seen = {}
    for q in sorted(questions, key=lambda q: q["question_id"]):
        keys = {"question:" + q["group_id"]}
        keys |= {"source:" + document_groups[a["document_id"]]
                 for u in q["evidence_units"] for a in u["alternatives"]}
        for key in sorted(keys):
            if key in seen:
                left, right = sorted((root(q["question_id"]), root(seen[key])))
                parent[right] = left
            seen[key] = q["question_id"]
    return {qid: root(qid) for qid in parent}


def aggregate(rows, category=False):
    groups = defaultdict(list)
    for row in rows:
        key = (row["method"], row["answerability"], row["category"] if category else "all")
        groups[key].append(row)
    output = []
    for (method, answerability, label), group in sorted(groups.items()):
        for kind in ("retrieval", "answer"):
            for metric in group[0][kind]["metrics"]:
                values = [r[kind]["metrics"][metric] for r in group if r[kind]["metrics"][metric] is not None]
                output.append({"method": method, "answerability": answerability, "category": label,
                               "metric": metric, "mean": sum(values) / len(values) if values else None,
                               "scored_questions": len(values), "eligible_questions": len(group),
                               "value_counts": dict(Counter(values)) if metric == "relevance" else {}})
    return output


def _bootstrap_ci(groups, samples, seed):
    if len(groups) < 2:
        return None
    rng = np.random.default_rng(seed)
    sums = np.array([sum(g) for g in groups])
    counts = np.array([len(g) for g in groups])
    picks = rng.integers(0, len(groups), size=(samples, len(groups)))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def paired_metric_comparisons(values_by_method, clusters, metric_name, *, samples=2000, seed=42,
                              interpretation="Provisional pilot diagnostic; sparse groups do not establish a method winner."):
    """Generic paired cluster bootstrap over any {method: {question_id: value}} mapping.

    Same methodology as paired_comparisons (question-macro mean difference A
    minus B, resampled by cluster, seed 42, 2000 samples by default), reused
    for metrics beyond retrieval's context_complete (e.g. a RAGAS score).
    """
    output = []
    for a, b in combinations(sorted(values_by_method), 2):
        if set(values_by_method[a]) != set(values_by_method[b]):
            raise ValueError("Paired methods must include identical questions, including failed runs")
        grouped = defaultdict(list)
        for qid in sorted(values_by_method[a]):
            grouped[clusters[qid]].append(values_by_method[a][qid] - values_by_method[b][qid])
        groups = list(grouped.values())
        values = [v for group in groups for v in group]
        output.append({"method_a": a, "method_b": b, "metric": metric_name,
                       "difference_a_minus_b": sum(values) / len(values) if values else None,
                       "questions": len(values), "clusters": len(groups),
                       "ci95": _bootstrap_ci(groups, samples, seed),
                       "bootstrap_samples": samples, "seed": seed,
                       "interpretation": interpretation})
    return output


def paired_comparisons(rows, clusters, *, samples=2000, seed=42,
                       interpretation="Provisional pilot diagnostic; sparse groups do not establish a method winner."):
    """Resample clusters, retaining all their questions and method pairings.

    The estimate is the question-macro mean difference A minus B. Missing
    retrieval/context has already received the protocol's conservative zero.
    No interval is reported when fewer than two independent clusters exist.
    """
    answerable = [r for r in rows if r["answerability"] == "answerable"]
    by_method = defaultdict(dict)
    for row in answerable:
        by_method[row["method"]][row["question_id"]] = row["retrieval"]["metrics"]["context_complete"]
    return paired_metric_comparisons(by_method, clusters, "context_complete", samples=samples, seed=seed,
                                     interpretation=interpretation)
