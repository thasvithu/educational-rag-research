"""Offline evaluation of saved traces: provisional pilot checks, or final held-out test scoring."""

from collections import Counter
import csv
import json
from pathlib import Path

from ..common.config import ROOT, load_config
from ..common.load_corpus import digest_json, file_hash, load_corpus, require
from ..common.run_manifest import create_run, environment, finish_run, write_json
from ..common.tokenization import load_tokenizer
from ..embedding.encode_chunks import atomic_jsonl
from .answer_metrics import review_template, score_answer
from .compare_runs import aggregate, paired_comparisons, question_clusters
from .inputs import load_evaluation_inputs, read_rows
from .retrieval_metrics import score_retrieval

DEFAULT_RETRIEVAL = ROOT / "data/outputs/rag/phase9_retrieval_pilot_20260927"
DEFAULT_GENERATION = ROOT / "data/outputs/rag/phase10_generation_pilot_20260927"


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def run_evaluation(run_id, *, provisional=False, final_test_scoring=False, retrieval_run=DEFAULT_RETRIEVAL,
                   generation_run=DEFAULT_GENERATION, reviews=None):
    require(provisional != final_test_scoring,
            "Pass exactly one of --provisional (pilot, development-only checks) or "
            "--final-test-scoring (frozen benchmark, held-out test split).")
    split = "test" if final_test_scoring else "pilot"
    use = "final_test_scoring" if final_test_scoring else "provisional_pilot_only"
    retrieval_run, generation_run = Path(retrieval_run).resolve(), Path(generation_run).resolve()
    config = load_config(ROOT / "rag_pipeline/config.json")
    corpus = load_corpus(ROOT / config.corpus_dir, config)
    tokens = load_tokenizer(ROOT / "rag_pipeline/configs/models.json")
    questions, traces, validation, benchmark, retrieval_config, generation_config = load_evaluation_inputs(
        retrieval_run, generation_run, corpus, tokens, split=split)
    trial = generation_config["trial_id"]
    judgments = {}
    if reviews is not None:
        for review in read_rows(Path(reviews)):
            key = (review["method"], review["question_id"], review["trial_id"])
            require(key not in judgments, "Duplicate answer review")
            require(key[0] in traces and key[1] in traces[key[0]]["answers"] and key[2] == trial,
                    "Review targets an unknown method/question/trial")
            judgments[key] = review
    effective = {"schema_version": 1, "use": use, "corpus_hash": corpus.corpus_hash,
                 "retrieval_run": str(retrieval_run.relative_to(ROOT)), "generation_run": str(generation_run.relative_to(ROOT)),
                 "retrieval_manifest_sha256": file_hash(retrieval_run / "stage_manifest.json"),
                 "generation_manifest_sha256": file_hash(generation_run / "stage_manifest.json"),
                 "benchmark_sha256": validation["artifact_hashes"], "benchmark_manifest_sha256": validation["manifest_sha256"],
                 "protocol_sha256": validation["protocol_hashes"],
                 "reviews_sha256": file_hash(Path(reviews)) if reviews is not None else None,
                 "k_values": retrieval_config["settings"]["k_values"], "trial_id": trial,
                 "bootstrap_samples": 2000, "bootstrap_seed": 42, "judge_calls": 0,
                 "implementation_sha256": {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}
    run, manifest = create_run(ROOT, config, run_id)
    manifest.update(stage="phase11_final_test_evaluation" if final_test_scoring else "phase11_provisional_evaluation",
                    use=use, config_hash=digest_json(effective), corpus_hash=corpus.corpus_hash)
    write_json(run / "stage_manifest.json", manifest)
    try:
        write_json(run / "run_config.json", effective)
        write_json(run / "environment.json", environment(ROOT))
        write_json(run / "benchmark_validation.json", validation)
        write_json(run / "corpus_snapshot.json", corpus.snapshot)
        documents = {d.document_id: d for d in corpus.documents}
        rows, templates, coverage = [], [], []
        for method, trace in traces.items():
            method_rows = []
            for q in questions:
                qid = q["question_id"]
                answer, context = trace["answers"].get(qid), trace["contexts"].get(qid)
                row = {"schema_version": 1, "use": use, "question_id": qid,
                       "method": method, "trial_id": trial, "answerability": q["answerability"],
                       "category": q["category"], "benchmark_review_status": q["review"]["status"],
                       "retrieval": score_retrieval(q, trace["retrieval"].get(qid), context, documents, effective["k_values"]),
                       "answer": score_answer(q, answer, context, trial, judgments.get((method, qid, trial)))}
                method_rows.append(row)
                if answer is not None and answer["status"] == "complete":
                    templates.append(review_template(q, answer, context, trial, method))
            rows.extend(method_rows)
            variant = run / method / "variant_001"
            variant.mkdir(parents=True)
            atomic_jsonl(variant / "scores.jsonl", method_rows)
            completed = [a for a in trace["answers"].values() if a["status"] == "complete"]
            coverage.append({"method": method, "questions": len(questions),
                "retrieval_present": len(trace["retrieval"]), "contexts_present": len(trace["contexts"]),
                "answers_complete": len(completed), "answers_missing": len(questions) - len(trace["answers"]),
                "answers_failed": len(trace["answers"]) - len(completed),
                "answer_reviews_approved": sum(r["answer"]["review_status"] == "approved" for r in method_rows),
                "answer_reviews_invalid": sum(r["answer"]["review_status"] == "invalid" for r in method_rows),
                "saved_generation_tokens": sum(a["response"].get("provider", {}).get("usage", {}).get("total_tokens", 0) for a in completed),
                "saved_generation_latency_seconds": sum(a["response"].get("latency_seconds", 0) for a in completed),
                "mean_evidence_content_tokens": sum(c["evidence_content_tokens"] for c in trace["contexts"].values()) / len(trace["contexts"]) if trace["contexts"] else None,
                "ragas_status": "not_run"})
        clusters = question_clusters(questions, benchmark["document_groups"])
        comparison_interpretation = (
            "Final held-out test scoring on the frozen benchmark. Family grouping still limits statistical "
            "power with this few clusters; report alongside RAGAS and human review, not as a sole verdict."
            if final_test_scoring else
            "Provisional pilot diagnostic; sparse groups do not establish a method winner.")
        comparison = paired_comparisons(rows, clusters, samples=effective["bootstrap_samples"], seed=effective["bootstrap_seed"],
                                        interpretation=comparison_interpretation)
        write_json(run / "question_clusters.json", clusters)
        atomic_jsonl(run / "scores.jsonl", rows)
        atomic_jsonl(run / "answer_review_template.jsonl", templates)
        if judgments:
            atomic_jsonl(run / "submitted_answer_reviews.jsonl", list(judgments.values()))
        write_csv(run / "metrics_summary.csv", aggregate(rows))
        write_csv(run / "category_metrics.csv", aggregate(rows, category=True))
        write_csv(run / "coverage.csv", coverage)
        write_json(run / "paired_differences.json", comparison)
        from .ragas_adapter import describe
        try:
            ragas = describe()
        except (ImportError, ValueError) as exc:
            ragas = {"status": "unavailable", "error_type": type(exc).__name__, "judge_calls": 0}
        write_json(run / "ragas_status.json", ragas)
        summary = {"use": use, "split": split, "questions": len(questions), "method_question_records": len(rows),
                   "question_types": dict(Counter(q["answerability"] for q in questions)),
                   "human_reviewed_questions": validation["reviewed"], "freeze_ready": validation["freeze_ready"],
                   "methods": coverage, "generation_model": generation_config["settings"]["model_id"],
                   "judge_calls": 0, "final_test_scoring": final_test_scoring}
        write_json(run / "evaluation_summary.json", summary)
        if final_test_scoring:
            lines = ["# Phase 11 final held-out test evaluation", "",
                     "**Frozen benchmark, held-out test split.** This is the thesis-reportable scoring pass, "
                     "not a development check.", "",
                     f"{len(questions)} test questions × {len(traces)} methods; {validation['reviewed']} "
                     "human-approved reference annotations; benchmark verified against frozen_snapshot.json.",
                     f"{sum(c['answers_complete'] for c in coverage)} saved answers; "
                     f"{sum(c['answers_missing'] for c in coverage)} answers missing. No API calls were made "
                     "by this command (it only scores previously saved traces).", ""]
        else:
            lines = ["# Phase 11 provisional pilot evaluation", "",
                     "These are development checks against draft labels, not thesis findings or a method ranking.", "",
                     f"{len(questions)} questions × {len(traces)} methods; {validation['reviewed']} human-approved reference annotations.",
                     f"{sum(c['answers_complete'] for c in coverage)} saved answers; "
                     f"{sum(c['answers_missing'] for c in coverage)} answers missing. No API calls were made.", ""]
        lines += ["## Read the artifacts", "",
                 "- `scores.jsonl`: each question's source evidence coverage and answer-review status.",
                 "- `metrics_summary.csv` / `category_metrics.csv`: means and explicit scored/eligible counts.",
                 "- `coverage.csv`: missing/failed answers, review coverage, saved usage and context tokens.",
                 "- `paired_differences.json`: paired context-complete differences, grouped bootstrap, seed 42.",
                 "- `answer_review_template.jsonl`: all completed answers with exact evidence and blank review fields.",
                 "- `ragas_status.json`: installed version and prompt definitions; judge metrics have not run.", "",
                 "## Interpretation rules", "",
                 "Recall requires all non-whitespace characters of one full evidence alternative. Several chunks may jointly cover it; MRR requires one complete chunk. The primary metric applies that rule to the context actually supplied to generation.", "",
                 "Missing retrieval/context receives zero for answerable questions. Other question types have no retrieval score. Missing answers and unreviewed/invalid judgments remain missing, with coverage reported. Citation label checks do not establish factual support.", "",
                 ("Bootstrap intervals describe the held-out test split only. Related source/question families are "
                  "resampled together; with this few clusters they remain descriptive, not a proof of significance."
                  if final_test_scoring else
                  "Bootstrap intervals describe this small draft pilot only. Related source/question families are resampled together; they do not establish statistical power or a winning method."), "",
                 ("## Status" if final_test_scoring else "## Remaining before final evaluation"), "",
                 ("This split was frozen in frozen_snapshot.json before this run; the manifest's "
                  "reserved_development_groups, protocol_status and target_counts gates were all satisfied. "
                  "Pair these scores with the RAGAS judge pass and completed human answer review before reporting."
                  if final_test_scoring else
                  "Resolve imported benchmark evidence/reviews, approve the protocol, create protected development/test groups, select settings on development only, freeze inputs, complete comparable generation and answer review/judging. The 153 imported drafts were not scored."), ""]
        (run / "report.md").write_text("\n".join(lines), encoding="utf-8")
        finish_run(run, manifest)
    except BaseException as exc:
        finish_run(run, manifest, error=f"{type(exc).__name__}: {exc}")
        raise
    print(json.dumps({"output": str(run), "questions": len(questions), "records": len(rows), "use": summary["use"]}))
    return run
