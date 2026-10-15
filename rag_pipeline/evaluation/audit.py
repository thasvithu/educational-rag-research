"""Independently recompute pilot retrieval metrics and compare an offline repeat."""

import argparse

from ..common.config import ROOT, load_config, validate_run_id, within
from ..common.load_corpus import file_hash, load_corpus, require
from ..common.run_manifest import write_json
from ..embedding.inputs import verify_manifest
from .inputs import read_json, read_rows


def audit(run_id, repeat_id):
    for name in (run_id, repeat_id):
        validate_run_id(name)
    run, repeat = [ROOT / "data/outputs/rag" / name for name in (run_id, repeat_id)]
    report = run.with_name(run.name + "_audit.json")
    require(not report.exists(), "Preserve existing audit; use a new run ID")
    manifest = verify_manifest(run)
    verify_manifest(repeat)
    settings = read_json(run / "run_config.json")
    require(settings == read_json(repeat / "run_config.json"), "Repeat settings differ")
    for name in manifest["artifact_sha256"]:
        if name != "environment.json":
            require(file_hash(run / name) == file_hash(repeat / name), f"Repeat differs: {name}")
    corpus = load_corpus(ROOT / "final_data", load_config(ROOT / "rag_pipeline/config.json"))
    documents = {d.document_id: d for d in corpus.documents}
    require(read_json(run / "corpus_snapshot.json")["files"] == corpus.snapshot["files"], "Prepared inputs changed")
    source = within(ROOT, settings["retrieval_run"])
    source_settings = read_json(source / "run_config.json")["settings"]
    questions = {q["question_id"]: q for q in read_rows(within(ROOT, source_settings["pilot"]))}
    traces = {}
    for method in {r["method"] for r in read_rows(run / "scores.jsonl")}:
        variant = source / method / "variant_001"
        traces[method] = tuple({r["question_id"]: r for r in read_rows(variant / name)}
                              for name in ("retrieval.jsonl", "generation_contexts.jsonl"))

    def complete(unit, passages):
        # A deliberately different oracle: enumerate required character positions
        # instead of using the production interval-subtraction implementation.
        for alternative in unit["alternatives"]:
            text = documents[alternative["document_id"]].text
            positions = {i for i in range(alternative["char_start"], alternative["char_end"]) if not text[i].isspace()}
            for passage in passages:
                if passage["document_id"] == alternative["document_id"]:
                    positions = {i for i in positions if not passage["char_start"] <= i < passage["char_end"]}
            if not positions:
                return True
        return False

    checked = 0
    for row in read_rows(run / "scores.jsonl"):
        q = questions[row["question_id"]]
        if q["answerability"] != "answerable":
            require(all(v is None for v in row["retrieval"]["metrics"].values()), "Nonanswerable retrieval score")
            continue
        retrieved, contexts = traces[row["method"]]
        hits = retrieved.get(q["question_id"], {}).get("hits", [])
        pieces = contexts.get(q["question_id"], {}).get("pieces", [])
        units = q["evidence_units"]
        metrics = {}
        for k in settings["k_values"]:
            flags = [complete(unit, hits[:k]) for unit in units]
            rank = next((h["rank"] for h in hits[:k] if any(complete(unit, [h]) for unit in units)), None)
            metrics.update({f"recall_at_{k}": sum(flags)/len(units), f"complete_at_{k}": int(all(flags)),
                            f"mrr_at_{k}": 1/rank if rank else 0.0})
        flags = [complete(unit, pieces) for unit in units]
        metrics.update(context_recall=sum(flags)/len(units), context_complete=int(all(flags)))
        require(row["retrieval"]["metrics"] == metrics, "Independent metric oracle disagrees")
        checked += 1
    result = {"status": "passed", "use": "provisional_pilot_only", "run_id": run_id, "repeat_id": repeat_id,
              "answerable_question_method_records_recomputed": checked, "identical_repeat": True,
              "prepared_files_unchanged": len(corpus.snapshot["files"]), "judge_quality_validated": False}
    write_json(report, result)
    print(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repeat-id", required=True)
    args = parser.parse_args()
    audit(args.run_id, args.repeat_id)
