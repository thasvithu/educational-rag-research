"""Read and verify existing experiment traces without making model/API calls."""

import hashlib
import json
import math

from ..common.config import ROOT, within
from ..common.load_corpus import digest_json, file_hash, require
from ..common.provenance import pages_for_span
from ..embedding.inputs import verify_manifest
from ..retrieval.assemble_context import assemble_context
from ..retrieval.inputs import METHODS
from .validate_benchmark import validate_directory


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def by_question(rows, questions, method):
    mapped = {}
    for row in rows:
        qid = row["question_id"]
        require(qid in questions and qid not in mapped, "Unexpected or duplicate question in trace")
        require(row["method"] == method, "Mixed methods in trace")
        mapped[qid] = row
    return mapped


def validate_trace(question, retrieval, context, documents, tokens, settings, corpus_hash, config_hash, method):
    """Reject changed questions, wrong source spans and context packing drift."""
    for row in (retrieval, context):
        if row is None:
            continue
        require(row["question"] == question["question"] and row["question_sha256"] ==
                hashlib.sha256(question["question"].encode()).hexdigest(), "Question text/hash differs")
        require(row["corpus_hash"] == corpus_hash and row["retrieval_config_hash"] == config_hash, "Trace configuration differs")
    if retrieval is None:
        require(context is None, "Context without retrieval lineage")
        return
    require(retrieval["filters"] is None, "Evaluation requires whole-corpus retrieval")
    for name in ("similarity", "tie_policy", "k_values"):
        require(retrieval[name] == settings[name], f"Retrieval policy differs: {name}")
    hits = retrieval["hits"]
    require(len(hits) <= max(settings["k_values"]), "Unexpected retrieval depth")
    require(len({h["chunk_id"] for h in hits}) == len(hits), "Duplicate retrieved chunk")
    for rank, hit in enumerate(hits, 1):
        require(hit["rank"] == rank and math.isfinite(hit["score"]), "Invalid rank/score")
        require(hit["method"] == method and hit["corpus_hash"] == corpus_hash, "Mixed retrieved chunks")
        doc = documents[hit["document_id"]]
        pages = pages_for_span(doc, hit["char_start"], hit["char_end"])
        require(hit["source"] == doc.source and hit["text"] == doc.text[hit["char_start"]:hit["char_end"]]
                and hit["page_numbers"] == [p.page_number for p in pages], "Retrieved source provenance differs")
    require(hits == sorted(hits, key=lambda h: (-h["score"], h["chunk_id"])), "Ranking/tie policy differs")
    require(retrieval["chunk_ids_at_k"] == {str(k): [h["chunk_id"] for h in hits[:k]] for k in settings["k_values"]}, "Top-K identity differs")
    if context is not None:
        rebuilt = assemble_context(hits, documents, tokens, settings["context_content_tokens"])
        require(all(context[k] == value for k, value in rebuilt.items()), "Saved generation context differs from source packing")


def load_evaluation_inputs(retrieval_run, generation_run, corpus, tokens, split="pilot"):
    require(split in ("pilot", "test"), "Unsupported evaluation split")
    retrieval_manifest = verify_manifest(retrieval_run)
    generation_manifest = verify_manifest(generation_run)
    require(retrieval_manifest["stage"] == "phase9_retrieval", "Expected a retrieval run")
    require(generation_manifest["stage"] == "phase10_generation", "Expected a generation run")
    for manifest in (retrieval_manifest, generation_manifest):
        require("run_config.json" in manifest["artifact_sha256"], "Unhashed source configuration")
    retrieval_config = read_json(retrieval_run / "run_config.json")
    generation_config = read_json(generation_run / "run_config.json")
    config_hash = retrieval_config["config_hash"]
    require(digest_json({k: v for k, v in retrieval_config.items() if k != "config_hash"}) == config_hash
            == retrieval_manifest["config_hash"], "Retrieval config hash differs")
    require(digest_json(generation_config) == generation_manifest["config_hash"], "Generation config hash differs")
    require(generation_config["corpus_hash"] == retrieval_config["corpus_hash"] == corpus.corpus_hash, "Corpus differs")
    require(generation_config["retrieval_manifest_sha256"] == file_hash(retrieval_run / "stage_manifest.json"), "Generation used a different retrieval run")
    require(within(ROOT, generation_config["settings"]["retrieval_run"]) == retrieval_run, "Generation source path differs")
    settings = retrieval_config["settings"]
    pilot = within(ROOT, settings["pilot"])
    benchmark_path = within(ROOT, settings["benchmark_manifest"])
    for path in (pilot, benchmark_path):
        require(retrieval_config["input_sha256"][path.relative_to(ROOT).as_posix()] == file_hash(path), "Benchmark changed since retrieval")
    validation = validate_directory(benchmark_path.parent, corpus)
    require(validation["valid"], f"Invalid benchmark: {validation['errors']}")
    if split == "pilot":
        question_path = pilot
    else:
        # Final test scoring is only permitted once the benchmark has been
        # frozen: validate_directory already refuses to call a benchmark
        # valid if its artifacts drifted from frozen_snapshot.json, but we
        # also require the snapshot to exist at all, so a never-frozen
        # benchmark cannot be scored as "final".
        frozen_path = benchmark_path.parent / "frozen_snapshot.json"
        require(frozen_path.exists(), "Final test scoring requires a frozen benchmark (run validate_benchmark --freeze first)")
        benchmark_manifest = read_json(benchmark_path)
        test_name = benchmark_manifest.get("files", {}).get("test")
        require(isinstance(test_name, str) and test_name, "No test split file declared in the frozen benchmark manifest")
        question_path = within(benchmark_path.parent, test_name)
    questions = read_rows(question_path)
    require(questions and all(q["split"] == split for q in questions),
            f"Expected every question in {question_path.name} to carry split={split!r}")
    question_map = {q["question_id"]: q for q in questions}
    want_ids = set(question_map)
    documents = {d.document_id: d for d in corpus.documents}
    traces = {}
    for method in METHODS:
        source = retrieval_run / method / "variant_001"
        generated = generation_run / method / "variant_001"
        for manifest, names in ((retrieval_manifest, ("retrieval.jsonl", "generation_contexts.jsonl")),
                                (generation_manifest, ("requests.jsonl", "generation_contexts.jsonl", "answers.jsonl"))):
            for name in names:
                require(f"{method}/variant_001/{name}" in manifest["artifact_sha256"], "Unhashed evaluation input")

        def scoped(path):
            # The saved traces cover every pilot question; restrict to the
            # split being scored (a no-op when split == "pilot", since
            # want_ids then equals every question in the trace).
            return [row for row in read_rows(path) if row["question_id"] in want_ids]

        retrieval = by_question(scoped(source / "retrieval.jsonl"), question_map, method)
        contexts = by_question(scoped(source / "generation_contexts.jsonl"), question_map, method)
        saved_contexts = by_question(scoped(generated / "generation_contexts.jsonl"), question_map, method)
        require(saved_contexts == contexts, "Generation evidence differs from retrieved contexts")
        requests = by_question(scoped(generated / "requests.jsonl"), question_map, method)
        answers = by_question(scoped(generated / "answers.jsonl"), question_map, method)
        for qid, question in question_map.items():
            validate_trace(question, retrieval.get(qid), contexts.get(qid), documents, tokens, settings,
                           corpus.corpus_hash, config_hash, method)
        for qid, request in requests.items():
            require(qid in contexts, "Request has no source context")
            require(request["context_hash"] == digest_json(contexts[qid]), "Request context hash differs")
            require(request["request_key"] == digest_json(request["identity"]), "Request identity hash differs")
            require(request["identity"]["trial_id"] == generation_config["trial_id"], "Generation trials differ")
            require(request["identity"]["payload"]["model"] == generation_config["settings"]["model_id"], "Generation models differ")
            payload = request["identity"]["payload"]
            for key in ("temperature", "top_p", "seed"):
                require(payload[key] == generation_config["settings"][key], f"Generation setting differs: {key}")
            require(payload["max_completion_tokens"] == generation_config["settings"]["max_output_tokens"], "Answer budget differs")
            expected_user = (f"QUESTION\n{contexts[qid]['question']}\n\nSOURCE EVIDENCE\n"
                             f"{contexts[qid]['rendered_context'] or '(No evidence was retrieved.)'}"
                             "\n\nAnswer the question using the evidence above.")
            require(len(payload["messages"]) == 2 and payload["messages"][1] == {"role": "user", "content": expected_user},
                    "Rendered generation evidence/question differs")
            prompt = within(ROOT, generation_config["settings"]["prompt"]).read_text()
            require(payload["messages"][0] == {"role": "system", "content": prompt}, "Generation system prompt differs")
        for qid, answer in answers.items():
            require(qid in requests, "Answer without request")
            require(answer["request_key"] == requests[qid]["request_key"] and
                    answer["context_hash"] == requests[qid]["context_hash"], "Answer lineage differs")
            require(answer["status"] in ("complete", "failed", "error"), "Unknown answer status")
            if answer["status"] == "complete":
                require(isinstance(answer["response"]["text"], str), "Missing answer text")
        traces[method] = {"retrieval": retrieval, "contexts": contexts, "answers": answers, "requests": requests}
    return questions, traces, validation, read_json(benchmark_path), retrieval_config, generation_config
