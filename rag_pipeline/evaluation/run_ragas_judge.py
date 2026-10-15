"""Explicit, opt-in RAGAS judge pass over a saved generation run.

This makes real, paid judge-model and embedding-model API calls. It is never
invoked by `evaluate`; it only runs when a caller deliberately asks for it,
and only scores answerable questions (RAGAS reference metrics here require a
reference answer, which only answerable benchmark rows carry).
"""

import json
import os
from pathlib import Path

from dotenv import dotenv_values

from ..common.config import ROOT, load_config
from ..common.load_corpus import require
from ..common.run_manifest import create_run, environment, finish_run, write_json
from ..common.tokenization import load_tokenizer
from ..embedding.encode_chunks import atomic_jsonl
from .inputs import load_evaluation_inputs
from .ragas_adapter import evaluate_samples, sample

DEFAULT_RETRIEVAL = ROOT / "data/outputs/rag/phase9_retrieval_pilot_20260927"
DEFAULT_GENERATION = ROOT / "data/outputs/rag/phase10_generation_pilot_20260927"


def build_judge(judge_model, embedding_model, judge_temperature):
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    key = os.environ.get("OPENAI_API_KEY") or dotenv_values(ROOT / ".env").get("OPENAI_API_KEY")
    require(bool(key), "OPENAI_API_KEY is missing")
    llm = LangchainLLMWrapper(ChatOpenAI(model=judge_model, api_key=key, temperature=judge_temperature))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=embedding_model, api_key=key))
    judge_identity = {"provider": "openai", "model": judge_model,
                      "decoding": {"temperature": judge_temperature}, "embeddings": embedding_model}
    return llm, embeddings, judge_identity


def run_ragas_judge(run_id, *, retrieval_run=DEFAULT_RETRIEVAL, generation_run=DEFAULT_GENERATION,
                    judge_model="gpt-4o-mini", embedding_model="text-embedding-3-small",
                    judge_temperature=0.0, question_ids=None, methods=None, max_workers=16, split="pilot"):
    retrieval_run, generation_run = Path(retrieval_run).resolve(), Path(generation_run).resolve()
    config = load_config(ROOT / "rag_pipeline/config.json")
    from ..common.load_corpus import load_corpus
    corpus = load_corpus(ROOT / config.corpus_dir, config)
    tokens = load_tokenizer(ROOT / "rag_pipeline/configs/models.json")
    questions, traces, validation, benchmark, retrieval_config, generation_config = load_evaluation_inputs(
        retrieval_run, generation_run, corpus, tokens, split=split)

    question_map = {q["question_id"]: q for q in questions}
    answerable_ids = {qid for qid, q in question_map.items() if q["answerability"] == "answerable"}
    if question_ids is not None:
        answerable_ids &= set(question_ids)
    require(answerable_ids, "No answerable questions selected")
    selected_methods = methods or sorted(traces)
    require(all(m in traces for m in selected_methods), "Unknown method selected")

    llm, embeddings, judge_identity = build_judge(judge_model, embedding_model, judge_temperature)

    run, manifest = create_run(ROOT, config, run_id)
    manifest.update(stage="phase11_ragas_judge", corpus_hash=corpus.corpus_hash)
    write_json(run / "stage_manifest.json", manifest)
    results_by_method = {}
    try:
        write_json(run / "environment.json", environment(ROOT))
        write_json(run / "run_config.json", {
            "retrieval_run": str(retrieval_run.relative_to(ROOT)), "generation_run": str(generation_run.relative_to(ROOT)),
            "split": split, "judge_identity": judge_identity, "answerable_question_ids": sorted(answerable_ids),
            "methods": selected_methods})
        for method in selected_methods:
            trace = traces[method]
            samples, sample_qids = [], []
            for qid in sorted(answerable_ids):
                answer = trace["answers"].get(qid)
                context = trace["contexts"].get(qid)
                if answer is None or answer["status"] != "complete" or context is None:
                    continue
                samples.append(sample(question_map[qid], answer, context))
                sample_qids.append(qid)
            if not samples:
                results_by_method[method] = {"status": "skipped", "scored": 0}
                continue
            result = evaluate_samples(samples, llm=llm, embeddings=embeddings, judge_identity=judge_identity,
                                      max_workers=max_workers)
            rows = [{"question_id": qid} | row for qid, row in zip(sample_qids, result["rows"])]
            variant = run / method
            variant.mkdir(parents=True)
            atomic_jsonl(variant / "ragas_scores.jsonl", rows)
            write_json(variant / "ragas_metadata.json", {k: v for k, v in result.items() if k != "rows"})
            metric_names = list(rows[0]["scores"]) if rows else []
            means = {}
            for name in metric_names:
                values = [r["scores"][name] for r in rows if r["scores"][name] is not None]
                means[name] = sum(values) / len(values) if values else None
            results_by_method[method] = {"status": result["status"], "scored": len(rows), "mean_scores": means}
        write_json(run / "summary.json", {"judge_identity": judge_identity, "split": split,
                                          "answerable_questions_selected": len(answerable_ids),
                                          "methods": results_by_method})
        finish_run(run, manifest)
    except BaseException as exc:
        finish_run(run, manifest, error=f"{type(exc).__name__}: {exc}")
        raise
    print(json.dumps({"output": str(run), "methods": results_by_method}))
    return run
