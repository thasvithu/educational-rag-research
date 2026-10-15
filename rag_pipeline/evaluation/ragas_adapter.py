"""Optional RAGAS 0.2.6 integration. Callers must supply their own judge clients.

The offline evaluation command never calls this adapter's evaluate function.
No default OpenAI client, credentials lookup, or provider fallback is created.
"""

from importlib.metadata import version
import math

from ..common.load_corpus import digest_json, require


def metric_instances():
    require(version("ragas") == "0.2.6", "This adapter is validated for ragas==0.2.6")
    from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference, LLMContextRecall
    return [Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()]


def describe():
    """Capture installed metrics, required columns, and exact prompt definitions."""
    metrics = metric_instances()
    descriptions = []
    for metric in metrics:
        prompts = {}
        for name, prompt in metric.get_prompts().items():
            prompts[name] = {"instruction": prompt.instruction,
                             "examples": [[a.model_dump(), b.model_dump()] for a, b in prompt.examples],
                             "language": prompt.language,
                             "input_schema": prompt.input_model.model_json_schema(),
                             "output_schema": prompt.output_model.model_json_schema()}
        descriptions.append({"name": metric.name, "required_columns": {
            str(key): sorted(value) for key, value in metric._required_columns.items()},
            "prompts": prompts, "prompt_hash": digest_json(prompts)})
    return {"ragas_version": version("ragas"), "metrics": descriptions,
            "status": "not_run", "reason": "No judge calls in the offline provisional evaluation.",
            "note": "RAGAS context judgments are separate from exact source-span recall."}


def sample(question, answer, context):
    """Pass only evidence actually sent to generation, not all retrieved hits."""
    require(question["answerability"] == "answerable", "RAGAS reference metrics here require answerable questions")
    require(answer["status"] == "complete", "RAGAS requires a completed answer")
    return {"user_input": question["question"], "response": answer["response"]["text"],
            "reference": question["reference_answer"], "retrieved_contexts": [p["text"] for p in context["pieces"]]}


def evaluate_samples(samples, *, llm, embeddings, judge_identity, evaluate_fn=None, max_workers=1):
    """Explicit opt-in API for future judge runs; failures become missing values.

    judge_identity records provider/model, decoding and embedding configuration.
    The caller must save the result and provider usage and enforce its budget.
    max_workers stays 1 by default (safe under slow/rate-limited judge accounts);
    callers on a paid, high-rate-limit judge account may pass a higher value to
    run independent per-sample judge calls concurrently. Concurrency only
    affects wall-clock time, not any individual sample's judged output.
    """
    require(llm is not None and embeddings is not None, "Explicit judge and embedding clients required")
    require(isinstance(judge_identity, dict) and all(judge_identity.get(k) for k in
            ("provider", "model", "decoding", "embeddings")), "Complete judge identity required")
    require(type(max_workers) is int and max_workers >= 1, "max_workers must be a positive integer")
    from ragas import EvaluationDataset, evaluate
    from ragas.run_config import RunConfig
    metrics = metric_instances()
    metadata = describe() | {"judge_identity": judge_identity, "input_hash": digest_json(samples), "max_workers": max_workers}
    try:
        result = (evaluate_fn or evaluate)(EvaluationDataset.from_list(samples), metrics=metrics,
            llm=llm, embeddings=embeddings, raise_exceptions=False, show_progress=False,
            run_config=RunConfig(max_workers=max_workers, max_retries=1, seed=42))
        require(len(result.scores) == len(samples), "RAGAS row count differs")
        rows = []
        for raw in result.scores:
            scores, errors = {}, {}
            for metric in metrics:
                value = raw.get(metric.name)
                valid = isinstance(value, (int, float)) and math.isfinite(value)
                scores[metric.name] = float(value) if valid else None
                if not valid:
                    errors[metric.name] = "Missing or non-finite judge output"
            rows.append({"scores": scores, "errors": errors})
        metadata.update(status="complete" if all(not r["errors"] for r in rows) else "partial", rows=rows)
    except Exception as exc:
        # Do not serialize exception messages: provider errors can contain secrets.
        metadata.update(status="failed", error_type=type(exc).__name__, rows=[{
            "scores": {m.name: None for m in metrics}, "errors": {m.name: "Judge execution failed" for m in metrics}
        } for _ in samples])
    return metadata
