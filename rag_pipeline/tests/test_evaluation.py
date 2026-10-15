"""Evaluation fixtures test research definitions, missingness and provenance."""

from copy import deepcopy
import hashlib
from types import SimpleNamespace
import unittest

from rag_pipeline.evaluation.answer_metrics import citation_checks, review_template, score_answer
from rag_pipeline.evaluation.compare_runs import aggregate, paired_comparisons, question_clusters
from rag_pipeline.evaluation.inputs import by_question, validate_trace
from rag_pipeline.evaluation.ragas_adapter import describe, evaluate_samples, sample
from rag_pipeline.evaluation.retrieval_metrics import score_retrieval
from rag_pipeline.evaluation.run import run_evaluation


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.docs = {"d": SimpleNamespace(text="Rule தமிழ். Other rule."),
                     "v2": SimpleNamespace(text="Rule தமிழ். Other rule.")}
        self.q = {"question_id": "q1", "question": "Which rule?", "reference_answer": "Rule தமிழ்.",
                  "answerability": "answerable", "category": "definition", "group_id": "g1",
                  "review": {"status": "pending"}, "rubric": ["State the rule", "Name the subject"],
                  "evidence_units": [{"unit_id": "u1", "alternatives": [self.span(0, 11)]}]}
        self.context = {"pieces": [dict(self.span(0, 11), text="Rule தமிழ்.", citation_id="S1")],
                        "evidence_content_tokens": 5}
        self.answer = {"status": "complete", "response": {"text": "Rule [S1]."}}

    def span(self, start, end, doc="d"):
        return {"document_id": doc, "char_start": start, "char_end": end}

    def retrieval(self, spans):
        return {"hits": [dict(span, rank=i+1) for i, span in enumerate(spans)]}

    def score(self, spans):
        return score_retrieval(self.q, self.retrieval(spans), self.context, self.docs)

    def test_joint_coverage_is_recall_but_not_single_chunk_mrr(self):
        result = self.score([self.span(0, 5), self.span(5, 11)])
        self.assertEqual(result["metrics"]["recall_at_1"], 0)
        self.assertEqual(result["metrics"]["recall_at_3"], 1)
        self.assertEqual(result["metrics"]["mrr_at_3"], 0)
        result = self.score([self.span(12, 17), self.span(0, 11)])
        self.assertEqual(result["metrics"]["mrr_at_3"], .5)

    def test_unicode_and_whitespace_gaps_and_wrong_version(self):
        self.assertEqual(self.score([self.span(0, 4), self.span(5, 11)])["metrics"]["recall_at_3"], 1)
        self.assertEqual(self.score([self.span(0, 5), self.span(6, 11)])["metrics"]["recall_at_3"], 0)
        self.assertEqual(self.score([self.span(0, 11, "v2")])["metrics"]["recall_at_3"], 0)

    def test_alternatives_cannot_mix_halves_and_overlap_does_not_double_count(self):
        self.q["evidence_units"][0]["alternatives"].append(self.span(12, 22))
        self.assertEqual(self.score([self.span(0, 5), self.span(17, 22)])["metrics"]["recall_at_3"], 0)
        self.assertEqual(self.score([self.span(0, 11), self.span(0, 11)])["metrics"]["recall_at_3"], 1)

    def test_macro_recall_and_complete_have_different_meanings(self):
        self.q["evidence_units"].append({"unit_id": "u2", "alternatives": [self.span(12, 22)]})
        metrics = self.score([self.span(0, 11)])["metrics"]
        self.assertEqual(metrics["recall_at_1"], .5)
        self.assertEqual(metrics["complete_at_1"], 0)
        self.assertEqual(metrics["context_recall"], .5)

    def test_missing_pipeline_zero_but_nonanswerable_metrics_not_applicable(self):
        result = score_retrieval(self.q, None, None, self.docs)
        self.assertEqual(result["metrics"]["context_complete"], 0)
        self.assertEqual(result["retrieval_status"], "missing")
        for kind in ("unanswerable", "ambiguous", "extraction_diagnostic"):
            self.q["answerability"] = kind
            self.assertTrue(all(v is None for v in score_retrieval(self.q, None, None, self.docs)["metrics"].values()))

    def test_empty_evidence_is_not_vacuous_success(self):
        self.q["evidence_units"] = []
        with self.assertRaises(ValueError): self.score([])

    def approved_review(self):
        review = review_template(self.q, self.answer, self.context, "t1", "fixed_size")
        review.update(status="approved", reviewer="human_reviewer", notes="Synthetic judgment", facts_satisfied=[True, False],
                      contradiction=False, checkable_claims=2, supported_claims=1, cited_claims=1,
                      supported_cited_claims=1, uncited_claims=1, relevance=2, abstention=False, unsupported_answer=False)
        return review

    def test_correctness_requires_review_and_counts_facts(self):
        self.assertIsNone(score_answer(self.q, self.answer, self.context, "t1")["metrics"]["answer_correctness"])
        result = score_answer(self.q, self.answer, self.context, "t1", self.approved_review())
        self.assertEqual(result["metrics"]["answer_correctness"], .5)
        self.assertEqual(result["metrics"]["faithfulness"], .5)
        self.assertEqual(result["metrics"]["citation_support"], 1)
        self.assertEqual(result["metrics"]["fully_correct"], 0)

    def test_stale_or_invalid_review_is_missing_not_zero(self):
        review = self.approved_review()
        self.answer["response"]["text"] = "Changed answer"
        result = score_answer(self.q, self.answer, self.context, "t1", review)
        self.assertEqual(result["review_status"], "invalid")
        self.assertTrue(all(v is None for v in result["metrics"].values()))
        review = self.approved_review()
        review["supported_claims"] = 3
        self.assertEqual(score_answer(self.q, self.answer, self.context, "t1", review)["review_status"], "invalid")

    def test_no_claims_is_na_and_missing_answer_is_not_abstention(self):
        self.q["answerability"] = "unanswerable"
        review = self.approved_review()
        review.update(checkable_claims=0, supported_claims=0, cited_claims=0, supported_cited_claims=0,
                      uncited_claims=0, abstention=True)
        result = score_answer(self.q, self.answer, self.context, "t1", review)
        self.assertEqual(result["metrics"]["abstention"], 1)
        self.assertIsNone(result["metrics"]["faithfulness"])
        self.assertIsNone(score_answer(self.q, None, self.context, "t1")["metrics"]["abstention"])

    def test_answerable_abstention_zero_even_if_reviewer_marks_facts(self):
        review = self.approved_review()
        review.update(facts_satisfied=[True, True], abstention=True)
        metrics = score_answer(self.q, self.answer, self.context, "t1", review)["metrics"]
        self.assertEqual(metrics["answer_correctness"], 0)
        self.assertEqual(metrics["fully_correct"], 0)

    def test_citation_groups_check_existence_only(self):
        check = citation_checks("Claim [S1, S2]. More [S1].", self.context)
        self.assertEqual(check["cited_labels"], ["S1", "S2"])
        self.assertEqual(check["unknown_labels"], ["S2"])

    def test_clusters_connect_shared_sources_and_cross_source_questions(self):
        q2 = deepcopy(self.q)
        q2.update(question_id="q2", group_id="g2")
        q3 = deepcopy(self.q)
        q3.update(question_id="q3", group_id="g3")
        q3["evidence_units"][0]["alternatives"] = [self.span(0, 11, "v2")]
        q2["evidence_units"][0]["alternatives"].append(self.span(0, 11, "v2"))
        clusters = question_clusters([self.q, q2, q3], {"d": "family1", "v2": "family2"})
        self.assertEqual(len(set(clusters.values())), 1)

    def rows(self):
        return [{"method": m, "question_id": q, "answerability": "answerable", "category": "definition",
                 "retrieval": {"metrics": {"context_complete": value}}, "answer": {"metrics": {"answer_correctness": None}}}
                for m, values in (("a", [1, 0, 1]), ("b", [0, 0, 1])) for q, value in zip(("q1", "q2", "q3"), values)]

    def test_paired_bootstrap_repeats_and_retains_question_weighting(self):
        rows = self.rows()
        clusters = {"q1": "g1", "q2": "g1", "q3": "g2"}
        first = paired_comparisons(rows, clusters)
        self.assertEqual(first, paired_comparisons(list(reversed(rows)), clusters))
        self.assertAlmostEqual(first[0]["difference_a_minus_b"], 1/3)
        self.assertEqual(first[0]["clusters"], 2)
        self.assertIsNone(paired_comparisons(rows, dict.fromkeys(clusters, "one"))[0]["ci95"])
        with self.assertRaises(ValueError): paired_comparisons(rows[:-1], clusters)

    def test_aggregates_show_missing_review_denominators(self):
        summary = aggregate(self.rows())
        missing = next(r for r in summary if r["method"] == "a" and r["metric"] == "answer_correctness")
        self.assertEqual(missing["eligible_questions"], 3)
        self.assertEqual(missing["scored_questions"], 0)
        self.assertIsNone(missing["mean"])

    def test_duplicate_and_unknown_question_traces_rejected(self):
        row = {"question_id": "q1", "method": "a"}
        with self.assertRaises(ValueError): by_question([row, row], {"q1": self.q}, "a")
        with self.assertRaises(ValueError): by_question([row], {}, "a")
        with self.assertRaises(ValueError): by_question([row], {"q1": self.q}, "b")

    def test_final_scoring_cannot_be_requested_accidentally(self):
        with self.assertRaisesRegex(ValueError, "provisional"): run_evaluation("not-created")
        with self.assertRaisesRegex(ValueError, "provisional"):
            run_evaluation("not-created", provisional=True, final_test_scoring=True)

    def test_ragas_uses_supplied_context_and_explicit_clients(self):
        data = sample(self.q, self.answer, self.context)
        self.assertEqual(data["retrieved_contexts"], ["Rule தமிழ்."])
        with self.assertRaisesRegex(ValueError, "Explicit"):
            evaluate_samples([data], llm=None, embeddings=None, judge_identity={})
        names = [m["name"] for m in describe()["metrics"]]
        def fake_evaluate(dataset, **kwargs):
            self.assertFalse(kwargs["raise_exceptions"])
            self.assertIsNotNone(kwargs["llm"])
            self.assertEqual(len(dataset), 1)
            return SimpleNamespace(scores=[{name: float("nan") if i == 0 else .5 for i, name in enumerate(names)}])
        result = evaluate_samples([data], llm=object(), embeddings=object(),
            judge_identity={"provider": "stub", "model": "stub", "decoding": {"seed": 42}, "embeddings": "stub"},
            evaluate_fn=fake_evaluate)
        self.assertEqual(result["status"], "partial")
        self.assertIsNone(result["rows"][0]["scores"][names[0]])
        self.assertEqual(result["rows"][0]["scores"][names[1]], .5)

    def test_trace_rejects_wrong_pages_filters_and_context_edits(self):
        from rag_pipeline.common.provenance import Document, Page, eligible_regions
        from rag_pipeline.retrieval.assemble_context import assemble_context
        from rag_pipeline.tests.test_fixed_size import FixedSizeTests
        FixedSizeTests.setUp(self)
        pages = (Page(1, "1", 0, 4, True, (), (), False),)
        doc = Document("d", "source.pdf", "member", "rule", "hash", "unused", "hash", pages, eligible_regions(pages))
        docs = {"d": doc}
        settings = {"k_values": [1, 3, 5, 10], "context_content_tokens": 10,
                    "similarity": "normalized_inner_product", "tie_policy": "score_descending_chunk_id_ascending"}
        common = {"question": self.q["question"], "question_sha256": hashlib.sha256(self.q["question"].encode()).hexdigest(),
                  "corpus_hash": "corpus", "retrieval_config_hash": "config"}
        hit = {**self.span(0, 4), "text": "rule", "source": "source.pdf", "member": "member", "page_numbers": [1],
               "rank": 1, "chunk_id": "chunk", "method": "fixed_size", "corpus_hash": "corpus", "score": .5}
        retrieval = common | {k: settings[k] for k in ("k_values", "similarity", "tie_policy")}
        retrieval.update(filters=None, hits=[hit], chunk_ids_at_k={str(k): ["chunk"] for k in settings["k_values"]})
        context = assemble_context([hit], docs, self.tokens, 10) | common
        def validate(r, c):
            validate_trace(self.q, r, c, docs, self.tokens, settings, "corpus", "config", "fixed_size")
        validate(retrieval, context)
        edited = deepcopy(context)
        edited["pieces"][0]["text"] = "fake"
        with self.assertRaisesRegex(ValueError, "context"): validate(retrieval, edited)
        edited = deepcopy(retrieval)
        edited["hits"][0]["page_numbers"] = [2]
        with self.assertRaisesRegex(ValueError, "provenance"): validate(edited, context)
        edited = deepcopy(retrieval)
        edited["filters"] = {"gold_source": "d"}
        with self.assertRaisesRegex(ValueError, "whole-corpus"): validate(edited, context)

    def test_ragas_failure_has_explicit_missing_rows_and_no_secret_error_text(self):
        def fail(*args, **kwargs):
            raise RuntimeError("secret provider response")
        result = evaluate_samples([sample(self.q, self.answer, self.context)], llm=object(), embeddings=object(),
            judge_identity={"provider": "stub", "model": "stub", "decoding": {"seed": 42}, "embeddings": "stub"},
            evaluate_fn=fail)
        self.assertEqual(result["status"], "failed")
        self.assertNotIn("secret provider", str(result))
        self.assertTrue(all(v is None for v in result["rows"][0]["scores"].values()))


if __name__ == "__main__":
    unittest.main()
