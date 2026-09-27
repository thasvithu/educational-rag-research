"""Audit saved indexes, rankings, contexts and optional cached repeats."""

import argparse
from collections import defaultdict
import hashlib
import json

import numpy as np

from ..common.config import ROOT, load_config, validate_run_id, within
from ..common.load_corpus import digest_json, file_hash, load_corpus, require
from ..common.provenance import pages_for_span
from ..common.run_manifest import write_json
from ..common.tokenization import load_tokenizer
from ..embedding.inputs import verify_manifest
from .assemble_context import assemble_context
from .build_index import load_index
from .inputs import METHODS


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def evidence_present(alternative, passages, documents):
    """Independent character-by-character coverage check for short gold spans."""
    source = documents[alternative["document_id"]].text
    relevant = [p for p in passages if p["document_id"] == alternative["document_id"]]
    return all(source[i].isspace() or any(p["char_start"] <= i < p["char_end"] for p in relevant)
               for i in range(alternative["char_start"], alternative["char_end"]))


def audit(run_id, repeat_id=None):
    validate_run_id(run_id)
    run = ROOT / "data/outputs/rag" / run_id
    verify_manifest(run)
    effective = json.loads((run / "run_config.json").read_text())
    require(digest_json({k: v for k, v in effective.items() if k != "config_hash"}) == effective["config_hash"], "Run configuration hash differs")
    corpus = load_corpus(ROOT / "final_data", load_config(ROOT / "rag_pipeline/config.json"))
    require(corpus.corpus_hash == effective["corpus_hash"], "Run corpus differs")
    documents = {d.document_id: d for d in corpus.documents}
    tokens = load_tokenizer()
    for name, expected in effective["input_sha256"].items():
        require(file_hash(within(ROOT, name)) == expected, f"Input changed: {name}")
    questions = read_rows(within(ROOT, effective["settings"]["pilot"]))
    queries = read_rows(run / "queries.jsonl")
    query_map = read_rows(run / "query_embedding_rows.jsonl")
    vectors = np.load(run / "query_embeddings.npy", allow_pickle=False)
    require(len(queries) == len(query_map) == len(questions) and vectors.shape == (len(queries), 512), "Query rows differ")
    require(vectors.dtype == np.dtype("float32") and np.isfinite(vectors).all()
            and np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4, rtol=0), "Invalid query vectors")
    query_config = json.loads((run / "query_embedding_config.json").read_text())
    query_hash = digest_json(query_config["identity"])
    require(query_hash == query_config["embedding_hash"] == digest_json(effective["query_identity"]), "Query identity differs")
    for i, (question, query, mapping) in enumerate(zip(questions, queries, query_map)):
        expected = {"question_id": question["question_id"], "question": question["question"],
                    "question_sha256": hashlib.sha256(question["question"].encode()).hexdigest()}
        require(query == expected and all(mapping[k] == v for k, v in expected.items()), "Query contains changed text or labels")
        require(mapping["row_index"] == i and mapping["query_embedding_hash"] == query_hash, "Wrong query row")
        require(mapping["vector_sha256"] == hashlib.sha256(vectors[i].astype("<f4").tobytes()).hexdigest(), "Query vector checksum differs")
        require(mapping["input_token_count"] == tokens.count(query["question"]), "Question token count differs")
    maximum_error, totals = 0.0, {}
    for method in METHODS:
        variant = run / method / "variant_001"
        verify_manifest(variant)
        store, chunks = load_index(variant)
        source = within(ROOT, effective["settings"]["inputs"][method])
        original = np.load(source / "embeddings.npy", allow_pickle=False)
        require(chunks == read_rows(source / "chunks.jsonl"), "Index source chunks changed")
        require(np.array_equal(store.index.reconstruct_n(0, len(chunks)), original), "Index contains wrong vectors")
        positions = {r["chunk_id"]: i for i, r in enumerate(chunks)}
        retrieval, contexts, diagnoses = [read_rows(variant / name) for name in
                                         ("retrieval.jsonl", "generation_contexts.jsonl", "pilot_diagnostics.jsonl")]
        require(len(retrieval) == len(contexts) == len(diagnoses) == len(questions), "Missing method queries")
        double_vectors = original.astype(float)
        for qi, (result, context, diagnosis, question) in enumerate(zip(retrieval, contexts, diagnoses, questions)):
            require(result["question_id"] == context["question_id"] == diagnosis["question_id"] == question["question_id"], "Reordered questions")
            require(result["question"] == question["question"] and result["query_embedding_hash"] == query_hash
                    and result["filters"] is None and result["method"] == method, "Search query, method or filters differ")
            hits = result["hits"]
            require(len(hits) == min(10, len(chunks)) and len({h['chunk_id'] for h in hits}) == len(hits), "Invalid hit count/duplicates")
            # NumPy float64 dot products provide an independent numerical oracle.
            # FP32 FAISS scores may differ slightly, so test a stated tolerance.
            scores = double_vectors @ vectors[qi].astype(float)
            indices = []
            for rank, hit in enumerate(hits, 1):
                row_index = positions[hit["chunk_id"]]
                indices.append(row_index)
                require(hit == {"rank": rank, "score": hit["score"], **chunks[row_index]}, "Hit text/provenance differs")
                error = abs(float(scores[row_index])-hit["score"])
                maximum_error = max(maximum_error, error)
                require(error <= 2e-6, "FAISS score disagrees with direct dot product")
            require(hits == sorted(hits, key=lambda h: (-h['score'], h['chunk_id'])), "Ranking/tie policy differs")
            require(all(scores[a] >= scores[b]-2e-6 for a, b in zip(indices, indices[1:])), "Oracle ranking inversion")
            omitted = np.ones(len(chunks), dtype=bool); omitted[indices] = False
            require(not omitted.any() or float(scores[omitted].max()) <= scores[indices[-1]]+2e-6, "A better chunk was omitted")
            for k in effective["settings"]["k_values"]:
                require(result["chunk_ids_at_k"][str(k)] == [h["chunk_id"] for h in hits[:k]], "K results differ")
            replay = assemble_context(hits, documents, tokens, effective["settings"]["context_content_tokens"])
            require(all(context[key] == value for key, value in replay.items()), "Context policy replay differs")
            used, seen = 0, defaultdict(list)
            for piece in context["pieces"]:
                doc = documents[piece["document_id"]]
                start, end = piece["char_start"], piece["char_end"]
                parent = chunks[positions[piece["chunk_id"]]]
                require(parent["document_id"] == doc.document_id and parent["char_start"] <= start < end <= parent["char_end"], "Piece escaped retrieved parent")
                require(piece["text"] == doc.text[start:end] and piece["page_numbers"] ==
                        [p.page_number for p in pages_for_span(doc, start, end)], "Piece source/pages differ")
                require(all(end <= a or start >= b for a, b in seen[doc.document_id]), "Duplicate overlap in context")
                seen[doc.document_id].append((start, end))
                count = tokens.count(piece["text"], special_tokens=False)
                require(count == piece["content_token_count"], "Evidence tokens differ")
                used += count
            require(used == context["evidence_content_tokens"] <= context["evidence_budget"], "Evidence budget exceeded")
            expected_units = []
            if question["answerability"] == "answerable":
                for unit in question["evidence_units"]:
                    alternatives = unit["alternatives"]
                    expected_units.append({"unit_id": unit["unit_id"],
                        "first_single_chunk_rank": next((h["rank"] for h in hits if any(evidence_present(a, [h], documents) for a in alternatives)), None),
                        "covered_at_k": {str(k): any(evidence_present(a, hits[:k], documents) for a in alternatives)
                                         for k in effective["settings"]["k_values"]},
                        "covered_in_context": any(evidence_present(a, context["pieces"], documents) for a in alternatives)})
            require(diagnosis["units"] == expected_units, "Pilot evidence diagnostic differs")
        totals[method] = {"vectors": len(chunks), "queries": len(queries), "index_reload_and_contexts_valid": True}
    repeat = None
    if repeat_id:
        validate_run_id(repeat_id)
        other = run.parent / repeat_id
        verify_manifest(other)
        for name in ("queries.jsonl", "query_embeddings.npy", "query_embedding_rows.jsonl", "inspection_report.md"):
            require(file_hash(run / name) == file_hash(other / name), f"Repeat differs: {name}")
        for method in METHODS:
            for name in ("index.faiss", "index_rows.jsonl", "chunks.jsonl", "retrieval.jsonl", "generation_contexts.jsonl", "pilot_diagnostics.jsonl"):
                relative = f"{method}/variant_001/{name}"
                require(file_hash(run / relative) == file_hash(other / relative), f"Repeat differs: {relative}")
        repeat = {"run_id": repeat_id, "byte_identical_queries_indexes_rankings_and_contexts": True}
    phase1 = json.loads((run.parent / "phase1_clean_environment_20260919/corpus_snapshot.json").read_text())
    require(corpus.snapshot["files"] == phase1["files"], "Prepared corpus changed since Phase 1")
    report = {"passed": True, "run_id": run_id, "methods": totals,
              "max_faiss_vs_float64_score_error": maximum_error, "score_tolerance": 2e-6,
              "no_reference_filters": True, "source_spans_and_pages_valid": True, "context_budgets_and_overlap_valid": True,
              "pilot_evidence_recomputed": True, "prepared_files_unchanged": len(corpus.snapshot["files"]),
              "source_artifacts_unchanged": len(effective["input_sha256"]), "repeat": repeat}
    destination = run.parent / (run_id + "_audit.json")
    require(not destination.exists(), "Audit report already exists; preserve completed reports")
    write_json(destination, report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repeat-id")
    args = parser.parse_args()
    audit(args.run_id, args.repeat_id)
