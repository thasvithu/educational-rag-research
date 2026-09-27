"""Build five indexes and preserve the shared pilot's retrieval/context traces."""

import hashlib
import json
from pathlib import Path
import resource
import time

import faiss
import numpy as np

from ..common.config import ROOT, load_config, within
from ..common.load_corpus import digest_json, file_hash, load_corpus, require
from ..common.run_manifest import create_run, environment, finish_run, write_json
from ..common.tokenization import load_tokenizer
from ..embedding.encode_chunks import atomic_jsonl
from ..embedding.vector_cache import ChunkEmbeddings, embedding_identity, model_factory
from ..evaluation.validate_benchmark import validate_records
from .assemble_context import assemble_context
from .build_index import build_index
from .diagnostics import diagnose, write_report
from .inputs import METHODS, load_inputs, load_settings
from .search import search


def run_retrieval(run_id, *, config_path=ROOT / "rag_pipeline/config.json",
                  retrieval_config=ROOT / "rag_pipeline/configs/retrieval.json",
                  model_config=ROOT / "rag_pipeline/configs/models.json", cache_only=False):
    started = time.perf_counter()
    config, tokens = load_config(Path(config_path)), load_tokenizer(Path(model_config))
    settings = load_settings(retrieval_config)
    independent = embedding_identity(model_config, tokens, batch_size=1)
    corpus = load_corpus(ROOT / config.corpus_dir, config)
    documents = {d.document_id: d for d in corpus.documents}
    inputs, input_hashes = load_inputs(settings, corpus, tokens, independent)
    pilot = within(ROOT, settings["pilot"])
    benchmark_manifest = within(ROOT, settings["benchmark_manifest"])
    questions = [json.loads(line) for line in pilot.read_text().splitlines() if line.strip()]
    benchmark = validate_records(questions, corpus, json.loads(benchmark_manifest.read_text()))
    require(benchmark["valid"], f"Invalid pilot: {benchmark['errors']}")
    require(bool(questions) and all(q["split"] == "pilot" for q in questions), "This development command accepts pilot questions only")
    for path in (pilot, benchmark_manifest):
        input_hashes[path.relative_to(ROOT).as_posix()] = file_hash(path)
    # Reference labels stay outside the retrieval interface. Only question text
    # is encoded; even the reference university/year scope is never a filter.
    queries = [{"question_id": q["question_id"], "question": q["question"],
                "question_sha256": hashlib.sha256(q["question"].encode()).hexdigest()} for q in questions]
    for query in queries:
        require(tokens.special_tokens < tokens.count(query["question"]) <= tokens.max_input_tokens, "Question overflow or no content")
    query_identity = independent | {"purpose": "retrieval_query_embedding", "batch_policy": "one_query_at_a_time"}
    effective = {"schema_version": 1, "settings": settings, "corpus_hash": corpus.corpus_hash,
                 "query_identity": query_identity, "input_sha256": input_hashes,
                 "retrieval_code_sha256": {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))}}
    config_hash = digest_json(effective)
    effective["config_hash"] = config_hash
    preflight_seconds = time.perf_counter()-started
    run, manifest = create_run(ROOT, config, run_id)
    manifest.update(stage="phase9_retrieval", corpus_hash=corpus.corpus_hash, config_hash=config_hash)
    write_json(run / "stage_manifest.json", manifest)
    encoder = None
    try:
        write_json(run / "run_config.json", effective)
        write_json(run / "environment.json", environment(ROOT))
        write_json(run / "corpus_snapshot.json", corpus.snapshot)
        write_json(run / "pilot_validation.json", benchmark)
        atomic_jsonl(run / "queries.jsonl", queries)
        cache = ROOT / config.output_root / "query_embedding_cache" / (digest_json(query_identity) + ".sqlite3")
        encoder = ChunkEmbeddings(cache, query_identity, tokens,
                                  model_factory(model_config, query_identity["embedding"]["device"], 1), cache_only=cache_only)
        query_vectors, mapping, timings = [], [], {"query_encoding_seconds": [], "methods": {}}
        for index, query in enumerate(queries):
            tick = time.perf_counter()
            vector = encoder.embed_query(query["question"])
            timings["query_encoding_seconds"].append(time.perf_counter()-tick)
            query_vectors.append(vector)
            mapping.append({"row_index": index, **query, "query_embedding_hash": encoder.identity_hash,
                            "input_token_count": tokens.count(query["question"]), **encoder.last_entries[0]})
        query_vectors = np.asarray(query_vectors, dtype="<f4")
        np.save(run / "query_embeddings.npy", query_vectors, allow_pickle=False)
        atomic_jsonl(run / "query_embedding_rows.jsonl", mapping)
        write_json(run / "query_embedding_config.json", {"identity": query_identity, "embedding_hash": encoder.identity_hash,
                                                        "cache_path": cache.relative_to(ROOT).as_posix()})
        faiss.omp_set_num_threads(settings["faiss_threads"])
        all_retrieval, all_contexts, all_diagnostics, summaries = {}, {}, {}, {}
        for method in METHODS:
            source = inputs[method]
            variant = run / method / "variant_001"
            variant.mkdir(parents=True)
            tick = time.perf_counter()
            store = build_index(variant, source["chunks"], source["vectors"], corpus_hash=corpus.corpus_hash,
                                embedding_hash=source["embedding_hash"], input_hashes=source["hashes"])
            method_timing = {"build_and_reload_seconds": time.perf_counter()-tick, "search_seconds": [], "context_seconds": []}
            retrieval, contexts = [], []
            for query, vector in zip(queries, query_vectors):
                tick = time.perf_counter()
                hits = search(store, vector, max(settings["k_values"]))
                method_timing["search_seconds"].append(time.perf_counter()-tick)
                common = {**query, "method": method, "corpus_hash": corpus.corpus_hash, "retrieval_config_hash": config_hash}
                retrieval.append({"schema_version": 1, **common, "query_embedding_hash": encoder.identity_hash,
                                  "document_embedding_hash": source["embedding_hash"], "filters": None,
                                  "similarity": settings["similarity"], "tie_policy": settings["tie_policy"],
                                  "k_values": settings["k_values"], "hits": hits,
                                  "chunk_ids_at_k": {str(k): [h["chunk_id"] for h in hits[:k]] for k in settings["k_values"]}})
                tick = time.perf_counter()
                context = assemble_context(hits, documents, tokens, settings["context_content_tokens"])
                contexts.append({**context, **common})
                method_timing["context_seconds"].append(time.perf_counter()-tick)
            # Rankings and context selection are complete before examining labels.
            diagnostics = [diagnose(q, r["hits"], c, documents, settings["k_values"])
                           for q, r, c in zip(questions, retrieval, contexts)]
            atomic_jsonl(variant / "retrieval.jsonl", retrieval)
            atomic_jsonl(variant / "generation_contexts.jsonl", contexts)
            atomic_jsonl(variant / "pilot_diagnostics.jsonl", diagnostics)
            summaries[method] = {"indexed_chunks": len(source["chunks"]), "indexed_documents": len({r['document_id'] for r in source['chunks']}),
                                 "queries": len(queries), "index_bytes": (variant / "index.faiss").stat().st_size,
                                 "mean_evidence_tokens": float(np.mean([c['evidence_content_tokens'] for c in contexts])),
                                 "budget_skipped_candidates": sum(d['status'] == 'skipped_budget' for c in contexts for d in c['decisions']),
                                 "overlap_characters_removed": sum(d['overlap_characters_removed'] for c in contexts for d in c['decisions'])}
            timings["methods"][method] = method_timing
            all_retrieval[method], all_contexts[method], all_diagnostics[method] = retrieval, contexts, diagnostics
            finish_run(variant, {"schema_version": 1, "method": method, "stage": "phase9_retrieval", "run_id": run_id,
                                 "corpus_hash": corpus.corpus_hash, "config_hash": config_hash})
            print(f"{method}: {len(source['chunks']):,} indexed chunks; {len(queries)} pilot queries", flush=True)
            del store
        write_report(run / "inspection_report.md", questions, all_retrieval, all_contexts, all_diagnostics)
        for name, expected in corpus.snapshot["files"].items():
            require(file_hash(within(ROOT / config.corpus_dir, name)) == expected, f"Prepared corpus changed: {name}")
        for name, expected in input_hashes.items():
            require(file_hash(within(ROOT, name)) == expected, f"Source artifact changed: {name}")
        write_json(run / "timings.json", timings)
        summary = {"methods": summaries, "query_count": len(queries), "searches": len(queries)*len(METHODS),
                   "query_cache": encoder.counters(), "query_gpu_memory": encoder.memory(),
                   "preflight_seconds": preflight_seconds, "total_elapsed_seconds": time.perf_counter()-started,
                   "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                   "prepared_files_unchanged": len(corpus.snapshot["files"]), "input_artifacts_unchanged": len(input_hashes),
                   "human_reviewed_pilot_questions": benchmark["reviewed"], "pilot_only": True,
                   "note": "No answers generated or final benchmark scores. Search timing includes global tie sorting."}
        write_json(run / "retrieval_summary.json", summary)
        finish_run(run, manifest)
        print(json.dumps({"output": str(run), **summary}, indent=2))
    except BaseException as exc:
        finish_run(run, manifest, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if encoder is not None:
            encoder.close()
    return run
