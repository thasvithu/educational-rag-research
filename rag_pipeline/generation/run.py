"""Prepare all pilot prompts, then run a small balanced hosted-model pilot."""

import json
from pathlib import Path

from ..common.config import ROOT, load_config, validate_run_id, within
from ..common.load_corpus import digest_json, file_hash, load_corpus, require
from ..common.provenance import pages_for_span, region_for_span
from ..common.run_manifest import create_run, environment, finish_run, now, write_json
from ..embedding.encode_chunks import atomic_jsonl
from ..embedding.inputs import verify_manifest
from ..retrieval.inputs import METHODS
from .answer import GeneratorTokenizer, inspect_answer, load_settings, prepare_request
from .client import OpenAIAnswerClient, ProviderFailure
from .state import AnswerStore, FreePlanLimiter, answer_once, locked_cache


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def load_contexts(settings, corpus):
    """Validate Phase 9 and exact evidence. Never open reference-answer files."""
    source = within(ROOT, settings['retrieval_run'])
    require(source.is_relative_to(ROOT / 'data/outputs/rag'), 'Expected a local retrieval artifact')
    manifest = verify_manifest(source)
    require(manifest['corpus_hash'] == corpus.corpus_hash, 'Retrieval corpus differs')
    queries = read_rows(source / 'queries.jsonl')
    require(len({q['question_id'] for q in queries}) == len(queries), 'Duplicate query ID')
    documents = {d.document_id: d for d in corpus.documents}
    contexts = []
    for method in METHODS:
        rows = read_rows(source / method / 'variant_001/generation_contexts.jsonl')
        require([r['question_id'] for r in rows] == [q['question_id'] for q in queries], 'Context/question order differs')
        for row, query in zip(rows, queries):
            require(row['method'] == method and row['corpus_hash'] == corpus.corpus_hash, 'Context method/corpus differs')
            require(all(row[k] == query[k] for k in ('question_id', 'question', 'question_sha256')), 'Context question differs')
            for p in row['pieces']:
                doc = documents[p['document_id']]
                start, end = p['char_start'], p['char_end']
                require(doc.text[start:end] == p['text'] and doc.source == p['source'], 'Evidence is not an exact source slice')
                region_for_span(doc, start, end)
                require([page.page_number for page in pages_for_span(doc, start, end)] == p['page_numbers'], 'Evidence pages differ')
            contexts.append(row)
    # Question-first ordering balances methods even if an external quota stops us.
    order = {q['question_id']: i for i, q in enumerate(queries)}
    contexts.sort(key=lambda r: (order[r['question_id']], METHODS.index(r['method'])))
    return contexts, source, manifest


def run_generation(run_id, *, generation_config=ROOT / 'rag_pipeline/configs/generation.json',
                   dry_run=False, cache_only=False, resume=False, all_pilot=False, trial_id='trial_001'):
    validate_run_id(run_id)
    validate_run_id(trial_id)
    require(not (dry_run and cache_only), 'Dry run and cache-only are separate modes')
    settings = load_settings(generation_config)
    config = load_config(ROOT / 'rag_pipeline/config.json')
    corpus = load_corpus(within(ROOT, config.corpus_dir), config)
    contexts, source, source_manifest = load_contexts(settings, corpus)
    tokens = GeneratorTokenizer(settings)
    prompt_path = within(ROOT, settings['prompt'])
    prompt = prompt_path.read_text()
    code = {p.relative_to(ROOT).as_posix(): file_hash(p)
            for p in sorted((ROOT / 'rag_pipeline/generation').rglob('*.py'))}
    code[settings['prompt']] = file_hash(prompt_path)
    implementation_hash = digest_json(code)
    requests = [prepare_request(c, settings, tokens, prompt, trial_id, implementation_hash) for c in contexts]
    wanted = {r['question_id'] for r in requests} if all_pilot else set(settings['pilot_question_ids'])
    require(wanted <= {r['question_id'] for r in requests}, 'Unknown pilot selection')
    selected = [r for r in requests if r['question_id'] in wanted]
    effective = {'settings': settings, 'trial_id': trial_id, 'question_ids': sorted(wanted),
                 'mode': 'prepare' if dry_run else 'answers', 'implementation_hash': implementation_hash,
                 'retrieval_manifest_sha256': file_hash(source / 'stage_manifest.json'),
                 'corpus_hash': corpus.corpus_hash, 'request_keys': [r['request_key'] for r in requests]}
    run = within(ROOT, f'{config.output_root}/{run_id}')
    cache = within(ROOT, f'{config.output_root}/generation_answer_cache_v1')
    with locked_cache(cache):
        if resume:
            manifest = json.loads((run / 'stage_manifest.json').read_text())
            require(manifest['status'] in ('failed', 'running'), 'Only failed/interrupted runs can resume')
            require(json.loads((run / 'run_config.json').read_text()) == effective, 'Resume inputs, code or settings changed')
            manifest.update(status='running', resumed_at_utc=now())
        else:
            run, manifest = create_run(ROOT, config, run_id)
            write_json(run / 'run_config.json', effective)
            write_json(run / 'environment.json', environment(ROOT))
            write_json(run / 'corpus_snapshot.json', corpus.snapshot)
        manifest.update(stage='phase10_generation', corpus_hash=corpus.corpus_hash, config_hash=digest_json(effective))
        write_json(run / 'stage_manifest.json', manifest)
        client = None
        answers = []
        try:
            for method in METHODS:
                variant = run / method / 'variant_001'
                variant.mkdir(parents=True, exist_ok=True)
                atomic_jsonl(variant / 'requests.jsonl', [r for r in requests if r['method'] == method])
                atomic_jsonl(variant / 'generation_contexts.jsonl', [c for c in contexts if c['method'] == method])
                if dry_run:
                    # Empty, hashed exports make unexecuted answers explicit to
                    # the offline evaluator without contacting the provider.
                    atomic_jsonl(variant / 'answers.jsonl', [])
            preparation = {'prepared_requests': len(requests), 'selected_requests': len(selected),
                           'estimated_prompt_tokens_all': sum(r['estimated_prompt_tokens'] for r in requests),
                           'reserved_tokens_all': sum(r['reserved_tokens'] for r in requests),
                           'reserved_tokens_selected': sum(r['reserved_tokens'] for r in selected),
                           'largest_prompt_tokens': max(r['estimated_prompt_tokens'] for r in requests),
                           'account_tier': settings['account_tier'], 'human_review_status': 'draft pilot; not final evaluation',
                           'rate_note': 'Local reservations cover this pipeline only; provider quotas also include other account use.'}
            write_json(run / 'preparation.json', preparation)
            print(json.dumps(preparation), flush=True)
            if not dry_run:
                store = AnswerStore(cache / 'answers')
                limiter = FreePlanLimiter(cache / 'quota_reservations.jsonl', settings)
                by_key = {(c['method'], c['question_id']): c for c in contexts}
                for index, request in enumerate(selected, 1):
                    if not cache_only and store.get(request['request_key']) is None and client is None:
                        client = OpenAIAnswerClient(settings)
                    context = by_key[(request['method'], request['question_id'])]
                    base = {k: request[k] for k in ('method', 'question_id', 'request_key', 'context_hash')}
                    try:
                        response, cached = answer_once(request, client, store, limiter, run / 'attempts.jsonl',
                                                       settings['max_attempts'], cache_only=cache_only)
                        result = base | {'status': 'complete', 'cache_hit': cached, 'response': response,
                                         'checks': inspect_answer(response['text'], context['pieces'],
                                                                  response['provider']['finish_reason'])}
                    except ProviderFailure as exc:
                        result = base | {'status': 'failed', 'error': exc.record}
                    answers.append(result)
                    # Export after every answer; resume uses the durable cache.
                    method = request['method']
                    atomic_jsonl(run / method / 'variant_001/answers.jsonl', [a for a in answers if a['method'] == method])
                    print(f'{index}/{len(selected)} {method} {request["question_id"]}: {result["status"]}', flush=True)
            summary = {'prepared': len(requests), 'selected': len(selected), 'answers': len(answers),
                       'successful': sum(a['status'] == 'complete' for a in answers),
                       'failed': sum(a['status'] == 'failed' for a in answers),
                       'cache_hits': sum(a.get('cache_hit', False) for a in answers),
                       'provider_usage': {key: sum(a['response']['provider']['usage'].get(key, 0) or 0
                                                 for a in answers if a['status'] == 'complete')
                                          for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')},
                       'live_requests_this_invocation': sum(a['status'] == 'complete' and not a['cache_hit'] for a in answers),
                       'dry_run': dry_run, 'account_tier': settings['account_tier'],
                       'billing_note': 'Paid OpenAI account; cost is billed per the provider\'s token-based pricing.'}
            write_json(run / 'usage_report.json', summary)
            for name, expected in corpus.snapshot['files'].items():
                require(file_hash(within(ROOT / config.corpus_dir, name)) == expected, 'Prepared input changed during generation')
            require(file_hash(source / 'stage_manifest.json') == effective['retrieval_manifest_sha256'], 'Retrieval manifest changed')
            for name, expected in source_manifest['artifact_sha256'].items():
                require(file_hash(within(source, name)) == expected, 'Retrieval input changed during generation')
            finish_run(run, manifest, error='Some generation requests failed; see saved answers and attempts' if summary['failed'] else None)
            print(json.dumps(summary | {'output': str(run)}), flush=True)
            require(not summary['failed'], 'Generation has failed requests; resume after resolving the recorded cause')
        except Exception as exc:
            if manifest['status'] == 'running':
                finish_run(run, manifest, error=f'Generation stopped: {type(exc).__name__}; inspect local artifacts')
            raise
        finally:
            if client is not None:
                client.close()
    return run
