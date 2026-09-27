"""Read-only answer/prompt/source audit with an optional cache replay comparison."""

import argparse
import json

from ..common.config import ROOT, load_config, validate_run_id, within
from ..common.load_corpus import digest_json, file_hash, load_corpus, require
from ..common.run_manifest import write_json
from ..embedding.inputs import verify_manifest
from ..retrieval.inputs import METHODS
from .answer import GeneratorTokenizer, inspect_answer
from .run import load_contexts, read_rows
from .state import AnswerStore


def audit(run_id, repeat_id=None):
    validate_run_id(run_id)
    run = ROOT / 'data/outputs/rag' / run_id
    report_path = run.with_name(run.name + '_audit.json')
    require(not report_path.exists(), 'Audit report already exists; preserve it')
    manifest = verify_manifest(run)
    effective = json.loads((run / 'run_config.json').read_text())
    require(digest_json(effective) == manifest['config_hash'], 'Generation configuration checksum differs')
    settings = effective['settings']
    corpus = load_corpus(ROOT / 'final_data', load_config(ROOT / 'rag_pipeline/config.json'))
    require(corpus.corpus_hash == effective['corpus_hash'], 'Current corpus differs')
    snapshot = json.loads((run / 'corpus_snapshot.json').read_text())
    require(corpus.snapshot['files'] == snapshot['files'], 'Prepared source hashes differ')
    contexts, source, _ = load_contexts(settings, corpus)
    require(file_hash(source / 'stage_manifest.json') == effective['retrieval_manifest_sha256'], 'Retrieval input differs')
    originals = {(c['method'], c['question_id']): c for c in contexts}
    prompt = within(ROOT, settings['prompt']).read_text()
    tokens = GeneratorTokenizer(settings)
    cache = AnswerStore(ROOT / 'data/outputs/rag/generation_answer_cache_v1/answers')
    repeat = None
    if repeat_id:
        validate_run_id(repeat_id)
        repeat = ROOT / 'data/outputs/rag' / repeat_id
        verify_manifest(repeat)
        require(json.loads((repeat / 'run_config.json').read_text()) == effective, 'Replay experiment settings differ')
    count, answers, deltas, flags = 0, 0, [], []
    observed_keys = {}
    for method in METHODS:
        variant = run / method / 'variant_001'
        requests = read_rows(variant / 'requests.jsonl')
        saved_contexts = read_rows(variant / 'generation_contexts.jsonl')
        require(len(requests) == len(saved_contexts), 'Missing context/request records')
        if repeat:
            for name in ('requests.jsonl', 'generation_contexts.jsonl'):
                require(file_hash(variant / name) == file_hash(repeat / method / 'variant_001' / name), 'Replay prompt/context changed')
        for request, context in zip(requests, saved_contexts):
            key = (method, request['question_id'])
            require(key not in observed_keys, 'Duplicate prepared request')
            observed_keys[key] = request['request_key']
            require(context == originals[key], 'Saved evidence changed from Phase 9')
            require(request['context_hash'] == digest_json(context), 'Evidence checksum differs')
            require(request['request_key'] == digest_json(request['identity']), 'Request checksum differs')
            identity = request['identity']
            require(identity['trial_id'] == effective['trial_id']
                    and identity['implementation_hash'] == effective['implementation_hash'], 'Trial/code identity differs')
            payload = identity['payload']
            expected_messages = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content':
                'QUESTION\n' + context['question'] + '\n\nSOURCE EVIDENCE\n' +
                (context['rendered_context'] or '(No evidence was retrieved.)') +
                '\n\nAnswer the question using the evidence above.'}]
            require(payload['messages'] == expected_messages, 'Prompt contains changed or extra material')
            require(payload == {'messages': expected_messages, 'model': settings['model_id'],
                                'temperature': settings['temperature'], 'top_p': settings['top_p'], 'seed': settings['seed'],
                                'max_tokens': settings['max_output_tokens'], 'reasoning_effort': settings['reasoning_effort'],
                                'n': 1, 'stream': False, 'stop': None}, 'Decoding controls differ')
            estimated = tokens.count_messages(payload['messages'])
            require(estimated == request['estimated_prompt_tokens'], 'Saved prompt count differs')
            require(estimated + settings['max_output_tokens'] + settings['token_count_margin'] == request['reserved_tokens']
                    <= settings['max_total_tokens'], 'Prompt budget differs')
            count += 1
        if effective['mode'] == 'prepare':
            continue
        results = read_rows(variant / 'answers.jsonl')
        require([a['question_id'] for a in results] == [r['question_id'] for r in requests
                if r['question_id'] in effective['question_ids']], 'Selected answers missing or reordered')
        replay = read_rows(repeat / method / 'variant_001/answers.jsonl') if repeat else None
        for i, result in enumerate(results):
            key = (method, result['question_id'])
            require(result['status'] == 'complete' and result['request_key'] == observed_keys[key], 'Failed/mismatched answer')
            require(result['context_hash'] == digest_json(originals[key]), 'Answer points to different evidence')
            response = result['response']
            require(cache.get(result['request_key']) == response, 'Saved answer differs from committed provider response')
            require(response['provider']['model'] == settings['model_id'], 'Provider returned a different model')
            checks = inspect_answer(response['text'], originals[key]['pieces'], response['provider']['finish_reason'])
            require(result['checks'] == checks, 'Citation diagnostics differ')
            request = next(r for r in requests if r['question_id'] == result['question_id'])
            actual = response['provider']['usage']['prompt_tokens']
            delta = actual - request['estimated_prompt_tokens']
            require(delta <= settings['token_count_margin'], 'Provider prompt count exceeds the reserved margin')
            require(actual + settings['max_output_tokens'] <= settings['max_total_tokens'], 'Actual context budget exceeded')
            deltas.append(delta)
            if checks['unknown_citations'] or checks['output_limit_reached']:
                flags.append({'method': method, 'question_id': result['question_id'], 'checks': checks})
            if replay:
                require(replay[i]['cache_hit'] and {k: v for k, v in replay[i].items() if k != 'cache_hit'} ==
                        {k: v for k, v in result.items() if k != 'cache_hit'}, 'Cache replay changed an answer')
            answers += 1
    require(set(observed_keys.values()) == set(effective['request_keys']), 'Prepared request inventory differs')
    report = {'passed': True, 'run_id': run_id, 'repeat_id': repeat_id, 'prepared_requests_checked': count,
              'answers_checked': answers, 'prepared_files_unchanged': len(snapshot['files']),
              'minimum_provider_minus_local_prompt_tokens': min(deltas) if deltas else None,
              'maximum_provider_minus_local_prompt_tokens': max(deltas) if deltas else None,
              'citation_or_length_flags': flags,
              'scope': 'Prompt/source/usage integrity and saved replay; not a factual accuracy score or fresh provider repeat.'}
    write_json(report_path, report)
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--repeat-id')
    args = parser.parse_args()
    audit(args.run_id, args.repeat_id)


if __name__ == '__main__':
    main()
