"""Build inspectable LangChain messages without reading benchmark answers."""

import json
import re
from hashlib import sha256
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from transformers import PreTrainedTokenizerFast

from ..common.config import ROOT, within
from ..common.load_corpus import digest_json, file_hash, require


def load_settings(path):
    settings = json.loads(Path(path).read_text())
    require(settings['schema_version'] == 1, 'Unsupported generation configuration')
    require(settings['provider'] == 'groq' and settings['account_tier'] == 'free'
            and settings['paid_budget'] == 0, 'Only the declared zero-paid-budget Groq experiment is supported')
    require(settings['model_id'] == 'qwen/qwen3.8-27b' and settings['reasoning_effort'] == 'none',
            'A different model requires a new tokenizer/capability decision')
    for key in ('max_output_tokens', 'max_total_tokens', 'token_count_margin', 'timeout_seconds',
                'max_attempts', 'requests_per_minute', 'tokens_per_minute', 'requests_per_day', 'tokens_per_day'):
        require(type(settings[key]) is int and settings[key] > 0, f'Invalid {key}')
    require(0 < settings['temperature'] <= 2 and 0 < settings['top_p'] <= 1, 'Invalid decoding settings')
    require(type(settings['seed']) is int, 'Seed must be an integer')
    require(settings['max_total_tokens'] <= 131072, 'Model context limit exceeded')
    require(settings['requests_per_minute'] <= 30 and settings['tokens_per_minute'] <= 8000
            and settings['requests_per_day'] <= 1000 and settings['tokens_per_day'] <= 200000,
            'Configured limits exceed the verified free-plan limits')
    require(settings['pilot_question_ids'] and len(set(settings['pilot_question_ids'])) == len(settings['pilot_question_ids']),
            'Pilot IDs must be nonempty and unique')
    return settings


class GeneratorTokenizer:
    """Load only the pinned tokenizer and chat template, never model weights.

    Groq may serialize chat differently. The local count is an estimate with a
    safety margin; the provider's returned usage is recorded as authoritative.
    """

    def __init__(self, settings):
        identity = settings['tokenizer']
        folder = within(ROOT, identity['directory'])
        manifest = json.loads((folder / 'manifest.json').read_text())
        for key in ('repository', 'revision', 'files'):
            require(manifest[key] == identity[key], 'Tokenizer manifest differs from the pinned configuration')
        for name, expected in identity['files'].items():
            require(file_hash(within(folder, name)) == expected, 'Generator tokenizer checksum differs')
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(folder / 'tokenizer.json'),
                                               chat_template=(folder / 'chat_template.jinja').read_text())

    def count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def count_messages(self, messages):
        return len(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                      enable_thinking=False))


def prepare_request(context, settings, tokens, prompt, trial_id, implementation_hash):
    """Whitelist question/context fields. Gold annotations never enter messages."""
    question = context['question']
    require(sha256(question.encode()).hexdigest() == context['question_sha256'], 'Question hash differs')
    pieces = context['pieces']
    require([p['citation_id'] for p in pieces] == [f'S{i+1}' for i in range(len(pieces))], 'Invalid citation sequence')
    rendered = '\n\n'.join(p['citation_wrapper'] + p['text'] for p in pieces)
    require(rendered == context['rendered_context'], 'Context text differs from its saved pieces')
    # Format variables once. Braces or instructions in source text stay data.
    template = ChatPromptTemplate.from_messages([
        ('system', '{system_prompt}'),
        ('human', 'QUESTION\n{question}\n\nSOURCE EVIDENCE\n{evidence}\n\nAnswer the question using the evidence above.')])
    formatted = template.format_messages(system_prompt=prompt, question=question,
                                         evidence=rendered or '(No evidence was retrieved.)')
    messages = [{'role': 'system' if m.type == 'system' else 'user', 'content': m.content} for m in formatted]
    prompt_tokens = tokens.count_messages(messages)
    reservation = prompt_tokens + settings['max_output_tokens'] + settings['token_count_margin']
    require(reservation <= settings['max_total_tokens'], 'Full prompt exceeds the generator budget; no truncation performed')
    payload = {'model': settings['model_id'], 'messages': messages, 'temperature': settings['temperature'],
               'top_p': settings['top_p'], 'seed': settings['seed'], 'max_tokens': settings['max_output_tokens'],
               'reasoning_effort': settings['reasoning_effort'], 'n': 1, 'stream': False, 'stop': None}
    identity = {'payload': payload, 'trial_id': trial_id, 'prompt_version': settings['prompt_version'],
                'tokenizer': settings['tokenizer'], 'implementation_hash': implementation_hash,
                'provider_url': 'https://api.groq.com/openai/v1/chat/completions'}
    return {'schema_version': 1, 'question_id': context['question_id'], 'method': context['method'],
            'question_sha256': context['question_sha256'], 'corpus_hash': context['corpus_hash'],
            'context_hash': digest_json(context), 'request_key': digest_json(identity), 'identity': identity,
            'estimated_prompt_tokens': prompt_tokens, 'reserved_tokens': reservation,
            'evidence_generator_tokens': sum(tokens.count(p['text']) for p in pieces),
            'wrapper_generator_tokens': sum(tokens.count(p['citation_wrapper']) for p in pieces),
            'rendered_context_generator_tokens': tokens.count(rendered),
            'count_note': 'Separate piece counts are descriptive; full chat-template count governs the limit.'}


def inspect_answer(text, pieces, finish_reason):
    """Check citation syntax, not factual support; that needs later review."""
    known = {p['citation_id']: p for p in pieces}
    labels = sorted(set(re.findall(r'\[(S\d+)\]', text)))
    return {'citations': [{'label': label, 'known': label in known,
                           'source_span': {k: known[label][k] for k in
                                           ('document_id', 'source', 'page_numbers', 'char_start', 'char_end')}
                           if label in known else None} for label in labels],
            'unknown_citations': [label for label in labels if label not in known],
            'no_citations': not labels,
            'abstention_prefix': text.lstrip().lower().startswith('insufficient evidence:'),
            'output_limit_reached': finish_reason == 'length',
            'semantic_support_checked': False}
