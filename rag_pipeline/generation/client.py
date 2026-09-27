"""LangChain/Groq transport with safe response metadata and explicit retries."""

import os
import time

import httpx
from dotenv import dotenv_values
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langsmith import tracing_context

from ..common.config import ROOT
from ..common.load_corpus import require


class ProviderFailure(Exception):
    def __init__(self, category, status=None, retry_after=0, retryable=False):
        # Never copy an SDK exception string: it may contain sensitive values.
        super().__init__(category)
        self.record = {'category': category, 'http_status': status, 'retry_after_seconds': retry_after}
        self.retryable = retryable


class GroqAnswerClient:
    def __init__(self, settings, *, transport=None):
        key = os.environ.get('GROQ_API_KEY') or dotenv_values(ROOT / '.env').get('GROQ_API_KEY')
        require(bool(key), 'GROQ_API_KEY is missing')
        self.settings = settings
        self.observed = {}
        self.http = httpx.Client(event_hooks={'response': [self._capture]}, trust_env=False, transport=transport)
        self.model = ChatGroq(model=settings['model_id'], api_key=key, base_url='https://api.groq.com',
                              temperature=settings['temperature'], max_tokens=settings['max_output_tokens'],
                              timeout=settings['timeout_seconds'], max_retries=0, http_client=self.http,
                              cache=False)

    def _capture(self, response):
        response.read()
        headers = {k: v for k, v in response.headers.items()
                   if k.startswith('x-ratelimit-') or k in ('retry-after', 'x-request-id')}
        self.observed = {'http_status': response.status_code, 'headers': headers}
        if response.is_success:
            body = response.json()
            self.observed.update({k: body.get(k) for k in
                                  ('id', 'model', 'created', 'system_fingerprint', 'service_tier', 'usage')})
            self.observed['finish_reason'] = body['choices'][0].get('finish_reason')

    def invoke(self, request):
        self.observed = {}
        payload = request['identity']['payload']
        messages = [(SystemMessage if m['role'] == 'system' else HumanMessage)(content=m['content'])
                    for m in payload['messages']]
        start = time.monotonic()
        try:
            # The old SDK supports new API fields through extra_body. LangChain
            # still constructs and sends the messages. No external tracing.
            with tracing_context(enabled=False):
                answer = self.model.invoke(messages, seed=payload['seed'], top_p=payload['top_p'],
                                           extra_body={'reasoning_effort': payload['reasoning_effort']})
        except Exception as exc:
            status = getattr(exc, 'status_code', self.observed.get('http_status'))
            try:
                delay = max(0, float(self.observed.get('headers', {}).get('retry-after', 0)))
            except (ValueError, TypeError):
                delay = 0
            retryable = status == 429 or (isinstance(status, int) and status >= 500)
            retryable |= type(exc).__name__ in ('APIConnectionError', 'APITimeoutError')
            raise ProviderFailure(type(exc).__name__, status, delay, retryable) from None
        if self.observed.get('model') != payload['model']:
            raise ProviderFailure('UnexpectedResponseModel')
        if not isinstance(answer.content, str) or not answer.content.strip():
            raise ProviderFailure('EmptyAnswer')
        return {'text': answer.content, 'provider': self.observed, 'latency_seconds': time.monotonic() - start}

    def close(self):
        self.http.close()
