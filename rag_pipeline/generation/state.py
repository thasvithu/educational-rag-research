"""Durable answers and conservative free-plan scheduling shared across runs."""

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
import time

from ..common.load_corpus import digest_json, require
from ..common.run_manifest import write_json
from .client import ProviderFailure


@contextmanager
def locked_cache(directory):
    """One process owns the provider ledger, including across different runs."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.generation.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another generation process owns the shared cache') from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def append_record(path, record):
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


class AnswerStore:
    def __init__(self, folder):
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)

    def get(self, key):
        path = self.folder / f'{key}.json'
        if not path.exists():
            return None
        value = json.loads(path.read_text())
        require(value['request_key'] == key and digest_json(value['response']) == value['response_hash'],
                'Cached answer checksum differs')
        return value['response']

    def put(self, key, response):
        write_json(self.folder / f'{key}.json', {'request_key': key, 'response': response,
                                               'response_hash': digest_json(response)})


class FreePlanLimiter:
    """Persist pessimistic token reservations before each HTTP attempt.

    This ledger covers this pipeline only, not other use of the Groq account.
    All attempts count, including interrupted/failed attempts. Provider 429s
    remain authoritative. Daily limits stop the run instead of waiting a day.
    """
    def __init__(self, path, settings, clock=time.time, sleep=time.sleep):
        self.path, self.settings, self.clock, self.sleep = path, settings, clock, sleep
        self.rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def reserve(self, request):
        cost, s = request['reserved_tokens'], self.settings
        require(cost <= s['tokens_per_minute'], 'One request exceeds the configured free-plan minute budget')
        while True:
            now = self.clock()
            date = datetime.fromtimestamp(now, timezone.utc).date()
            today = [r for r in self.rows if datetime.fromtimestamp(r['time'], timezone.utc).date() == date]
            if len(today) >= s['requests_per_day'] or sum(r['tokens'] for r in today) + cost > s['tokens_per_day']:
                raise ProviderFailure('LocalDailyQuotaReached')
            recent = [r for r in self.rows if now - r['time'] < 61]
            if len(recent) < s['requests_per_minute'] and sum(r['tokens'] for r in recent) + cost <= s['tokens_per_minute']:
                row = {'time': now, 'request_key': request['request_key'], 'tokens': cost}
                append_record(self.path, row)
                self.rows.append(row)
                return
            delay = max(0.1, 61 - (now - min(r['time'] for r in recent)))
            print(f'Free-plan pacing: waiting {delay:.1f} seconds', flush=True)
            self.sleep(min(delay, 30))


def answer_once(request, client, store, limiter, attempts_path, max_attempts,
                *, cache_only=False, sleep=time.sleep):
    """Reuse committed answers; write failures and each retry as separate events."""
    cached = store.get(request['request_key'])
    if cached is not None:
        return cached, True
    if cache_only:
        raise ProviderFailure('CacheMiss')
    for attempt in range(1, max_attempts + 1):
        limiter.reserve(request)
        event = {'request_key': request['request_key'], 'question_id': request['question_id'],
                 'method': request['method'], 'attempt': attempt, 'time': time.time()}
        append_record(attempts_path, event | {'status': 'started'})
        try:
            response = client.invoke(request)
        except ProviderFailure as exc:
            append_record(attempts_path, event | {'status': 'failed', 'error': exc.record})
            if not exc.retryable or attempt == max_attempts:
                raise
            # Long provider waits remain visible; sleep in small intervals.
            delay = max(2 ** attempt, exc.record['retry_after_seconds'])
            if delay > 300:
                raise ProviderFailure('ProviderQuotaWaitExceedsFiveMinutes', exc.record['http_status'], delay) from None
            print(f'Retry after {delay:.1f} seconds ({exc.record["category"]})', flush=True)
            while delay > 0:
                step = min(30, delay)
                sleep(step)
                delay -= step
        else:
            store.put(request['request_key'], response)
            append_record(attempts_path, event | {'status': 'complete'})
            return response, False
