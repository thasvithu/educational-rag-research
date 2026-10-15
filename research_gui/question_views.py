"""Follow a saved question through retrieval, context packing and generation."""

import json
import re

import streamlit as st

from .artifacts import LABELS, METHODS, ROOT, read_json, rows
from .components import download_record, explain, raw, source_button


def question_picker(exp):
    questions = rows(exp.root / 'queries.jsonl')
    lookup = {q['question_id']: q for q in questions}
    if st.session_state.get('question_id') not in lookup:
        st.session_state.question_id = questions[0]['question_id']
    qid = st.selectbox('Question', list(lookup), format_func=lambda q: q + ' · ' + lookup[q]['question'], key='question_id', persist_state='session')
    return qid


def context_panel(context, key):
    cols = st.columns(3)
    cols[0].metric('Evidence tokens (Jina)', context['evidence_content_tokens'])
    cols[1].metric('Evidence budget (Jina)', context['evidence_budget'])
    cols[2].metric('Source pieces', len(context['pieces']))
    for piece in context['pieces']:
        with st.expander(f"[{piece['citation_id']}] {piece['source']} · pages {piece['page_numbers']}"):
            st.code(piece['text'], language=None, wrap_lines=True)
            st.caption(f"Exact source [{piece['char_start']}, {piece['char_end']})")
            source_button(piece, f'{key}_{piece["citation_id"]}')
    with st.expander('Packing decisions: what was included or skipped'):
        st.dataframe([{'rank': d['rank'], 'status': d['status'],
                       'new tokens': d.get('new_content_tokens'), 'overlap characters removed': d.get('overlap_characters_removed')}
                      for d in context['decisions']], hide_index=True, width='stretch')
        raw(context['decisions'], 'Exact remaining source spans')
    st.caption('The evidence budget excludes citation wrappers. Full generator prompt counts use a separate tokenizer.')


def retrieval(exp, doc, method):
    st.header('Retrieval & context')
    explain('The same question vector for five full-corpus indexes.',
            'Rank matching chunks, then pack evidence within the shared budget.',
            'Raw ranked results and the exact passages supplied to the generator.',
            'retrieval/search.py · retrieval/assemble_context.py')
    qid = question_picker(exp)
    k = st.select_slider('Inspect top K', options=[1, 3, 5, 10], value=10)
    st.caption('K changes the displayed ranking only. Saved generation contexts were packed from the original top 10.')
    overview = []
    for name in METHODS:
        r = exp.query_rows(name, 'retrieval.jsonl', qid)[0]
        c = exp.query_rows(name, 'generation_contexts.jsonl', qid)[0]
        overview.append({'method': LABELS[name], 'first source': r['hits'][0]['source'],
                         'first score': r['hits'][0]['score'], 'context pieces': len(c['pieces']),
                         'evidence tokens': c['evidence_content_tokens']})
    st.dataframe(overview, hide_index=True, width='stretch')
    ranking = exp.query_rows(method, 'retrieval.jsonl', qid)[0]
    context = exp.query_rows(method, 'generation_contexts.jsonl', qid)[0]
    left, right = st.columns(2)
    with left:
        st.subheader(LABELS[method] + ' · ranked chunks')
        st.caption('Similarity scores are not correctness probabilities.')
        for hit in ranking['hits'][:k]:
            with st.expander(f"#{hit['rank']} · {hit['score']:.4f} · {hit['source']}", expanded=hit['rank'] == 1):
                st.caption(f"Physical pages {hit['page_numbers']}")
                st.code(hit['text'], language=None, wrap_lines=True)
                source_button(hit, f'hit_{hit["rank"]}', method=method)
    with right:
        st.subheader('Evidence selected for generation')
        context_panel(context, 'retrieval_context')
    download_record(ranking, f'{method}_{qid}_retrieval', 'download_retrieval')


def citation_labels(text):
    """Include grouped labels for inspection; do not claim semantic support."""
    return sorted({label for group in re.findall(r'\[([^\]]+)\]', text)
                   for label in re.findall(r'\bS\d+\b', group)})


def answers(exp, doc, method, generation_roots):
    st.header('Answers & review')
    explain('The question and saved context only.', 'Use the same grounded prompt and generator settings across methods.',
            'Answers with provider usage and links to their supplied evidence.', 'generation/answer.py · generation/client.py')
    qid = question_picker(exp)
    compatible = exp.compatible_answers(generation_roots)
    if not compatible:
        st.info('No completed answer run is linked to this retrieval run. Choose a different retrieval run in the sidebar.')
        return
    selected = st.selectbox('Saved answer run', compatible, format_func=lambda p: p.name)
    effective = exp.answer_run(selected)
    overview = []
    for name in METHODS:
        found = rows(selected / name / 'variant_001/answers.jsonl', question_id=qid)
        a = found[0] if found else None
        overview.append({'method': LABELS[name], 'status': a['status'] if a else 'Not generated',
                         'answer': a.get('response', {}).get('text', '') if a else '',
                         'tokens': a.get('response', {}).get('provider', {}).get('usage', {}).get('total_tokens') if a else None})
    st.dataframe(overview, hide_index=True, width='stretch')
    variant = selected / method / 'variant_001'
    request = rows(variant / 'requests.jsonl', question_id=qid)[0]
    context = rows(variant / 'generation_contexts.jsonl', question_id=qid)[0]
    found = rows(variant / 'answers.jsonl', question_id=qid)
    if found and found[0]['status'] == 'complete':
        a = found[0]
        st.subheader(LABELS[method] + ' · actual answer')
        st.code(a['response']['text'], language=None, wrap_lines=True)
        st.caption('Loaded from saved output. No model call is made when viewing this page.')
        usage = a['response']['provider']['usage']
        cols = st.columns(3)
        cols[0].metric('Prompt tokens (Qwen)', usage.get('prompt_tokens', '—'))
        cols[1].metric('Answer tokens (Qwen)', usage.get('completion_tokens', '—'))
        cols[2].metric('Recorded request latency', f"{a['response']['latency_seconds']:.2f} s")
        labels = citation_labels(a['response']['text'])
        pieces = {p['citation_id']: p for p in context['pieces']}
        st.write('Citations in this answer:', ', '.join(f'[{label}]' for label in labels) or 'None')
        for label in labels:
            if label in pieces:
                with st.expander(f'[{label}] · inspect cited evidence'):
                    piece = pieces[label]
                    st.caption(f"{piece['source']} · physical pages {piece['page_numbers']}")
                    st.code(piece['text'], language=None, wrap_lines=True)
                    source_button(piece, 'answer_citation_'+label)
            else:
                st.warning(f'[{label}] is not a supplied citation label.')
        raw(a, 'Saved response and original diagnostics')
        st.caption('The explorer recognises individual and grouped labels. A matching label does not prove a supported claim.')
    elif found:
        st.error('This saved request failed.')
        st.json(found[0])
    else:
        st.info('Not generated: this question has a prepared prompt but was outside the selected live sample.')
    with st.expander('Exact messages sent, or prepared for an unexecuted request'):
        for message in request['identity']['payload']['messages']:
            st.markdown('**'+message['role'].capitalize()+'**')
            st.code(message['content'], language=None, wrap_lines=True)
        raw(request['identity']['payload'] | {'messages': '(shown above)'}, 'Provider request settings')
    with st.expander('All evidence in this prompt'):
        context_panel(context, 'answer_context')
    with st.expander('Reference answer for human review · never part of the generator prompt'):
        # Reference annotations are deliberately read only in this review panel.
        benchmark = ROOT / 'data/benchmark/pilot.jsonl'
        if benchmark.exists():
            record = next((r for r in map(json.loads, benchmark.read_text().splitlines()) if r['question_id'] == qid), None)
            question = context['question']
            if record and record['question'] == question and record.get('corpus_hash') == exp.corpus.corpus_hash:
                st.warning('Current local annotation. Draft unless its recorded human review says otherwise; it may have changed since generation.')
                st.write(record['reference_answer'])
                raw(record, 'Reference evidence and review status')
            else:
                st.info('No matching current annotation is available.')
    with st.expander('Write a review note'):
        reviewer = st.text_input('Your name')
        label = st.selectbox('Observation', ['Needs review', 'Supported answer', 'Missing evidence', 'Wrong source or year',
                                           'Context packing loss', 'Unsupported answer claim', 'Citation problem', 'Extraction problem'], key='review_label_'+request['request_key'])
        note = st.text_area('Explain what the evidence shows', key='review_note_'+request['request_key'])
        review = {'status': 'draft_review_note', 'reviewer': reviewer, 'observation': label, 'note': note,
                  'method': method, 'question_id': qid, 'request_key': request['request_key'],
                  'context_hash': request['context_hash'], 'run': selected.name}
        st.download_button('Download review note', json.dumps(review, indent=2), file_name=f'{qid}_{method}_review.json',
                           mime='application/json', disabled=not reviewer.strip() or not note.strip())
        st.caption('Exported notes do not modify benchmark approvals or saved experiment outputs.')
