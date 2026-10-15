"""Document, boundary and vector views for teaching the five methods."""

from dataclasses import asdict

import numpy as np
import pandas as pd
import streamlit as st

from .artifacts import LABELS, METHODS, intersects, read_json, rows, vector
from .components import chunk_card, chunk_picker, explain, highlight, page_picker, pdf_preview, raw


DESCRIPTIONS = {
    'fixed_size': 'Take consecutive token windows. The next chunk starts where the previous one ends.',
    'sliding_window': 'Repeat part of the previous chunk so a boundary can retain nearby context. Orange marks shared text.',
    'structure_aware': 'Use heading hints and fitting paragraphs, lists and tables. Fall back when structure is missing or too large.',
    'semantic': 'Compare neighbouring text units. Keep recorded topic-change decisions separate from cuts forced by the token cap.',
    'late': 'Encode a longer passage first, then average the token representations belonging to this fixed-size target chunk.',
}


def documents(exp, doc, method):
    st.header('Documents')
    st.caption('Compare what the PDF shows with the exact text the pipeline received.')
    explain('Original PDF and prepared page records.', 'Load exact text and identify eligible regions.',
            'Shared text and source-page references for every method.', 'common/load_corpus.py · common/provenance.py')
    page = page_picker(doc)
    left, right = st.columns(2)
    with left:
        st.subheader(f'PDF · physical page {page.page_number}')
        pdf_preview(doc, page)
    with right:
        st.subheader('Prepared text')
        if page.is_barrier:
            st.warning('Chunking barrier: ' + ('excluded page' if not page.include_in_chunking else 'blank prepared page'))
        else:
            st.success('Eligible text · an ordinary page break does not end a region')
        st.code(doc.text[page.char_start:page.char_end] or '(No prepared text)', language=None, wrap_lines=True)
        st.caption(f'Source characters [{page.char_start}, {page.char_end})')
        st.write('Quality flags:', ', '.join(page.quality_flags) or 'None recorded')
        raw(asdict(page), 'Page metadata and heading candidates')
    with st.expander('Eligible document regions and page overview'):
        st.dataframe([asdict(r) for r in doc.regions], hide_index=True, width='stretch')
        st.dataframe([{'physical_page': p.page_number, 'characters': p.char_end-p.char_start,
                       'barrier': p.is_barrier, 'included': p.include_in_chunking} for p in doc.pages],
                     hide_index=True, width='stretch')


def method_details(exp, doc, method, chunk):
    origin = exp.origin(method)
    st.caption('Recorded boundary source: ' + origin.parent.parent.name)
    if method == 'structure_aware':
        decisions = rows(origin / 'heading_decisions.jsonl', document_id=doc.document_id)
        relevant = [r for r in decisions if r.get('page_number') in chunk['page_numbers']]
        st.subheader('Heading decisions on these pages')
        st.dataframe(relevant, hide_index=True, width='stretch')
        st.write('Recorded fallback and packing details')
        st.json(chunk['method_metadata'], expanded=False)
    elif method == 'semantic':
        diagnostics = rows(origin / 'boundary_diagnostics.jsonl', document_id=doc.document_id, region_id=chunk['region_id'])
        if diagnostics:
            d = diagnostics[0]
            # Stored diagnostics use region-local offsets; display document offsets.
            base = d['region_char_start']
            boundaries = [r | {'document_offset': base+r['char_offset']} for r in d['boundaries']
                          if chunk['char_start']-500 <= base+r['char_offset'] <= chunk['char_end']+500]
            st.subheader('Neighbouring boundary scores')
            if boundaries:
                frame = pd.DataFrame(boundaries)
                st.line_chart(frame.set_index('document_offset')[['cosine_distance', 'threshold']])
                st.dataframe(frame[['document_offset', 'cosine_distance', 'threshold', 'selected', 'decision']],
                             hide_index=True, width='stretch')
            st.caption('Larger cosine distance means less similarity. Offsets above are document-level characters.')
            st.write('This chunk ends because:', chunk['method_metadata'].get('end_reason'))
            with st.expander('Exact text units around this chunk'):
                for unit in d['units']:
                    a, b = base+unit['start'], base+unit['end']
                    if a < chunk['char_end'] and chunk['char_start'] < b:
                        st.caption(f"{unit['kind']} · [{a}, {b}) · forced token split: {unit['forced_token_split']}")
                        st.code(doc.text[a:b], language=None, wrap_lines=True)
    elif method == 'late':
        record = rows(origin / 'embedding_rows.jsonl', chunk_id=chunk['chunk_id'])[0]
        windows = rows(origin / 'encoder_windows.jsonl', document_id=doc.document_id)
        window = next(w for w in windows if w['window_id'] == record['encoder_window_id'])
        st.subheader('Text read before this chunk was pooled')
        st.caption(f"{window['input_token_count']} total tokens in this window. Colour marks the indexed target; surrounding text supplies context.")
        st.html(highlight(doc.text, window['char_start'], window['char_end'], [chunk]))
        comparison = rows(origin / 'early_late_comparison.jsonl', late_chunk_id=chunk['chunk_id'])
        if comparison:
            c = comparison[0]
            st.metric('Same-span early/late vector cosine', f"{c['cosine_similarity']:.4f}")
            st.caption('Representation similarity is not a retrieval-quality score.')
            raw(c, 'Context and matched baseline details')
        raw({k: v for k, v in window.items() if k != 'token_indices'}, 'Encoder window metadata')
    else:
        raw(chunk['method_metadata'], 'Token window and overlap details')


def chunks(exp, doc, method):
    st.header('Chunks · ' + LABELS[method])
    explain('An eligible source region.', DESCRIPTIONS[method], 'Exact source slices with token counts and pages.',
            f'chunking/{method}/chunker.py')
    page = page_picker(doc)
    all_chunks = exp.chunks(method, doc.document_id)
    focus = st.session_state.get('focus_chunk')
    page_only = st.checkbox('Show chunks touching this page', value=True)
    selected = [r for r in all_chunks if intersects(page.char_start, page.char_end, r)] if page_only else all_chunks
    if focus:
        selected = [r for r in all_chunks if r['chunk_id'] == focus] or selected
    st.caption(f'{len(all_chunks)} chunks in this document · {len(selected)} in the current selection')
    if page.char_end > page.char_start:
        st.html(highlight(doc.text, page.char_start, page.char_end, all_chunks))
        st.caption('Colours show chunk membership on this page. Orange means overlapping chunks.')
    chunk = chunk_picker(selected)
    if chunk:
        chunk_card(chunk)
        method_details(exp, doc, method, chunk)
        neighbours = [r for r in all_chunks if abs(r['chunk_index']-chunk['chunk_index']) == 1
                      and r['region_id'] == chunk['region_id']]
        with st.expander('Previous and next chunks in this region'):
            for neighbour in neighbours:
                st.caption(f"Chunk {neighbour['chunk_index']+1} · pages {neighbour['page_numbers']}")
                st.code(neighbour['text'], language=None, wrap_lines=True)


def comparison(exp, doc, method):
    st.header('Compare boundaries')
    st.caption('Both panels show the same source page. Chunk numbers are method-specific.')
    page = page_picker(doc)
    choices = st.columns(2)
    left = choices[0].selectbox('Left method', METHODS, format_func=LABELS.get, key='compare_left')
    right = choices[1].selectbox('Right method', METHODS, index=2, format_func=LABELS.get, key='compare_right')
    if page.is_barrier:
        st.info('This page is a shared barrier. No method should supply a chunk from it.')
    for col, name in zip(st.columns(2), (left, right)):
        with col:
            st.subheader(LABELS[name])
            st.caption(DESCRIPTIONS[name])
            touching = [r for r in exp.chunks(name, doc.document_id) if intersects(page.char_start, page.char_end, r)]
            st.html(highlight(doc.text, page.char_start, page.char_end, touching))
            st.dataframe([{'chunk': r['chunk_index']+1, 'start': r['char_start'], 'end': r['char_end'],
                           'content_tokens': r['method_metadata'].get('content_token_count'),
                           'section': ' → '.join(r['section_path'])} for r in touching], hide_index=True, width='stretch')
    st.subheader('Document-wide comparison')
    summary = []
    for name in METHODS:
        data = exp.chunks(name, doc.document_id)
        counts = [r['method_metadata'].get('content_token_count', r['token_count']-2) for r in data]
        summary.append({'method': LABELS[name], 'chunks': len(data), 'total content tokens': sum(counts),
                        'mean content tokens': round(float(np.mean(counts)), 1) if counts else 0})
    st.dataframe(summary, hide_index=True, width='stretch')
    st.caption('These are size and workload comparisons. They do not rank answer quality.')


def embeddings(exp, doc, method):
    st.header('Embeddings · ' + LABELS[method])
    explain('Chunk text, or a longer context window for late chunking.',
            'The shared Jina encoder represents text as 512 numbers for similarity search.',
            'A vector row linked to the original chunk.', 'embedding/encode_chunks.py · embedding/model_adapter.py')
    chunk = chunk_picker(exp.chunks(method, doc.document_id), key='vector_chunk')
    if chunk is None:
        return
    values, mapping = vector(exp.variants[method], chunk)
    cols = st.columns(3)
    cols[0].metric('Dimensions', len(values))
    cols[1].metric('Vector row', mapping['row_index'])
    cols[2].metric('Vector length (L2)', f'{np.linalg.norm(values):.6f}')
    st.line_chart(pd.DataFrame({'value': values}).rename_axis('dimension'))
    st.caption('Each point is one numeric dimension. Dimensions are not named facts, and this plot is not a quality score.')
    chunk_card(chunk)
    raw(mapping, 'Exact vector-to-source mapping')
    config = read_json(exp.variants[method].parent.parent / 'embedding_config.json')
    raw(config['identity']['embedding'], 'Recorded encoder settings')
    if method == 'late':
        method_details(exp, doc, method, chunk)
