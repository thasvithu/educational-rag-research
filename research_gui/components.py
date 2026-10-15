"""Small reusable controls and exact-text highlighting."""

from html import escape
import json

import streamlit as st

from .artifacts import ROOT, intersects, pdf_path, render_pdf, stamp

COLORS = ('#dbeafe', '#ccfbf1', '#fef3c7', '#ede9fe', '#ffe4e6')


def highlight(text, start, end, spans):
    """Escape all source text before placing exact character intervals in HTML."""
    spans = [r for r in spans if intersects(start, end, r)]
    cuts = sorted({start, end} | {max(start, r['char_start']) for r in spans}
                  | {min(end, r['char_end']) for r in spans})
    parts = []
    for left, right in zip(cuts, cuts[1:]):
        active = [i for i, r in enumerate(spans) if r['char_start'] <= left and right <= r['char_end']]
        content = escape(text[left:right])
        if active:
            color = '#fdba74' if len(active) > 1 else COLORS[active[0] % len(COLORS)]
            title = escape('Shared by ' + str(len(active)) + ' chunks' if len(active) > 1
                           else f'Source characters {left}–{right}', quote=True)
            parts.append(f'<mark title="{title}" style="background:{color};color:#16243a;border-radius:3px">{content}</mark>')
        else:
            parts.append(content)
    return '<div class="source-text">' + ''.join(parts) + '</div>'


def explain(input_text, operation, output, implementation):
    with st.expander('How this stage works'):
        st.markdown(f'**Input:** {input_text}\n\n**Operation:** {operation}\n\n**Output:** {output}')
        st.caption('Implementation: ' + implementation)
        if st.checkbox('Show the Python implementation', key='show_implementation'):
            for name in implementation.split(' · '):
                path = ROOT / 'rag_pipeline' / name
                if path.is_file():
                    st.caption(name)
                    st.code(path.read_text(), language='python')


def raw(record, title='Advanced details'):
    with st.expander(title):
        st.json(record, expanded=False)


def jump(document_id, page, *, chunk_id=None, method=None):
    st.session_state.document_id = document_id
    st.session_state.physical_page = page
    if method:
        st.session_state.method = method
    st.session_state.focus_chunk = chunk_id
    st.session_state.screen = 'Chunks' if chunk_id else 'Documents'


def source_button(record, key, method=None):
    st.button('Open source' if method is None else 'Open chunk', key=key, on_click=jump,
              args=(record['document_id'], record['page_numbers'][0]),
              kwargs={'chunk_id': record.get('chunk_id') if method else None, 'method': method})


def page_picker(doc):
    value = st.session_state.get('physical_page', 1)
    if not 1 <= value <= len(doc.pages):
        st.session_state.physical_page = 1
    return doc.pages[int(st.number_input('Physical PDF page', min_value=1, max_value=len(doc.pages),
                                        step=1, key='physical_page')) - 1]


def pdf_preview(doc, page):
    try:
        path = pdf_path(doc)
        st.image(render_pdf(stamp(path), page.page_number), width='stretch')
    except (OSError, ValueError, RuntimeError) as exc:
        st.warning('PDF preview unavailable: ' + str(exc))


def chunk_picker(chunks, key='selected_chunk'):
    if not chunks:
        st.info('No eligible chunks at this source location.')
        return None
    ids = [r['chunk_id'] for r in chunks]
    focus = st.session_state.pop('focus_chunk', None)
    if focus in ids:
        st.session_state[key] = focus
    elif st.session_state.get(key) not in ids:
        st.session_state[key] = ids[0]
    lookup = {r['chunk_id']: r for r in chunks}
    cid = st.selectbox('Chunk', ids, key=key, format_func=lambda i:
                       f"Chunk {lookup[i]['chunk_index'] + 1} · pages {lookup[i]['page_numbers']} · {lookup[i]['token_count']} encoder tokens")
    return lookup[cid]


def chunk_card(chunk):
    cols = st.columns(3)
    cols[0].metric('Content tokens', chunk['method_metadata'].get('content_token_count', '—'))
    cols[1].metric('Encoder tokens', chunk['token_count'])
    cols[2].metric('Characters', chunk['char_end'] - chunk['char_start'])
    st.caption(f"{chunk['source']} · physical pages {chunk['page_numbers']} · source [{chunk['char_start']}, {chunk['char_end']})")
    st.code(chunk['text'], language=None, wrap_lines=True)
    if chunk['section_path']:
        st.write('Section:', ' → '.join(chunk['section_path']))
    raw(chunk)


def download_record(record, name, key):
    st.download_button('Download this record', json.dumps(record, ensure_ascii=False, indent=2),
                       file_name=name + '.json', mime='application/json', key=key)
