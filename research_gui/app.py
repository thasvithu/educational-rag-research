"""Run with: .venv/bin/streamlit run research_gui/app.py"""

from pathlib import Path
import sys

# Streamlit executes this file as a script. Make repository imports explicit.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from research_gui.artifacts import Experiment, LABELS, METHODS, catalog, corpus, read_json, rows
from research_gui import document_views, question_views
from research_gui.components import raw


st.set_page_config(page_title='Educational RAG · Research Explorer', page_icon='🔎', layout='wide')
st.markdown('''<style>
  .stApp { background: #f7f9fc; }
  .block-container { max-width: 1500px; padding-top: 2rem; }
  h1, h2, h3 { color: #132d46; }
  [data-testid="stSidebar"] { background: #edf3f8; }
  .source-text { white-space: pre-wrap; overflow-wrap: anywhere; font-family: ui-monospace, monospace;
    font-size: .88rem; line-height: 1.7; background: #fff; color: #172b42;
    border: 1px solid #dbe4ee; border-radius: 10px; padding: 18px; max-height: 560px; overflow-y: auto; }
  .stage-strip { color: #246c73; font-size: .85rem; margin-bottom: 1.2rem; }
  [data-testid="stMetric"] { background: #fff; border: 1px solid #dbe4ee; padding: 12px; border-radius: 10px; }
  .stCode { border-radius: 8px; }
</style>''', unsafe_allow_html=True)


def overview(exp, generation_roots, unavailable):
    st.header('Your experiment, step by step')
    st.write('Follow a source document through five methods, or follow a question from retrieved evidence to its answer.')
    snapshot = exp.corpus.snapshot
    metrics = st.columns(4)
    for col, label, value in zip(metrics, ('Documents', 'Physical PDF pages', 'Methods', 'Saved questions'),
                                 (snapshot['documents'], snapshot['pages'], len(METHODS), len(rows(exp.root / 'queries.jsonl')))):
        col.metric(label, value)
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.subheader('Follow a document')
            st.write('PDF → prepared text → chunk boundaries → embeddings')
            st.button('Explore documents', on_click=lambda: st.session_state.update(screen='Documents'))
    with right:
        with st.container(border=True):
            st.subheader('Follow a question')
            st.write('Question → ranked chunks → packed evidence → answer')
            st.button('Explore retrieval', on_click=lambda: st.session_state.update(screen='Retrieval & context'))
    summary = []
    for method in METHODS:
        chunks = exp.chunks(method)
        summary.append({'method': LABELS[method], 'chunks / vectors': len(chunks),
                        'documents': len({c['document_id'] for c in chunks}),
                        'vector run': exp.variants[method].parent.parent.name})
    st.subheader('Five methods, one corpus')
    st.dataframe(summary, hide_index=True, width='stretch')
    st.bar_chart(summary, x='method', y='chunks / vectors', color='#268489')
    compatible = exp.compatible_answers(generation_roots)
    if compatible:
        usage = read_json(compatible[0] / 'usage_report.json')
        st.info(f"Generation: {usage['prepared']} prompts prepared · {usage['successful']} successful saved answers · "
                f"{usage['prepared']-usage['answers']} not generated in {compatible[0].name}.")
    else:
        st.info('No completed answer run is linked to this retrieval snapshot.')
    st.caption('Development outputs. Benchmark review and final evaluation remain pending; these counts do not rank quality.')
    with st.expander('A useful first demonstration'):
        st.markdown('1. Choose `aysha/UCSC_HB_2024.pdf` and physical page **71**.\n'
                    '2. Compare fixed-size and structure-aware boundaries.\n'
                    '3. Open **Retrieval & context**, question **pilot_001**.\n'
                    '4. Open **Answers & review** and inspect each method’s evidence.\n'
                    '5. Try **pilot_004** to inspect context-packing loss, then **pilot_019** for an abstention.')
    raw(exp.config['settings'], 'Recorded retrieval settings')
    raw({'corpus_hash': exp.corpus.corpus_hash, 'retrieval_run': exp.root.name,
         'excluded_pages': snapshot['excluded_pages'], 'blank_pages': snapshot['blank_included_pages'],
         'eligible_regions': snapshot['eligible_regions']}, 'Source identity and exclusions')
    if unavailable:
        with st.expander('Incomplete or unavailable runs'):
            st.dataframe(unavailable, hide_index=True, width='stretch')


def main():
    st.title('Educational RAG · Research Explorer')
    st.markdown('<div class="stage-strip">SOURCE → CHUNKS → VECTORS → RETRIEVAL → EVIDENCE → ANSWER</div>', unsafe_allow_html=True)
    retrieval_roots, generation_roots, unavailable = catalog()
    if not retrieval_roots:
        st.info('No completed retrieval runs found under data/outputs/rag. Restore the local research artifacts to explore them.')
        if unavailable:
            st.dataframe(unavailable, hide_index=True)
        return
    # Prefer original live runs over their cached repeats for the first visit.
    retrieval_roots.sort(key=lambda p: ('repeat' in p.name, p.name))
    generation_roots.sort(key=lambda p: ('repeat' in p.name, p.name))
    st.sidebar.title('Explore the research')
    selected = st.sidebar.selectbox('Retrieval snapshot', retrieval_roots, format_func=lambda p: p.name)
    source_corpus = corpus()
    exp = Experiment(selected, source_corpus)
    options = list(exp.documents)
    if st.session_state.get('document_id') not in options:
        st.session_state.document_id = options[0]
    document_id = st.sidebar.selectbox('Document', options, key='document_id',
                                       format_func=lambda i: exp.documents[i].source,
                                       help='Type a filename or member name to search.')
    method = st.sidebar.selectbox('Method', METHODS, format_func=LABELS.get, key='method')
    screen = st.sidebar.radio('Stage', ['Overview', 'Documents', 'Chunks', 'Compare methods', 'Embeddings',
                                        'Retrieval & context', 'Answers & review'], key='screen')
    st.sidebar.caption('Saved-run explorer · local files · no GPU or API calls')
    if st.sidebar.button('Refresh saved files'):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    doc = exp.documents[document_id]
    if screen == 'Overview':
        overview(exp, generation_roots, unavailable)
    elif screen == 'Answers & review':
        question_views.answers(exp, doc, method, generation_roots)
    else:
        views = {'Documents': document_views.documents, 'Chunks': document_views.chunks,
                 'Compare methods': document_views.comparison, 'Embeddings': document_views.embeddings,
                 'Retrieval & context': question_views.retrieval}
        views[screen](exp, doc, method)


try:
    main()
except (OSError, ValueError, KeyError, IndexError) as exc:
    st.error('The selected research output could not be displayed: ' + str(exc))
    st.info('Check the selected run and its source files. The explorer does not repair or overwrite experiment data.')
