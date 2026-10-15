"""Read saved runs through their provenance links; never execute a pipeline stage."""

import json
from pathlib import Path

import numpy as np
import streamlit as st

from rag_pipeline.common.config import ROOT, load_config, within
from rag_pipeline.common.load_corpus import file_hash, load_corpus, require

OUTPUTS = ROOT / 'data/outputs/rag'
METHODS = ('fixed_size', 'sliding_window', 'structure_aware', 'semantic', 'late')
LABELS = dict(zip(METHODS, ('Fixed-size', 'Sliding-window', 'Structure-aware', 'Semantic', 'Late')))


def stamp(path):
    info = Path(path).stat()
    return (str(Path(path).resolve()), info.st_mtime_ns, info.st_size)


def artifact_path(relative):
    path = within(ROOT, str(relative))
    require(path.is_relative_to(OUTPUTS), 'Expected an artifact under data/outputs/rag')
    return path


@st.cache_data(show_spinner=False, max_entries=256)
def _verify(path_stamp, expected):
    require(file_hash(Path(path_stamp[0])) == expected, f'Artifact checksum differs: {path_stamp[0]}')


def checked(path):
    """Check a file against the nearest completed manifest that lists it."""
    path = Path(path).resolve()
    require(path.is_relative_to(OUTPUTS), 'Artifact escapes the output directory')
    for parent in path.parents:
        if parent == OUTPUTS:
            break
        manifest_path = parent / 'stage_manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            relative = path.relative_to(parent).as_posix()
            expected = manifest.get('artifact_sha256', {}).get(relative)
            if expected:
                require(manifest['status'] == 'complete', 'Selected artifact belongs to an incomplete stage')
                _verify(stamp(path), expected)
                return path
    raise ValueError(f'No completed manifest covers {path.name}')


@st.cache_data(show_spinner=False, max_entries=80)
def _json(path_stamp):
    return json.loads(Path(path_stamp[0]).read_text())


def read_json(path):
    return _json(stamp(checked(path)))


@st.cache_data(show_spinner=False, max_entries=40)
def _rows(path_stamp):
    with Path(path_stamp[0]).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rows(path, **filters):
    data = _rows(stamp(checked(path)))
    return [r for r in data if all(r.get(k) == v for k, v in filters.items())]


def catalog():
    """Discover finished runs; malformed/partial runs are listed separately."""
    retrieval, generation, unavailable = [], [], []
    for path in sorted(OUTPUTS.glob('*/stage_manifest.json')):
        try:
            manifest = json.loads(path.read_text())
            root = path.parent
            if manifest['status'] != 'complete':
                unavailable.append({'run': root.name, 'status': manifest['status']})
                continue
            if (root / 'queries.jsonl').exists():
                settings = read_json(root / 'run_config.json')['settings']
                if set(settings.get('inputs', {})) == set(METHODS):
                    retrieval.append(root)
            if (root / 'usage_report.json').exists():
                effective = read_json(root / 'run_config.json')
                if effective.get('mode') == 'answers':
                    generation.append(root)
        except (OSError, ValueError, KeyError) as exc:
            unavailable.append({'run': path.parent.name, 'status': 'unreadable: ' + str(exc)})
    return retrieval, generation, unavailable


@st.cache_resource(show_spinner='Checking the prepared corpus…', max_entries=2)
def _corpus(signature):
    config = load_config(ROOT / 'rag_pipeline/config.json')
    return load_corpus(within(ROOT, config.corpus_dir), config)


def corpus():
    folder = ROOT / 'final_data'
    signature = tuple(stamp(p) for p in sorted(folder.rglob('*')) if p.is_file())
    return _corpus(signature)


class Experiment:
    def __init__(self, retrieval, source_corpus):
        self.root = Path(retrieval)
        self.config = read_json(self.root / 'run_config.json')
        self.corpus = source_corpus
        require(self.config['corpus_hash'] == source_corpus.corpus_hash, 'Run and prepared corpus do not match')
        self.documents = {d.document_id: d for d in source_corpus.documents}
        self.variants = {}
        for method, relative in self.config['settings']['inputs'].items():
            variant = artifact_path(relative)
            manifest_path = variant / 'stage_manifest.json'
            expected = self.config['input_sha256'].get(manifest_path.relative_to(ROOT).as_posix())
            require(expected is not None, 'Missing input manifest fingerprint')
            _verify(stamp(manifest_path), expected)
            manifest = json.loads(manifest_path.read_text())
            require(manifest['status'] == 'complete' and manifest['method'] == method
                    and manifest['corpus_hash'] == source_corpus.corpus_hash, 'Incompatible vector input')
            self.variants[method] = variant

    def chunks(self, method, document_id=None):
        filters = {'document_id': document_id} if document_id else {}
        return rows(self.variants[method] / 'chunks.jsonl', **filters)

    def origin(self, method):
        """Find the exact chunking output that supplied the saved embedding run."""
        variant = self.variants[method]
        if method == 'late':
            return variant
        config = read_json(variant.parent.parent / 'run_config.json')
        expected = file_hash(checked(variant / 'chunks.jsonl'))
        candidates = [artifact_path(p) for p, sha in config['input_sha256'].items()
                      if p.endswith(f'/{method}/variant_001/chunks.jsonl') and sha == expected]
        require(len(candidates) == 1, 'Cannot identify the original chunking output unambiguously')
        checked(candidates[0])
        return candidates[0].parent

    def query_rows(self, method, name, question_id):
        return rows(self.root / method / 'variant_001' / name, question_id=question_id)

    def compatible_answers(self, roots):
        return [p for p in roots if read_json(p / 'run_config.json')['settings']['retrieval_run']
                == self.root.relative_to(ROOT).as_posix()]

    def answer_run(self, path):
        effective = read_json(path / 'run_config.json')
        require(effective['corpus_hash'] == self.corpus.corpus_hash
                and effective['retrieval_manifest_sha256'] == file_hash(self.root / 'stage_manifest.json'),
                'Answer run references a different retrieval snapshot')
        return effective


def intersects(start, end, row):
    return row['char_start'] < end and start < row['char_end']


def vector(variant, chunk):
    mapping = rows(variant / 'embedding_rows.jsonl', chunk_id=chunk['chunk_id'])
    require(len(mapping) == 1, 'Expected exactly one vector row for this chunk')
    record = mapping[0]
    require(all(record[k] == chunk[k] for k in ('document_id', 'char_start', 'char_end')), 'Vector/source mismatch')
    matrix = np.load(checked(variant / 'embeddings.npy'), mmap_mode='r', allow_pickle=False)
    return np.array(matrix[record['row_index']]), record


@st.cache_data(show_spinner=False, max_entries=24)
def render_pdf(path_stamp, page_number):
    import pymupdf
    with pymupdf.open(path_stamp[0]) as pdf:
        return pdf[page_number - 1].get_pixmap(matrix=pymupdf.Matrix(1.3, 1.3), alpha=False).tobytes('png')


def pdf_path(document):
    config = load_config(ROOT / 'rag_pipeline/config.json')
    path = within(within(ROOT, config.source_dir), document.source)
    # Source identity is the original PDF SHA-256, not just a filename.
    _verify(stamp(path), document.document_id)
    return path
