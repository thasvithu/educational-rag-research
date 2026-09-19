from dataclasses import replace
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from rag_pipeline.cli import inspect

from rag_pipeline.common.chunk_schema import make_chunk
from rag_pipeline.common.config import Config, load_config, within
from rag_pipeline.common.load_corpus import digest_bytes, digest_json, load_corpus, load_structure_metadata
from rag_pipeline.common.provenance import pages_for_span, region_for_span
from rag_pipeline.common.run_manifest import create_run, finish_run


class FoundationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.corpus_dir = self.root / 'final_data'
        (self.corpus_dir / 'member').mkdir(parents=True)
        self.config = Config(1, 'final_data', 'data', 'data/outputs/rag', 1, 6,
                             {'member': 1}, 'excluded_and_blank_pages', 100, 42)
        self.doc_id = 'a' * 64
        # Duplicate text, Unicode, a blank page, and an explicit exclusion.
        pieces = ['Title தமிழ் x²\nrepeat', 'repeat\nNext', '', 'After blank', '', 'Final evidence']
        pages, offset = [], 0
        for n, text in enumerate(pieces, 1):
            headings = [{'text': 'Title', 'char_start': 0, 'char_end': 5,
                         'level': None, 'origin': 'pdf_font_candidate', 'inferred': True}] if n == 1 else []
            page = {'page_number': n, 'page_label': str(n), 'char_start': offset,
                    'char_end': offset+len(text), 'quality_flags': ['table_structure_review'] if n == 2 else [],
                    'include_in_chunking': n != 5, 'headings': headings, 'clean_text': text,
                    'raw_text': 'NEVER INDEX RAW', 'heading_candidates': [], 'detected_tables': []}
            if n == 5:
                page.update(exclusion_reason='corrupt_font_encoding', excluded_text='NEVER INDEX EXCLUDED')
            pages.append(page)
            offset += len(text)+2
        self.detail = {'document_id': self.doc_id, 'source': 'member/example.pdf', 'source_sha256': self.doc_id,
                       'cleaning_config': {}, 'pages': pages, 'pdf_toc': []}
        self.entry = {'document_id': self.doc_id, 'source': 'member/example.pdf', 'text': '\n\n'.join(pieces),
                      'metadata_path': 'member/example.json', 'pages': [self.light_page(p) for p in pages]}
        self.row = {'document_id': self.doc_id, 'source': self.entry['source'], 'source_sha256': self.doc_id,
                    'member': 'member', 'status': 'processed', 'include_in_default_corpus': True,
                    'text_path': 'member/example.txt', 'metadata_path': self.entry['metadata_path'],
                    'page_count': 6, 'clean_characters': len(self.entry['text']),
                    'clean_sha256': digest_bytes(self.entry['text'].encode())}
        self.manifest = {'schema_version': 1, 'config': {}, 'text_preparation': {'text_format': 'plain_text'},
                         'documents': [self.row], 'summary': {'pages_processed': 6,
                         'documents_in_corpus': 1, 'clean_characters': len(self.entry['text'])}}
        self.save()

    @staticmethod
    def light_page(page):
        return {k: v for k, v in page.items() if k not in {'clean_text', 'raw_text', 'excluded_text', 'heading_candidates', 'detected_tables'}}

    def save(self):
        for path, value in [('manifest.json', self.manifest), ('member/example.json', self.detail)]:
            (self.corpus_dir/path).write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
        (self.corpus_dir/'corpus.jsonl').write_text(json.dumps(self.entry, ensure_ascii=False)+'\n', encoding='utf-8')
        (self.corpus_dir/'member/example.txt').write_bytes(self.entry['text'].encode('utf-8'))

    def load(self):
        return load_corpus(self.corpus_dir, self.config)

    def test_exact_unicode_text_and_no_raw_exposure(self):
        doc = self.load().documents[0]
        self.assertEqual(doc.text, self.entry['text'])
        self.assertIn('தமிழ் x²', doc.text)
        self.assertNotIn('NEVER INDEX', doc.text)
        self.assertFalse(hasattr(doc, 'raw_text'))
        self.assertEqual([r.page_numbers for r in doc.regions], [(1, 2), (4,), (6,)])

    def test_cross_page_mapping_and_empty_page_exclusion(self):
        doc = self.load().documents[0]
        start, end = doc.pages[0].char_end-6, doc.pages[1].char_start+6
        self.assertEqual([p.page_number for p in pages_for_span(doc, start, end)], [1, 2])
        self.assertEqual([p.page_number for p in pages_for_span(doc, 0, doc.pages[0].char_end)], [1])
        for start, end in [(0, len(doc.text)), (doc.pages[1].char_start, doc.pages[3].char_end),
                           (doc.pages[3].char_start, doc.pages[5].char_end), (True, 5), (0, 0), (-1, 8)]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                region_for_span(doc, start, end)

    def test_duplicate_text_uses_supplied_offsets(self):
        corpus = self.load();doc = corpus.documents[0]
        start = doc.pages[1].char_start
        chunk = make_chunk(doc, start=start, end=start+6, method='fixed_size', corpus_hash=corpus.corpus_hash,
                           config_hash='b'*64, run_id='test_1', chunk_index=1)
        self.assertEqual(chunk.text, 'repeat')
        self.assertEqual(chunk.page_numbers, (2,))
        self.assertEqual(chunk.quality_flags, ('table_structure_review',))

    def test_chunk_ids_ignore_run_and_change_with_method_config_or_span(self):
        corpus = self.load();doc = corpus.documents[0]
        args = dict(start=0, end=5, method='fixed_size', corpus_hash=corpus.corpus_hash,
                    config_hash='b'*64, run_id='first', chunk_index=0)
        first = make_chunk(doc, **args)
        self.assertEqual(first.chunk_id, make_chunk(doc, **{**args, 'run_id': 'second'}).chunk_id)
        for change in [{'method': 'sliding_window'}, {'config_hash': 'c'*64}, {'end': 6}]:
            self.assertNotEqual(first.chunk_id, make_chunk(doc, **{**args, **change}).chunk_id)
        self.assertIsNone(first.token_count)
        for change in [{'token_count': 1}, {'token_count': True, 'tokenizer_id': 'test'}, {'method': 'unknown'},
                       {'run_id': '../bad'}, {'chunk_index': -1}, {'method_metadata': {'score': float('nan')}}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                make_chunk(doc, **{**args, **change})

    def test_heading_conversion_on_later_page(self):
        p = self.detail['pages'][1]
        p['headings'] = [{'text': 'Next', 'char_start': 7, 'char_end': 11, 'level': 2,
                          'origin': 'extractor_markdown', 'inferred': True}]
        self.entry['pages'][1] = self.light_page(p);self.save()
        doc = self.load().documents[0];heading = doc.pages[1].headings[0]
        self.assertEqual(heading.char_start, p['char_start']+7)
        self.assertEqual(doc.text[heading.char_start:heading.char_end], 'Next')

    def test_late_metadata_access_detects_changes_and_excludes_barriers(self):
        doc = self.load().documents[0]
        metadata = load_structure_metadata(self.corpus_dir, doc)
        self.assertEqual([p['page_number'] for p in metadata['pages']], [1, 2, 4, 6])
        self.assertNotIn('NEVER INDEX', json.dumps(metadata))
        self.detail['pdf_toc'] = [[1, 'changed', 1]];self.save()
        with self.assertRaises(ValueError):load_structure_metadata(self.corpus_dir, doc)

    def test_tampered_text_is_rejected(self):
        (self.corpus_dir/'member/example.txt').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'hash differs'):self.load()

    def test_excluded_text_in_jsonl_is_rejected(self):
        self.entry['text'] += 'NEVER INDEX EXCLUDED';self.save()
        with self.assertRaises(ValueError):self.load()

    def test_heading_offset_tampering_is_rejected(self):
        self.detail['pages'][0]['headings'][0]['char_end'] = 4
        self.entry['pages'][0] = self.light_page(self.detail['pages'][0]);self.save()
        with self.assertRaisesRegex(ValueError, 'Heading text'):self.load()

    def test_jsonl_page_offsets_must_equal_detail(self):
        self.entry['pages'][1]['char_start'] += 1;self.save()
        with self.assertRaisesRegex(ValueError, 'metadata differs'):self.load()

    def test_duplicate_sources_and_membership_fail_closed(self):
        path = self.corpus_dir/'corpus.jsonl';original = path.read_text()
        path.write_text(original+original)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):self.load()
        path.write_text('')
        with self.assertRaisesRegex(ValueError, 'membership'):self.load()

    def test_stable_snapshot_and_changed_metadata_changes_identity(self):
        a, b = self.load(), self.load()
        self.assertEqual(a.corpus_hash, b.corpus_hash)
        self.detail['pdf_toc'] = [[1, 'Title', 1]];self.save()
        self.assertNotEqual(a.corpus_hash, self.load().corpus_hash)

    def test_paths_cannot_escape_or_follow_outside_symlinks(self):
        for name in ['../escape', '/tmp/escape']:
            with self.assertRaises(ValueError):within(self.root, name)
        (self.root/'link').symlink_to('/tmp', target_is_directory=True)
        with self.assertRaises(ValueError):within(self.root, 'link/escape')
        self.row['metadata_path'] = '../escape.json';self.save()
        with self.assertRaises(ValueError):self.load()

    def test_configuration_rejects_unsafe_outputs_unknown_keys_and_bool_counts(self):
        for config in [replace(self.config, output_root='final_data'),
                       replace(self.config, expected_documents=True),
                       replace(self.config, gap_policy='ignore_exclusions')]:
            with self.assertRaises(ValueError):config.validate(self.root)
        path = self.root/'config.json'
        path.write_text(json.dumps({**self.config.to_dict(), 'unknown': 4}))
        with self.assertRaises(ValueError):load_config(path, self.root)

    def test_runs_require_ignore_rule_never_overwrite_and_record_failure(self):
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        with self.assertRaises(ValueError):create_run(self.root, self.config, 'inspection')
        (self.root/'.gitignore').write_text('/data/\n')
        run, manifest = create_run(self.root, self.config, 'inspection')
        (run/'example.txt').write_text('example')
        finish_run(run, manifest, error='synthetic failure')
        result = json.loads((run/'stage_manifest.json').read_text())
        self.assertEqual(result['status'], 'failed')
        self.assertIn('example.txt', result['artifact_sha256'])
        with self.assertRaises(FileExistsError):create_run(self.root, self.config, 'inspection')

    def test_failed_inspection_has_manifest_and_keeps_inputs(self):
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        (self.root/'.gitignore').write_text('/data/\n')
        path = self.root/'config.json';path.write_text(json.dumps(self.config.to_dict()))
        (self.corpus_dir/'member/example.txt').write_text('corrupted input')
        before = {p: p.read_bytes() for p in self.corpus_dir.rglob('*') if p.is_file()}
        with patch('rag_pipeline.cli.environment', return_value={}), self.assertRaises(ValueError):
            inspect(path, 'failed_inspection', self.root)
        manifest = json.loads((self.root/'data/outputs/rag/failed_inspection/stage_manifest.json').read_text())
        self.assertEqual(manifest['status'], 'failed')
        self.assertIn('Text hash differs', manifest['errors'][0])
        self.assertEqual(before, {p: p.read_bytes() for p in self.corpus_dir.rglob('*') if p.is_file()})

    def test_blank_page_cannot_be_reintroduced_as_evidence(self):
        corpus = self.load();doc = corpus.documents[0]
        self.assertTrue(doc.pages[2].is_barrier)
        with self.assertRaises(ValueError):
            make_chunk(doc, start=doc.pages[2].char_start, end=doc.pages[3].char_end,
                       method='fixed_size', corpus_hash=corpus.corpus_hash,
                       config_hash='b'*64, run_id='test', chunk_index=0)


if __name__ == '__main__':
    unittest.main()
