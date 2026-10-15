"""Integrity, display and navigation checks for the saved-run explorer."""

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research_gui import artifacts
from research_gui.components import highlight
from research_gui.question_views import citation_labels


class ArtifactTests(unittest.TestCase):
    def test_highlights_escape_source_and_mark_overlap(self):
        text = '<script>x</script> café'
        html = highlight(text, 0, len(text), [{'char_start': 0, 'char_end': 10}, {'char_start': 5, 'char_end': 20}])
        self.assertNotIn('<script>', html)
        self.assertIn('&lt;scr', html)
        self.assertIn('Shared by 2 chunks', html)
        self.assertIn('café', highlight(text, 0, len(text), []))

    def test_comparison_uses_intersecting_source_spans(self):
        self.assertTrue(artifacts.intersects(10, 20, {'char_start': 5, 'char_end': 15}))
        self.assertFalse(artifacts.intersects(10, 20, {'char_start': 20, 'char_end': 25}))

    def test_grouped_and_individual_citations(self):
        self.assertEqual(citation_labels('A [S1, S2]. B [S2] and [S99].'), ['S1', 'S2', 'S99'])
        self.assertEqual(citation_labels('S1 without brackets is not a citation.'), [])

    def test_changed_artifact_and_incomplete_stage_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / 'run'
            run.mkdir()
            data = run / 'record.json'
            data.write_text('{"value":1}')
            manifest = {'status': 'complete', 'artifact_sha256': {'record.json': sha256(data.read_bytes()).hexdigest()}}
            (run / 'stage_manifest.json').write_text(json.dumps(manifest))
            with patch.object(artifacts, 'OUTPUTS', root):
                self.assertEqual(artifacts.read_json(data), {'value': 1})
                data.write_text('{"value":22}')
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    artifacts.read_json(data)
                manifest['status'] = 'failed'
                (run / 'stage_manifest.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, 'incomplete'):
                    artifacts.read_json(data)

    def test_paths_cannot_escape_output_root(self):
        with self.assertRaises(ValueError):
            artifacts.artifact_path('../../.env')
        with self.assertRaises(ValueError):
            artifacts.artifact_path('.env')


@unittest.skipUnless((artifacts.OUTPUTS / 'phase10_generation_pilot_20260927').exists(), 'Local research outputs required')
class RealAppTests(unittest.TestCase):
    def app(self):
        from streamlit.testing.v1 import AppTest
        return AppTest.from_file(str(artifacts.ROOT / 'research_gui/app.py'), default_timeout=120).run()

    def healthy(self, app):
        self.assertEqual([str(e.value) for e in app.exception], [])
        self.assertEqual([e.value for e in app.error], [])

    def test_all_screens_and_five_methods_without_provider_calls(self):
        with patch('httpx.Client.send', side_effect=AssertionError('No HTTP calls allowed in explorer')):
            app = self.app()
            self.healthy(app)
            self.assertIn(('Documents', '60'), [(m.label, m.value) for m in app.metric])
            for screen in ['Documents', 'Chunks', 'Compare methods', 'Embeddings', 'Retrieval & context', 'Answers & review']:
                app.sidebar.radio(key='screen').set_value(screen).run()
                self.healthy(app)
            for method in artifacts.METHODS:
                app.sidebar.radio(key='screen').set_value('Chunks')
                app.sidebar.selectbox(key='method').set_value(method).run()
                self.healthy(app)
            app.sidebar.radio(key='screen').set_value('Embeddings').run()
            self.healthy(app)
            self.assertIn(('Dimensions', '512'), [(m.label, m.value) for m in app.metric])

    def test_unexecuted_question_and_source_navigation(self):
        app = self.app()
        app.sidebar.radio(key='screen').set_value('Answers & review').run()
        app.selectbox(key='question_id').set_value('pilot_002').run()
        self.healthy(app)
        self.assertTrue(any('Not generated' in x.value for x in app.info))
        app.sidebar.radio(key='screen').set_value('Retrieval & context').run()
        app.selectbox(key='question_id').set_value('pilot_004').run()
        app.button(key='hit_1').click().run()
        self.healthy(app)
        self.assertEqual(app.sidebar.radio(key='screen').value, 'Chunks')
        app.sidebar.radio(key='screen').set_value('Retrieval & context').run()
        self.assertEqual(app.selectbox(key='question_id').value, 'pilot_004')
        self.healthy(app)

    def test_excluded_page_has_no_chunks_and_long_document_navigation(self):
        app = self.app()
        doc = next(d for d in artifacts.corpus().documents if any(not p.include_in_chunking for p in d.pages))
        page = next(p for p in doc.pages if not p.include_in_chunking)
        app.sidebar.selectbox(key='document_id').set_value(doc.document_id)
        app.sidebar.radio(key='screen').set_value('Documents').run()
        app.number_input(key='physical_page').set_value(page.page_number).run()
        self.healthy(app)
        self.assertTrue(any('barrier' in x.value for x in app.warning))
        app.sidebar.radio(key='screen').set_value('Compare methods').run()
        self.healthy(app)
        self.assertTrue(any('shared barrier' in x.value for x in app.info))


if __name__ == '__main__':
    unittest.main()
