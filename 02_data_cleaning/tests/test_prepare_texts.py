"""Safety regressions for the final formatting pass."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_texts import prepare_page, page_number_candidates, remove_page_number, heading_spans


class PreparationTests(unittest.TestCase):
    def test_fallback_preserves_source_symbols_and_spacing(self):
        source = 'Biology*/Biology**  CS_101  2 2 3\nGPA ≥ 3.5; x²; தமிழ் සිංහල\n    x = 2 ** 3\n"""docstring"""'
        actual, stats = prepare_page('```text\n' + source + '\n```', 'poppler_layout')
        self.assertEqual(source, actual)
        self.assertEqual(stats['outer_fences_removed'], 1)

    def test_native_markdown_preserves_code_and_unpaired_stars(self):
        source = '# **Heading**\n**Bold** and _italic_\nBiology** / Biology*\n`x ** y`\n```python\nx = 2 ** 3\n"""docstring"""\n```'
        actual, _ = prepare_page(source, 'pymupdf4llm_markdown')
        self.assertEqual(actual, 'Heading\nBold and italic\nBiology** / Biology*\n`x ** y`\nx = 2 ** 3\n"""docstring"""')

    def test_table_values_are_not_deduplicated(self):
        actual, _ = prepare_page('|**Code**|Credits|\n|---|---|\n|CS_101<br>CS_102|2 2 3|', 'pymupdf4llm_markdown')
        self.assertEqual(actual, '|Code|Credits|\n|CS_101 / CS_102|2 2 3|')

    def test_emphasis_removal_keeps_word_number_boundaries(self):
        actual, _ = prepare_page('To complete**72** credits; programme**4 ** years', 'pymupdf4llm_markdown')
        self.assertEqual(actual, 'To complete 72 credits; programme 4 years')

    def test_footer_requires_repeated_offset_and_geometry(self):
        pages = [{'page_number': n, 'height': 1000, 'clean_text': f'Evidence\n{n-1}', 'source_lines': [
            {'text': str(n-1), 'bbox': [0, 940, 10, 950]},
            {'text': '3', 'bbox': [0, 600, 10, 610]},
        ]} for n in range(2, 6)]
        self.assertEqual(page_number_candidates(pages), {n: {str(n-1)} for n in range(2, 6)})
        self.assertEqual(page_number_candidates(pages[:3]), {})
        self.assertEqual(remove_page_number('Credits 3\n2 2 3\n1', {'1'})[0], 'Credits 3\n2 2 3')
        self.assertEqual(remove_page_number('1\nEvidence', {'1'})[0], '1\nEvidence')

    def test_heading_offsets_are_page_local_and_exact(self):
        original = '## **Entry requirements**\n\nGPA ≥ 3.5'
        text, _ = prepare_page(original, 'pymupdf4llm_markdown')
        page = {'clean_text': text, 'extraction_method': 'pymupdf4llm_markdown', 'heading_candidates': []}
        spans = heading_spans(page, original)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]['level'], 2)
        self.assertEqual(text[spans[0]['char_start']:spans[0]['char_end']], 'Entry requirements')

    def test_number_only_footer_has_no_retrieval_content(self):
        page = {'page_number': 103, 'height': 842, 'clean_text': '80',
                'source_lines': [{'text': '80 ', 'bbox': [292, 757, 306, 768]}]}
        self.assertEqual(page_number_candidates([page]), {103: {'80'}})
        self.assertEqual(remove_page_number('80\n80', {'80'})[0], '')
        page['source_lines'][0]['bbox'] = [292, 300, 306, 311]
        self.assertEqual(page_number_candidates([page]), {})


if __name__ == '__main__':
    unittest.main()
