"""Protect source offsets and honest review state during spreadsheet import."""

from dataclasses import replace
import unittest

from rag_pipeline.common.load_corpus import Corpus, digest_bytes
from rag_pipeline.common.provenance import Document, Page, eligible_regions
from rag_pipeline.evaluation.import_question_sheet import (
    REQUIRED, EvidenceMatcher, convert, normalize, parse_pages, resolve_source,
)


def document(text, pages=None):
    pages = pages or (Page(1, 'i', 0, len(text), True, (), (), False),)
    return Document('a' * 64, 'member/example.pdf', 'member', text,
                    digest_bytes(text.encode()), 'detail.json', 'b' * 64,
                    pages, eligible_regions(pages))


class QuestionSheetImportTests(unittest.TestCase):
    def test_whitespace_unicode_and_quotes_keep_exact_offsets(self):
        doc = document('  தமிழ்: “Credits”\n  require practical work.  ')
        match, = EvidenceMatcher().find(doc, 'தமிழ்: "Credits" require practical work.')
        self.assertEqual(match['text'], 'தமிழ்: “Credits”\n  require practical work.')
        self.assertEqual(match['text'], doc.text[match['char_start']:match['char_end']])
        self.assertEqual(match['match_rule'], 'whitespace_and_quotes')

    def test_repeated_passage_not_silently_first_matched(self):
        quote = 'A credit requires fifteen lecture hours.'
        text = quote + '\n\n' + quote
        pages = (Page(1, 'i', 0, len(quote), True, (), (), False),
                 Page(2, 'ii', len(quote)+2, len(text), True, (), (), False))
        doc = document(text, pages)
        row = self.row(quote)
        row['Physical PDF page(s)'] = '2'
        result = convert(row, 2, Corpus((doc,), 'c'*64, {}), {}, EvidenceMatcher(), set())
        self.assertEqual(len(result['evidence_candidates']), 2)
        self.assertEqual(result['selected_evidence_candidate']['page_numbers'], [2])
        row['Physical PDF page(s)'] = ''
        result = convert(row, 2, Corpus((doc,), 'c'*64, {}), {}, EvidenceMatcher(), set())
        self.assertIsNone(result['selected_evidence_candidate'])
        self.assertIn('multiple_evidence_locations', result['unresolved'])

    def test_gap_cannot_become_evidence(self):
        text = 'First source sentence.\n\n\n\nSecond source sentence.'
        split = text.index('\n')
        pages = (Page(1, '1', 0, split, True, (), (), False),
                 Page(2, '2', split+2, split+2, False, (), (), True, 'excluded'),
                 Page(3, '3', split+4, len(text), True, (), (), False))
        self.assertEqual(EvidenceMatcher().find(document(text, pages),
                         'First source sentence. Second source sentence.'), [])

    def row(self, quote):
        return dict.fromkeys(REQUIRED, '') | {
            'Question ID': 'q1', 'Question': 'According to the University 2024 handbook, what is a credit?',
            'Answerability': 'Answerable', 'Question category': 'Definition',
            'Reference answer': quote, 'Exact PDF filename': 'member/example.pdf',
            'Supporting passage(s)': 'same', 'Author': 'Writer', 'Review decision': 'Approved'}

    def test_same_resolves_but_approval_and_split_are_not_invented(self):
        quote = 'A credit requires fifteen lecture hours.'
        doc = document(quote)
        row = self.row(quote)
        result = convert(row, 2, Corpus((doc,), 'c'*64, {}), {}, EvidenceMatcher(), {doc.document_id})
        self.assertEqual(result['evidence_alignment'], 'aligned')
        self.assertEqual(result['review']['status'], 'pending')
        self.assertIsNone(result['review']['reviewer'])
        self.assertIsNone(result['split'])
        self.assertTrue(result['development_only'])
        self.assertEqual(result['evidence_units'], [])
        self.assertEqual(result['original_row']['Review decision'], 'Approved')

    def test_short_answer_and_unresolved_answerability_stay_pending(self):
        doc = document('The minimum attendance is 80%.')
        row = self.row('80%')
        result = convert(row, 2, Corpus((doc,), 'c'*64, {}), {}, EvidenceMatcher(), set())
        self.assertEqual(result['evidence_alignment'], 'unresolved')
        self.assertIn('supporting_passage_too_short', result['unresolved'])
        row['Answerability'] = 'Unanswerable or Ambiguous'
        result = convert(row, 2, Corpus((doc,), 'c'*64, {}), {}, EvidenceMatcher(), set())
        self.assertIsNone(result['answerability'])
        self.assertIn('nonanswerability_rationale_missing', result['unresolved'])

    def test_source_resolution_is_unique_and_pages_strict(self):
        doc = document('Text')
        self.assertEqual(resolve_source(' Member/ example.pdf ', (doc,), {})[0], doc)
        self.assertEqual(resolve_source('other/example.pdf', (doc,), {})[0], doc)
        other = replace(doc, source='another/example.pdf')
        self.assertIsNone(resolve_source('other/example.pdf', (doc, other), {})[0])
        self.assertIsNone(resolve_source('member/exampel.pdf', (doc,), {})[0])
        self.assertEqual(parse_pages('2–4, 7'), [2, 3, 4, 7])
        self.assertEqual(parse_pages('25 (Year 4)'), [])
        self.assertEqual(parse_pages('4-2'), [])
        self.assertEqual(parse_pages('X'), [])


if __name__ == '__main__':
    unittest.main()
