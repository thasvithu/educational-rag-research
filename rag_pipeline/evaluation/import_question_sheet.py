"""Import a team's CSV into a separate draft workspace, preserving every decision.

This does not approve annotations or assign development/test splits. Exact source
matches are evidence candidates: a teammate still checks whether they support
every required fact before they become gold evidence units.
"""

import argparse
from array import array
from collections import Counter
import csv
import json
from pathlib import Path
import re

from ..common.config import ROOT, load_config
from ..common.load_corpus import file_hash, load_corpus, require
from ..common.provenance import pages_for_span

REQUIRED = ('Question ID', 'Question', 'Answerability', 'Question category',
            'Reference answer', 'Exact PDF filename', 'Physical PDF page(s)',
            'Supporting passage(s)', 'Author', 'Review decision',
            'Notes / Reason if unanswerable')
CATEGORIES = {'definition': 'definition', 'eligibility': 'eligibility',
              'procedure': 'procedure', 'explanation': 'explanation',
              'exception': 'exception', 'table lookup': 'table_lookup',
              'multiple passages': 'multi_passage'}
PUNCTUATION = str.maketrans({'“': '"', '”': '"', '‘': "'", '’': "'"})


def normalize(text, punctuation=False):
    """Normalize only typography, with reversible indices into untouched text."""
    chars, starts, ends = [], array('I'), array('I')
    for i, char in enumerate(text):
        if char.isspace():
            if chars and chars[-1] == ' ':
                ends[-1] = i + 1
            elif chars:
                chars.append(' '); starts.append(i); ends.append(i + 1)
        else:
            chars.append(char.translate(PUNCTUATION) if punctuation else char)
            starts.append(i); ends.append(i + 1)
    if chars and chars[-1] == ' ':
        chars.pop(); starts.pop(); ends.pop()
    return ''.join(chars), starts, ends


def parse_pages(value):
    value = value.strip()
    if not re.fullmatch(r'\d+(?:\s*[-–]\s*\d+)?(?:\s*[,;]\s*\d+(?:\s*[-–]\s*\d+)?)*', value):
        return []
    pages = []
    for part in re.split(r'[,;]', value):
        nums = [int(n) for n in re.split(r'[-–]', part)]
        if nums[0] < 1 or nums[-1] < nums[0] or nums[-1] > 10000:
            return []
        pages.extend(range(nums[0], nums[-1] + 1))
    return sorted(set(pages))


def resolve_source(value, documents, aliases):
    # A pasted page note on the next line is metadata, not part of a filename.
    cleaned = '/'.join(p.strip() for p in value.strip().splitlines()[0].split('/')) if value.strip() else ''
    lookup = {d.source.casefold(): d for d in documents}
    key = cleaned.casefold()
    if key in lookup:
        return lookup[key], 'exact' if value == lookup[key].source else 'case_spaces_or_pasted_note'
    if key in aliases:
        target = aliases[key]
        require(target.casefold() in lookup, 'Unknown alias target: ' + target)
        return lookup[target.casefold()], 'explicit_filename_alias'
    # Same basename in exactly one source directory is an unambiguous path repair.
    hits = [d for d in documents if Path(d.source).name.casefold() == Path(cleaned).name.casefold()]
    if cleaned and len(hits) == 1:
        return hits[0], 'unique_basename'
    return None, 'unresolved'


class EvidenceMatcher:
    def __init__(self):
        self.cache = {}

    def find(self, doc, passage):
        for punctuation in (False, True):
            key = (doc.document_id, doc.clean_sha256, punctuation)
            if key not in self.cache:
                self.cache[key] = normalize(doc.text, punctuation)
            text, starts, ends = self.cache[key]
            needle = normalize(passage, punctuation)[0]
            if not needle:
                return []
            found, pos = [], 0
            while (pos := text.find(needle, pos)) >= 0:
                start, end = starts[pos], ends[pos + len(needle) - 1]
                pos += 1
                try:
                    pages = pages_for_span(doc, start, end)
                except ValueError:
                    continue  # Never bridge a missing/excluded page.
                found.append({'document_id': doc.document_id, 'source': doc.source,
                              'source_sha256': doc.document_id, 'clean_sha256': doc.clean_sha256,
                              'char_start': start, 'char_end': end, 'text': doc.text[start:end],
                              'page_numbers': [p.page_number for p in pages],
                              'match_rule': 'whitespace_and_quotes' if punctuation else 'whitespace'})
            if found:
                return found
        return []


def convert(row, number, corpus, aliases, matcher, reserved_documents):
    actions, issues = [], []
    doc, rule = resolve_source(row['Exact PDF filename'], corpus.documents, aliases)
    if doc and rule != 'exact':
        actions.append('Source resolved by ' + rule + ': ' + doc.source)
    if not doc:
        issues.append('source_or_scope_missing')
    answerability = {'answerable': 'answerable', 'unanswerable': 'unanswerable',
                     'ambiguous': 'ambiguous'}.get(row['Answerability'].strip().casefold())
    if answerability is None:
        issues.append('choose_unanswerable_or_ambiguous')
    category = CATEGORIES.get(row['Question category'].strip().casefold())
    if category is None:
        issues.append('choose_question_category')
    if not row['Reference answer'].strip():
        issues.append('reference_response_missing')
    if answerability != 'answerable' and not row['Notes / Reason if unanswerable'].strip():
        issues.append('nonanswerability_rationale_missing')

    supplied = row['Supporting passage(s)'].strip()
    passage, origin = supplied, 'supporting_passage'
    if answerability == 'answerable' and supplied.casefold() == 'same':
        passage, origin = row['Reference answer'].strip(), 'same_means_reference_answer'
        actions.append('Interpreted same as reference answer, as instructed by Vithusan.')
    elif answerability == 'answerable' and not supplied:
        passage, origin = row['Reference answer'].strip(), 'reference_answer_candidate_for_blank_evidence'
        actions.append('Searched the reference answer as a candidate for missing evidence.')
    if passage.casefold() in {'same', 'content', 'contents', 'n/a', '-'}:
        passage = ''
    # A short number/name alone is not safe to turn into a gold supporting passage.
    too_short = len(normalize(passage)[0]) < 20 or len(passage.split()) < 4
    matches = matcher.find(doc, passage) if doc and passage and not too_short else []
    supplied_pages = parse_pages(row['Physical PDF page(s)'])
    chosen = None
    if len(matches) == 1:
        chosen = matches[0]
    elif matches:
        on_cited_pages = [m for m in matches if set(m['page_numbers']).issubset(supplied_pages)]
        if len(on_cited_pages) == 1:
            chosen = on_cited_pages[0]
        else:
            issues.append('multiple_evidence_locations')
    if chosen and chosen['page_numbers'] != supplied_pages:
        actions.append('Physical pages recovered from exact evidence: ' + str(chosen['page_numbers']))
    if answerability == 'answerable' and chosen is None:
        issues.append('supporting_passage_too_short' if passage and too_short else 'evidence_needs_source_check')
    if not chosen and doc and (not supplied_pages or any(p > len(doc.pages) for p in supplied_pages)):
        issues.append('physical_pages_need_confirmation')

    # Keep these as review flags, never silently change a question or feed gold
    # source metadata to retrieval to compensate for missing question scope.
    if answerability == 'answerable' and not re.search(
            r'University|UCSC|UJP|Rajarata|Peradeniya|Moratuwa|Kelaniya|Vavuniya|Uva Wellassa|Eastern|Wayamba', row['Question'], re.I):
        issues.append('question_source_scope_needs_review')
    if re.search(r'\b(current|right now)\b', row['Question'], re.I):
        issues.append('question_time_scope_needs_review')

    return {'intake_schema_version': 1, 'record_type': 'benchmark_intake_draft',
            'question_id': row['Question ID'].strip(), 'sheet_row': number,
            'corpus_hash': corpus.corpus_hash, 'question': row['Question'].strip(),
            'reference_answer': row['Reference answer'].strip(), 'language': 'en',
            'answerability': answerability, 'category': category,
            'scope_source': doc.source if doc else None, 'author': row['Author'].strip(),
            'rationale': row['Notes / Reason if unanswerable'].strip(),
            'split': None, 'source_family': None,
            'development_only': bool(doc and doc.document_id in reserved_documents),
            'source_family_review_required': True,
            'review': {'status': 'pending', 'reviewer': None, 'reviewer_kind': None,
                       'pdf_verified': False, 'reviewed_content_hash': None},
            'evidence_origin': origin, 'evidence_alignment': 'aligned' if chosen else 'unresolved',
            'evidence_candidates': matches, 'selected_evidence_candidate': chosen,
            'evidence_units': [], 'rubric': [],
            'annotation_status': 'draft_requires_content_review',
            'automatic_actions': actions, 'unresolved': issues, 'original_row': row}


def run(input_path, decisions_path, output):
    input_path, decisions_path, output = input_path.resolve(), decisions_path.resolve(), output.resolve()
    require(output.is_relative_to((ROOT / 'data/benchmark/imports').resolve()), 'Output must be under data/benchmark/imports')
    require(not output.exists(), 'Use a new output directory; imports are never overwritten')
    decisions = json.loads(decisions_path.read_text())
    require(file_hash(input_path) == decisions['input_sha256'], 'CSV changed since exclusion decisions; inspect the revised input first')
    with input_path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames is not None and set(REQUIRED).issubset(reader.fieldnames), 'Missing CSV columns')
        headers, rows = reader.fieldnames, list(reader)
    require(all(None not in r and all(v is not None for v in r.values()) for r in rows), 'Malformed CSV row')
    ids = [r['Question ID'].strip() for r in rows]
    require(len(ids) == len(set(ids)) and all(ids), 'Duplicate or missing question IDs')
    require(set(decisions['exclude']).issubset(ids), 'An exclusion ID is absent from the input')
    for removed, kept in decisions.get('duplicate_of', {}).items():
        require(removed in decisions['exclude'] and kept not in decisions['exclude'], 'Invalid duplicate exclusion')
        require(kept in ids, 'Retained duplicate ID missing')
        lookup = {r['Question ID'].strip(): r for r in rows}
        require(normalize(lookup[removed]['Question'])[0].casefold() == normalize(lookup[kept]['Question'])[0].casefold(),
                'Duplicate wording differs; recheck exclusion')
    corpus = load_corpus(ROOT / 'final_data', load_config(ROOT / 'rag_pipeline/config.json'))
    benchmark_path = ROOT / 'data/benchmark/benchmark_manifest.json'
    manifest = json.loads(benchmark_path.read_text())
    reserved = {key for key, value in manifest['document_groups'].items()
                if value in manifest['reserved_development_groups']}
    matcher = EvidenceMatcher()
    retained = [(i, r) for i, r in enumerate(rows, 2) if r['Question ID'].strip() not in decisions['exclude']]
    drafts = [convert(r, i, corpus, decisions.get('source_aliases', {}), matcher, reserved) for i, r in retained]
    # Validate original source identities, not just their cleaned metadata.
    sources = {d['scope_source'] for d in drafts if d['scope_source']}
    source_hashes = {}
    for doc in corpus.documents:
        if doc.source in sources:
            source_hashes[doc.source] = file_hash(ROOT / 'data' / doc.source)
            require(source_hashes[doc.source] == doc.document_id, 'Original PDF hash differs: ' + doc.source)
    output.mkdir(parents=True, exist_ok=False)
    def write_csv(name, values, fields):
        with (output / name).open('x', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(values)
    write_csv('retained_questions.csv', [r for _, r in retained], headers)
    normalized_rows = []
    for d in drafts:
        row = dict(d['original_row'])
        if d['scope_source']:
            row['Exact PDF filename'] = d['scope_source']
        candidate = d['selected_evidence_candidate']
        if candidate:
            row['Physical PDF page(s)'] = ', '.join(map(str, candidate['page_numbers']))
            row['Supporting passage(s)'] = candidate['text']
        if d['category']:
            row['Question category'] = d['category']
        row['Review decision'] = 'Pending'
        row.update({'Reviewer': '', 'Evidence alignment': d['evidence_alignment'],
                    'Automatic actions': '\n'.join(d['automatic_actions']),
                    'Still needed': '; '.join(d['unresolved']),
                    'Import status': 'Draft; source match is not approval or answer correctness'})
        normalized_rows.append(row)
    write_csv('normalized_drafts.csv', normalized_rows,
              headers + ['Reviewer', 'Evidence alignment', 'Automatic actions', 'Still needed', 'Import status'])
    excluded = [r | {'Exclusion reason': decisions['exclude'][r['Question ID'].strip()]}
                for r in rows if r['Question ID'].strip() in decisions['exclude']]
    write_csv('excluded_questions.csv', excluded, headers + ['Exclusion reason'])
    with (output / 'draft_records.jsonl').open('x', encoding='utf-8') as f:
        for d in drafts:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    queue = [d['original_row'] | {'Sheet row': d['sheet_row'], 'Resolved source': d['scope_source'] or '',
             'Matched physical pages': str(d['selected_evidence_candidate']['page_numbers']) if d['selected_evidence_candidate'] else '',
             'Still needed': '; '.join(d['unresolved']), 'Automatic actions': '\n'.join(d['automatic_actions']),
             'Reviewer': ''} for d in drafts if d['unresolved']]
    write_csv('unresolved_questions.csv', queue, headers + ['Sheet row', 'Resolved source', 'Matched physical pages',
                                                         'Still needed', 'Automatic actions', 'Reviewer'])
    (output / 'decisions.json').write_text(json.dumps(decisions, indent=2) + '\n')
    summary = {'input': str(input_path.relative_to(ROOT)), 'input_sha256': file_hash(input_path),
               'input_records': len(rows), 'excluded': len(excluded), 'retained_drafts': len(drafts),
               'aligned_evidence': sum(d['evidence_alignment'] == 'aligned' for d in drafts),
               'aligned_answerable_evidence': sum(d['answerability'] == 'answerable' and d['evidence_alignment'] == 'aligned' for d in drafts),
               'unresolved_records': len(queue),
               'without_intake_flags': sum(not d['unresolved'] for d in drafts),
               'unresolved_counts': dict(Counter(x for d in drafts for x in d['unresolved'])),
               'matched_reference_fallbacks': sum(d['evidence_origin'] != 'supporting_passage' and d['evidence_alignment'] == 'aligned' for d in drafts),
               'confirmed_human_approvals': 0, 'assigned_test_questions': 0, 'freeze_ready': False,
               'corpus_hash': corpus.corpus_hash, 'original_pdf_sha256': source_hashes,
               'benchmark_manifest_sha256': file_hash(benchmark_path),
               'importer_sha256': file_hash(Path(__file__)),
               'artifact_sha256': {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()}}
    (output / 'import_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--decisions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.decisions, args.output), indent=2))


if __name__ == '__main__':
    main()
