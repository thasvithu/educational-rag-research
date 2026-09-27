"""Heading decisions and fallback summaries kept separate from indexed text."""

from collections import Counter
import json

from ...common.load_corpus import load_structure_metadata
from ...common.run_manifest import write_json
from .headings import resolve_headings


def prepare_structure(corpus, selected, corpus_dir, variant):
    resolved, summaries = {}, []
    with (variant / "heading_decisions.jsonl").open("x", encoding="utf-8") as stream:
        for document in corpus.documents:
            if document.document_id not in selected:
                continue
            layout = load_structure_metadata(corpus_dir, document)
            headings, decisions, summary = resolve_headings(document, layout)
            resolved[document.document_id] = headings
            summaries.append(summary)
            for decision in decisions:
                stream.write(json.dumps({"document_id": document.document_id, "source": document.source,
                                         **decision}, ensure_ascii=False) + "\n")
    write_json(variant / "heading_summary.json", summaries)
    return resolved


def write_structure_report(variant):
    headings = json.loads((variant / "heading_summary.json").read_text())
    counts, unit_kinds, fallback_chunks, rejected = Counter(), Counter(), Counter(), Counter()
    section_keys, split_events, missing_regions = set(), set(), set()
    for line in (variant / "chunks.jsonl").open(encoding="utf-8"):
        chunk = json.loads(line)
        meta = chunk["method_metadata"]
        counts["chunks"] += 1
        counts["chunks_without_heading"] += meta["missing_heading"]
        counts["chunks_below_64_content_tokens"] += meta["content_token_count"] < 64
        section_keys.add((chunk["document_id"], meta["section_start"], meta["section_end"]))
        if meta["missing_heading"]:
            missing_regions.add((chunk["document_id"], chunk["region_id"]))
        unit_kinds.update(meta["unit_kind_counts"])
        fallback_chunks.update(meta["fallback_types"])
        for event in meta["split_events"]:
            split_events.add((chunk["document_id"], event["char_start"], event["char_end"], event["kind"], event["fallback"]))
    for summary in headings:
        rejected.update(summary["rejection_reasons"])
    split_counts = Counter(kind + ":" + fallback for _, _, _, kind, fallback in split_events)
    report = {**dict(counts), "sections": len(section_keys), "regions_with_unheaded_text": len(missing_regions),
              "documents_without_accepted_headings": [s["source"] for s in headings if not s["accepted_headings"]],
              "heading_candidates": sum(s["candidate_count"] for s in headings),
              "accepted_headings": sum(s["accepted_headings"] for s in headings),
              "heading_rejection_reasons": dict(rejected), "unit_fragments_by_kind": dict(unit_kinds),
              "chunks_by_fallback": dict(fallback_chunks), "distinct_unit_splits_by_kind_and_fallback": dict(split_counts),
              "notes": ["Unit fragment counts may count one split unit in several chunks.",
                        "Distinct split counts deduplicate source-unit spans; one unit can use both sentence and token fallback.",
                        "Detected table bounding boxes have no reliable cleaned-text offsets; visible table lines guide row boundaries.",
                        "Heading levels and table rows remain heuristic, not certified document structure."]}
    write_json(variant / "structure_stats.json", report)
    lines = ["# Structure-aware heading and fallback report", "",
             f"Accepted headings: {report['accepted_headings']:,}; candidates: {report['heading_candidates']:,}.",
             f"Sections: {report['sections']:,}; chunks: {report['chunks']:,}.",
             f"Chunks without a heading: {report['chunks_without_heading']:,}.", "",
             "## Chunks requiring fallback", "", "| Fallback | Chunks |", "|---|---:|"]
    lines += [f"| {kind} | {count} |" for kind, count in sorted(fallback_chunks.items())]
    lines += ["", "## Heading rejections", "", "| Reason | Candidates |", "|---|---:|"]
    lines += [f"| {reason} | {count} |" for reason, count in sorted(rejected.items())]
    lines += ["", "## Limitations", ""] + ["- " + note for note in report["notes"]]
    (variant / "structure_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
