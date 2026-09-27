"""Aggregate boundary choices, resource measurements and source examples."""

from collections import Counter
import json

from ...common.run_manifest import write_json


def write_semantic_report(variant, splitter, sources):
    counts, units, decisions = Counter(), Counter(), Counter()
    examples = {}
    with (variant / "boundary_diagnostics.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            report = json.loads(line)
            counts["regions"] += 1
            counts["short_regions_without_scoring"] += report["threshold"] is None
            counts["units"] += len(report["units"])
            counts["pre_split_fragments"] += sum(u["forced_token_split"] for u in report["units"])
            counts["below_minimum_chunks"] += sum(c["below_minimum"] for c in report["chunks"])
            units.update(u["kind"] for u in report["units"])
            decisions.update(b["decision"] for b in report["boundaries"])
            for boundary in report["boundaries"]:
                distance = boundary["cosine_distance"]
                if distance is None:
                    continue
                label = ("selected_topic_change" if boundary["decision"] == "semantic" else
                         "retained_neighbors" if not boundary["selected"] and not boundary["candidate"] else None)
                if label is None:
                    continue
                value = distance if label == "selected_topic_change" else -distance
                if label not in examples or value > examples[label]["selection_value"]:
                    doc = sources[report["document_id"]]
                    base = report["region_char_start"]
                    left, right = [report["units"][boundary[k]] for k in ("left_unit", "right_unit")]
                    examples[label] = {"selection_value": value, "source": doc.source, "region_id": report["region_id"],
                                       **boundary, "document_char_offset": base + boundary["char_offset"],
                                       "left_text": doc.text[base+left["start"]:base+left["end"]],
                                       "right_text": doc.text[base+right["start"]:base+right["end"]]}
    summary = {**dict(counts), "unit_kinds": dict(units), "boundary_decisions": dict(decisions),
               "embedding": splitter.embeddings.statistics(), "cache_path": splitter.cache_path,
               "cache_only": splitter.cache_only,
               "notes": ["Minimum size applies to semantic cuts; hard-cap cuts and tails can be shorter.",
                         "Threshold ties stay together unless size forces a cut; processing follows source order.",
                         "Vectors are boundary-unit embeddings only; final chunk embeddings belong to Phase 7."]}
    write_json(variant / "semantic_stats.json", summary)
    write_json(variant / "semantic_examples.json", examples)
    lines = ["# Semantic boundary examples", "", "Selected deterministically by score. Similarity is not a correctness label.", ""]
    for label, example in examples.items():
        lines += [f"## {label}", "", f"Source: `{example['source']}`; document offset: {example['document_char_offset']}.",
                  f"Cosine distance: {example['cosine_distance']:.6f}; threshold: {example['threshold']:.6f}.",
                  f"Decision: `{example['decision']}`.", "", "### Left unit", "", "~~~~text", example["left_text"], "~~~~", "",
                  "### Right unit", "", "~~~~text", example["right_text"], "~~~~", ""]
    (variant / "semantic_report.md").write_text("\n".join(lines), encoding="utf-8")
