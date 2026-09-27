"""LangChain semantic splitter with inspectable decisions and exact source spans."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter
import numpy as np

from ...common.load_corpus import require
from .units import segment_units


def load_settings(path):
    settings = json.loads(Path(path).read_text())
    expected = {"schema_version": 1, "method": "semantic", "algorithm_version": "1", "overlap": 0,
                "threshold_policy": "linear_percentile_strictly_greater_source_order",
                "unit_policy": "sentences_list_items_table_rows_exact_offsets_v1",
                "size_policy": "minimum_for_semantic_cuts_maximum_always_tail_may_be_short",
                "short_region_policy": "fewer_than_three_units_pack_without_scoring"}
    require(set(settings) == set(expected) | {"chunk_size", "min_chunk_size", "percentile"}, "Unknown/missing semantic setting")
    require(all(type(settings[k]) is type(v) and settings[k] == v for k, v in expected.items()), "Unsupported semantic policy")
    require(type(settings["chunk_size"]) is int and settings["chunk_size"] > 0, "Invalid chunk size")
    require(type(settings["min_chunk_size"]) is int and 0 < settings["min_chunk_size"] <= settings["chunk_size"], "Invalid minimum size")
    require(type(settings["percentile"]) in (int, float) and 0 <= settings["percentile"] < 100, "Invalid percentile")
    return settings


class SemanticSplitter(TextSplitter):
    """Similarity selects boundaries; size checks always take priority.

    Minimum size is a target for semantic cuts, not permission to overflow. A
    size-forced chunk or region tail can be shorter and is explicitly labeled.
    """

    def __init__(self, tokenizer, embeddings, *, chunk_size=256, min_chunk_size=64, percentile=90):
        require(type(chunk_size) is int and 0 < chunk_size <= tokenizer.max_input_tokens - tokenizer.special_tokens, "Invalid chunk cap")
        require(type(min_chunk_size) is int and 0 < min_chunk_size <= chunk_size, "Invalid minimum size")
        require(type(percentile) in (int, float) and 0 <= percentile < 100, "Invalid percentile")
        super().__init__(chunk_size=chunk_size, chunk_overlap=0, strip_whitespace=False)
        self.tokens, self.embeddings = tokenizer, embeddings
        self.minimum, self.percentile = min_chunk_size, percentile
        self.last_diagnostics = []

    def analyze(self, text):
        units = segment_units(text, self.tokens, self._chunk_size)
        threshold = None
        distances = [None] * (len(units) - 1)
        entries = [{} for _ in units]
        if len(units) >= 3:
            vectors = np.asarray(self.embeddings.embed_documents([text[u.start:u.end] for u in units]), dtype=np.float64)
            require(vectors.ndim == 2 and vectors.shape[0] == len(units) and vectors.shape[1] > 0
                    and np.isfinite(vectors).all(), "Invalid boundary embedding matrix")
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            require(bool((norms > 0).all()), "Zero boundary vector")
            vectors = vectors / norms
            distances = (1 - np.clip(np.sum(vectors[:-1] * vectors[1:], axis=1), -1, 1)).tolist()
            threshold = float(np.percentile(distances, self.percentile, method="linear"))
            entries = getattr(self.embeddings, "last_entries", entries)
            require(len(entries) == len(units), "Boundary cache provenance row count differs")
        boundaries = []
        for i, distance in enumerate(distances):
            candidate = threshold is not None and distance > threshold
            boundaries.append({"left_unit": i, "right_unit": i+1, "char_offset": units[i].end,
                               "cosine_distance": distance, "threshold": threshold, "candidate": candidate,
                               "selected": False, "decision": "short_region_packing" if threshold is None else "below_or_equal_threshold"})

        chunks, first = [], 0
        for i, unit in enumerate(units):
            start, end = units[first].start, unit.end
            content_count = self.tokens.count(text[start:end], special_tokens=False)
            require(0 < content_count <= self._chunk_size, "Semantic packing exceeded cap")
            reason = None
            if i == len(units) - 1:
                reason = "region_end"
            else:
                next_text = text[start:units[i+1].end]
                overflow = (self.tokens.count(next_text, special_tokens=False) > self._chunk_size
                            or self.tokens.count(next_text) > self.tokens.max_input_tokens)
                if overflow:
                    reason = "max_tokens"
                elif boundaries[i]["candidate"] and content_count >= self.minimum:
                    reason = "semantic"
                elif boundaries[i]["candidate"]:
                    boundaries[i]["decision"] = "minimum_size_suppressed"
            if reason:
                chunks.append({"start": start, "end": end, "first_unit": first, "last_unit": i,
                               "end_reason": reason, "below_minimum": content_count < self.minimum})
                if i < len(boundaries):
                    boundaries[i].update(selected=True, decision=reason)
                first = i + 1
        diagnostics = {"threshold": threshold, "percentile": self.percentile,
                       "units": [asdict(unit) | {"content_token_count": self.tokens.count(text[unit.start:unit.end], special_tokens=False),
                                                   **entry} for unit, entry in zip(units, entries)],
                       "boundaries": boundaries, "chunks": chunks}
        return diagnostics

    def split_text(self, text):
        return [d.page_content for d in self.create_documents([text])]

    def create_documents(self, texts, metadatas=None):
        if metadatas is None:
            metadatas = [{} for _ in texts]
        require(len(texts) == len(metadatas), "Each text needs metadata")
        output, self.last_diagnostics = [], []
        for text, original in zip(texts, metadatas):
            base = original.get("char_start", 0)
            require(type(base) is int and base >= 0 and original.get("char_end", base+len(text)) == base+len(text), "Invalid region offsets")
            if not text.strip():
                continue
            report = self.analyze(text)
            report.update(document_id=original.get("document_id"), source=original.get("source"),
                          region_id=original.get("region_id"), region_char_start=base,
                          offset_convention="unit/boundary/chunk offsets are region-local; add region_char_start")
            self.last_diagnostics.append(report)
            for chunk in report["chunks"]:
                a, b = chunk["start"], chunk["end"]
                part = text[a:b]
                meta = deepcopy({k: v for k, v in original.items() if k != "pages"})
                units = report["units"][chunk["first_unit"]:chunk["last_unit"]+1]
                meta.update(char_start=base+a, char_end=base+b, start_index=a, region_char_start=base,
                            token_count=self.tokens.count(part), tokenizer_id=self.tokens.tokenizer_id,
                            method_metadata={"content_token_count": self.tokens.count(part, special_tokens=False),
                                             "special_token_count": self.tokens.special_tokens, "boundary_backoffs": 0,
                                             "unit_start": chunk["first_unit"], "unit_end": chunk["last_unit"]+1,
                                             "threshold": report["threshold"], "end_reason": chunk["end_reason"],
                                             "below_minimum_target": chunk["below_minimum"],
                                             "contains_pre_split_unit": any(u["forced_token_split"] for u in units)})
                require(meta["token_count"] <= self.tokens.max_input_tokens, "Encoder overflow")
                if "pages" in original:
                    meta["pages"] = deepcopy([p for p in original["pages"] if p["char_start"] < base+b and base+a < p["char_end"]])
                    meta["page_numbers"] = [p["page_number"] for p in meta["pages"]]
                    meta["quality_flags"] = sorted({f for p in meta["pages"] for f in p["quality_flags"]})
                output.append(Document(page_content=part, metadata=meta))
        return output
