# Phase 3 — Fixed-size chunking

Implemented on 20 September 2026. Phase 1 and the revised Phase 2 development
prerequisites were satisfied. The larger reviewed benchmark remains due before
Phase 11; the 20 pilot drafts can support development through answer generation.

## What it does

The baseline divides each eligible source region into consecutive token windows
with **256 content tokens at most and zero overlap**. The final shorter chunk is
kept. The selected Jina tokenizer adds two special tokens during encoding, so a
full chunk has **258 encoder input tokens**. This fits the tested 2,048-token
encoder limit. No embedding inference or API call is needed at this stage.

This is a token baseline: boundaries may fall inside a sentence, a word, a list or
a table. Those cuts are part of the method we will evaluate. We do not move them
to paragraph boundaries, which would introduce a different segmentation rule.

## Where to read the code

| File | Responsibility |
|---|---|
| `common/langchain_loader.py` | Load validated prepared text as LangChain `Document` regions. |
| `common/tokenization.py` | Verify the pinned local Jina tokenizer files; count tokens and retain character offsets. |
| `chunking/fixed_size/chunker.py` | `FixedSizeSplitter`, a LangChain `TextSplitter` subclass. |
| `chunking/fixed_size/run.py` | Process regions, create shared chunk records and save reports/manifests. |
| `configs/fixed_size.json` | Versioned baseline settings. |
| `tests/test_fixed_size.py` | Synthetic boundary, Unicode, metadata and token-limit checks. |

### Why the splitter has a small custom implementation

We use LangChain's `split_text()`, `split_documents()` and `transform_documents()`
interfaces. The boundary logic retains Jina token offsets so that every chunk is
an **exact substring of the prepared source**. Token decoding can change case or
spacing; locating text with a first-match search can select the wrong occurrence.
The override also updates source offsets and page references for each chunk.

The rules are explicit:

1. Tokenize a complete eligible region without truncation or special tokens.
2. Propose a window of up to 256 tokens from that region's token stream.
3. Keep tokens sharing a source character together. Never cut inside that character.
4. Slice the original text, then count the slice again as an independent input.
   Cutting a WordPiece word can change its standalone tokenization. If the slice
   exceeds the cap, move the boundary back until it fits. Record these adjustments.
5. Continue from that boundary without overlap, retaining the tail.

Leading whitespace belongs to the first chunk, inter-token whitespace to the
chunk on its left, and trailing whitespace to the last chunk. This covers every
eligible character. An indivisible character group that cannot fit, or nonblank
input without encodable tokens, raises an explicit error instead of dropping text.
Short chunks can occur at region ends and after token-accounting adjustments.

Excluded and blank pages remain barriers. Normal nonblank page boundaries may be
crossed. The splitter carries source quality flags forward; it does not repair
extraction or infer missing image content.

## Run it

Use the existing environment and downloaded Phase 2 model bundle. The dependency
profile is `requirements-phase2.txt`, now including
`langchain-text-splitters==0.3.8`. For a fresh checkout, follow the bundle download
instructions in the [LangChain guide](langchain_guide.md). Chunking reads only its
tokenizer files, works on CPU and requires no GPU access or API key.

From the repository root, use a **new run ID** each time:

```bash
# Inspect one deterministically selected document per member first.
.venv/bin/python -m rag_pipeline chunk-fixed --run-id my_fixed_sample --sample

# Process all 60 documents.
.venv/bin/python -m rag_pipeline chunk-fixed --run-id my_fixed_full
```

The command also accepts `--config`, `--method-config` and `--model-config`.
To try another development size, copy the fixed-size configuration, change its
`chunk_size`, and supply the copy with `--method-config`. Do not tune on test results.
No completed output directory is overwritten; failed runs retain their manifests.

### A small LangChain example

```python
from rag_pipeline.common.langchain_loader import PreparedCorpusLoader
from rag_pipeline.common.tokenization import load_tokenizer
from rag_pipeline.chunking.fixed_size.chunker import FixedSizeSplitter

regions = PreparedCorpusLoader().load()
splitter = FixedSizeSplitter(load_tokenizer(), chunk_size=256)
chunks = splitter.split_documents(regions[:1])

print(chunks[0].page_content)
print(chunks[0].metadata["page_numbers"])
print(chunks[0].metadata["token_count"])  # Includes the two special tokens.
```

Use the CLI for research artifacts: it additionally creates validated shared
chunk records, deterministic IDs, coverage statistics and run manifests.

## Saved outputs

Under `data/outputs/rag/<run-id>/`:

```text
run_config.json
corpus_snapshot.json
environment.json
stage_manifest.json
fixed_size/variant_001/
    chunks.jsonl
    chunk_stats.json
    inspection_examples.json
    inspection_report.md
    stage_manifest.json
```

Use **`chunks.jsonl`** as the input to the later embedding stage. Each row has its
exact text, document/region identity, character span, physical pages, quality
flags, measured tokens and deterministic chunk ID. No vectors have been produced.
`method_metadata.content_token_count` excludes special tokens; `token_count`
includes them. `region_token_start/end` refer to the original region token stream,
not necessarily the independent chunk's retokenized count.

Run manifests include nested artifact hashes. The configuration hash covers the
tokenizer identity/files, token limit, boundary/whitespace policies and gap policy.
Chunk IDs exclude `run_id`. The ordered canonical-record hash likewise excludes
`run_id`, allowing reproducibility checks while retaining execution provenance.

## Verified results

| Check | Result |
|---|---|
| Three-member sample | 3 documents, 555 chunks |
| Full corpus | 60 documents, 169 eligible regions, **8,627 chunks** |
| Content tokens per chunk | 2–256; mean 253.18; median 256 |
| Maximum encoder input | 258 tokens, below the tested 2,048-token limit |
| Eligible character coverage | 11,424,026 characters; **100%** |
| Overlap / empty chunks / overflows | 0 / 0 / 0 |
| Cross-page chunks | 4,994 |
| Boundary backoff operations | 143 |
| First full-run processing time | Approximately 56 seconds on this laptop; not an embedding time |
| Tests | 51 pipeline + 17 cleaning tests passed |

The prepared corpus has 11,424,676 characters. The **650-character difference is
entirely whitespace outside eligible regions**, around the existing gap barriers.
No retained non-whitespace source content is lost.

Full and repeat outputs are in `phase3_fixed_full_20260920` and
`phase3_fixed_repeat_20260920`. An independent read-back audit checks every saved
span and token count, unique IDs, region coverage, unchanged prepared inputs,
artifact hashes and equality of both runs after excluding `run_id`.

- [Full-run statistics](../../data/outputs/rag/phase3_fixed_full_20260920/fixed_size/variant_001/chunk_stats.json)
- [Member/page inspection](../../data/outputs/rag/phase3_fixed_full_20260920/fixed_size/variant_001/inspection_report.md)
- [Boundary examples: definition, paragraph, table and tail](../../data/outputs/rag/phase3_verification_20260920/boundary_review.md)
- [Independent verification](../../data/outputs/rag/phase3_verification_20260920/verification.json)

These links refer to ignored local artifacts; teammates need a local copy or a
rerun. Repeat the independent audit with:

```bash
.venv/bin/python data/outputs/rag/phase3_verification_20260920/check_outputs.py
```

## What to review and what comes next

Read the boundary examples first. The LCD definition spans two chunks, and an
annual fee value is split between `500.` and `00`. These examples illustrate what
the baseline preserves and fragments; they are **not yet measured retrieval failures**.
The document tail remains intact in a shorter final chunk.

The results establish segmentation integrity, not semantic extraction accuracy
or answer quality. The 256-token setting is a starting baseline, not a proven
optimum. The next authorized phase can be **Phase 4 — sliding-window chunking**.
It will reuse these offset and token-accounting rules and add controlled overlap.
