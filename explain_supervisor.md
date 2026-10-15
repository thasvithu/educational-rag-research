# Explaining Our Research — Full Walkthrough for the Supervisor

**Project:** Impact of Document Chunking Strategies on Retrieval-Augmented Generation (RAG) and Failure Modes in Educational Systems
**Team:** Vithusan, Aysha, Haleema — Group 04, University of Vavuniya
**Document purpose:** A complete, chronological account of everything done in this research, from the very first setup step to the current final results, so nothing has to be reconstructed from memory in a supervisor meeting. Every number quoted below comes directly from saved run artifacts or project documentation — nothing here is a rough estimate unless explicitly labeled as one.

---

## 1. The research question

We are comparing **five different ways of splitting (chunking) documents** before feeding them into a Retrieval-Augmented Generation (RAG) system, to see which chunking strategy produces the best retrieval and the most trustworthy answers on real Sri Lankan university handbook/course material. The five methods are:

1. **Fixed-size chunking** — cut every document into equal token-sized blocks (the baseline).
2. **Sliding-window chunking** — like fixed-size, but overlapping windows so no sentence gets cut off without some repeated context.
3. **Structure-aware chunking** — follow the document's own headings/paragraphs/lists/tables instead of blind token counts.
4. **Semantic chunking** — cut where the *meaning* of the text changes (measured by embedding similarity), not at a fixed size.
5. **Late chunking** — embed a *larger* surrounding context first, then pool down to the small chunk, so each chunk's vector "knows about" its neighbors.

We asked four concrete questions (stated in the project plan from day one):
- Which method retrieves the right evidence most reliably?
- Which method produces the most correct, grounded (faithful) answers?
- When a method fails, why did it fail — bad retrieval, bad chunk boundaries, bad context packing, or a generation mistake?
- When is each method actually worth using, given its cost?

**We deliberately did not go in trying to prove one method is "the winner."** The plan explicitly states: *"We are testing alternatives, not implementing a predetermined winner. A result showing small or uncertain differences is still a valid research outcome."* This matters because our final result (Section 14) does show a real, statistically meaningful difference, and it's worth being able to say we weren't fishing for it.

---

## 2. The corpus (the raw material)

Before any chunking could happen, we needed a clean, verified source corpus:

- **60 PDF documents** total, collected by the three team members: Aysha 12, Haleema 24, Vithusan 24.
- **6,215 physical PDF pages** across those documents.
- **44 pages explicitly excluded** — a 2018–2022 UOC management/finance handbook had corrupt font encoding on these pages; original extraction was kept in detailed JSON for reference, but they're barriers for chunking.
- **172 other pages** that are included but legitimately blank/whitespace-only.
- **169 continuous "eligible regions"** of usable text after removing the above exclusions — this is the actual unit every chunking method operates on.
- **11,424,676 prepared text characters** total (11,424,026 of which fall inside eligible regions).

Every single later phase re-verifies these same corpus facts via file hashes, so if anyone ever asks "are you sure all five methods saw the exact same source text?" — yes, and it's checked automatically on every run, not just assumed once.

---

## 3. Phase 1 — Foundation (19 September 2026)

Before writing any chunking code, we built the shared plumbing every later phase depends on:

- A strict corpus loader that checks text consistency, page provenance, and metadata.
- A chunk-record schema with deterministic IDs (so re-running the same input always produces the same IDs).
- A "run manifest" system: every experiment run gets its own folder, records exactly which code/config/inputs it used (down to file hashes), and marks itself `complete` or `failed` — this is what lets us say with certainty "nothing was silently re-run with different settings halfway through."

**Checks passed:** 17 foundation tests + 17 existing data-cleaning tests. 187 file hashes (original PDFs + prepared files) confirmed unchanged. A separate isolated environment with only 4 pinned dependencies reproduced the identical corpus identity.

**No chunking, embedding, retrieval, or generation work happened yet** — this phase was explicitly "shared foundation only."

---

## 4. Phase 2 — Protocol, embedding choice, and a 20-question pilot (20 September 2026)

This is where the big upfront decisions got made and recorded, so later phases wouldn't have to re-litigate them:

- **Embedding model chosen:** `jinaai/jina-embeddings-v2-small-en`, pinned to an exact model revision (`44e7d1d6...`) and exact custom-code revision (`f3ec4cf7...`), 512-dimensional vectors. This model is used **identically across all five chunking methods** — it is the one thing we deliberately held constant, so any performance difference we measure is attributable to chunking, not to different embeddings.
- **Pooling decision:** we use *content-token-only mean pooling* (excluding padding/special tokens) rather than the model's stock pooling recipe — a documented, deliberate adaptation applied the same way everywhere.
- **Hardware decision:** Vithusan's own machine — 16 logical CPUs, ~14.93 GiB RAM, NVIDIA RTX 3050 Laptop GPU (4,096 MiB VRAM). No cloud GPU assumed.
- **Generator chosen (at the time):** GroqCloud's free tier — **this later changed**, see Section 13.
- A `PreparedCorpusLoader` and `JinaEmbeddings` wrapper were built as proper LangChain interfaces (`BaseLoader` / `Embeddings`), so every later phase reuses the same, tested loading/embedding code rather than five different ad hoc implementations.
- **First 20-question pilot** drafted: 18 answerable, 1 unanswerable, 1 ambiguous, drawn from 7 source PDFs with 13 pages visually checked. This was explicitly a *development pilot*, never meant to be a final result.

**Checks passed:** 56 tests (39 new + 17 existing). CUDA probe on the RTX 3050 passed at 2,048 tokens (~667 MiB peak GPU memory). All 122 prepared input files from Phase 1 remained unchanged.

**Human review at this point: zero.** This was flagged honestly in the handover doc.

---

## 5. Phases 3–6 — Building the four non-contextual chunking methods

Each of these phases built one chunking method, with its own test suite, and each one re-verified that the other methods' outputs and the underlying corpus hadn't changed.

### Phase 3 — Fixed-size (the baseline), 20 September 2026
- Rule: 256 content tokens maximum per chunk, **no overlap**, never cut inside a word (uses "boundary backoff" — moves the cut point back until it lands on a clean token boundary; this happened 143 times across the whole corpus).
- **Result: 8,627 chunks** from 60 documents / 169 regions. Mean 253.18 content tokens/chunk. 100% of eligible text covered. Zero empty chunks, zero overflow chunks.
- Checks: 51 pipeline + 17 cleaning tests passed.

### Phase 4 — Sliding window, 20 September 2026
- Rule: same 256-token cap, but **64 tokens of deliberate overlap** between consecutive windows (nominal stride 192).
- **Result: 11,419 chunks** (2,792 more than fixed-size) — about 33% more total content tokens to embed (2,904,243 vs. fixed-size's 2,184,219), because the overlapping text gets embedded twice.
- 11,250 adjacent chunk pairs confirmed to share *exactly* 64 tokens, no more, no less.
- Checks: 62 pipeline + 17 cleaning = 79 tests passed. Setting overlap to zero was confirmed to reproduce the Phase 3 baseline exactly (a built-in correctness check).

### Phase 5 — Structure-aware, 21 September 2026
- Rule: follow the document's own headings, paragraphs, lists, and tables, instead of blind token counting. Headings are accepted from three sources (a hint already in the prepared text, large/bold font detection, or the PDF's own outline/bookmark structure) with strict rules to reject false positives (bullet points, table-of-contents lines, etc.).
- **Result: 13,377 chunks** — noticeably more chunks than fixed-size, but *smaller* on average (mean 163.28 content tokens vs. 253.18), because real document sections are often shorter than 256 tokens.
- 8,448 headings accepted out of 89,071 candidate hints; 6,209 distinct sections found. When a unit was too big to fit (e.g. a huge table), it fell back to splitting by table row, then by sentence, then by raw token window, in that order — this fallback chain fired for 1,372 / 940 / 261 chunks respectively.
- **One real bug found and fixed here:** an early run hit a case where a unit contained only decorative glyphs the tokenizer couldn't encode at all, causing a failure. Fixed by attaching such glyphs to a neighboring unit instead of deleting them; the earlier failed run was kept on disk for diagnosis rather than silently discarded.
- Checks: 78 pipeline + 17 cleaning tests passed.

### Phase 6 — Semantic chunking, 21 September 2026
- Rule: embed each sentence/list-item/table-row, measure the cosine *distance* between consecutive units, and cut where that distance exceeds a threshold (the region's own 90th percentile distance, with a 64-token minimum chunk size and the same 256-token hard cap as everyone else).
- **Result: 11,927 chunks.** Of the chunk endings, 5,337 were genuine semantic cuts, 6,421 were "next unit would overflow the cap" cuts, and 169 were simply the end of a document region.
- A real example from the data: the model cut between a scholarship-fees sentence and a "network services" sentence at cosine distance 0.310, just above the computed threshold of 0.297 — while two related Network Operating Centre sentences stayed together at a much smaller distance of 0.130. This is a genuine, sensible semantic boundary, not a coincidence.
- Full run took ~642 seconds (10.7 minutes) the first time; a **cache-only repeat run** (reusing previously computed embeddings) took only ~113 seconds with zero new model inference and produced byte-identical output — this is one of our standard "did we actually get the same thing twice" checks, and it passed.
- Checks: 89 pipeline + 17 cleaning = 106 tests passed. An independent audit recomputed all 103,399 similarity scores from the saved vectors and matched the saved chunk boundaries exactly.

---

## 6. Phase 7 — Shared embeddings for the four non-contextual methods (21 September 2026)

Once all four chunk sets existed (fixed, sliding, structure, semantic), each chunk needed to be turned into a 512-dimensional vector using the *same* Jina model, so retrieval could compare across methods fairly.

- **Result: 45,350 embedding vectors total** across the four methods (fixed 8,627, sliding 11,419, structure-aware 13,377, semantic 11,927), each exactly 512 dimensions, stored as ~88.57 MiB of matrices.
- Batch size 8 was used (vs. the model's default of 1); a side-by-side numerical check confirmed the largest difference between batch-1 and batch-8 results was under 1.8×10⁻⁷ — effectively zero, just floating-point noise.
- Full run took ~205 seconds of actual stage time (about 4.2 minutes including input validation); a repeat run with everything cached took ~23 seconds with zero new inference, byte-identical matrices.
- Checks: 99 pipeline + 17 cleaning = 116 tests passed.

---

## 7. Phase 8 — Late chunking (21 September 2026)

This is the fifth and final chunking method, and the trickiest one conceptually: instead of embedding a small chunk directly, we embed a *larger window* (typically 5 chunks plus 1 neighboring chunk on each side, up to 2,048 tokens total) and then pool *down* to just the target chunk's portion of that window — so the chunk's final vector has "seen" its surrounding context before being compressed.

- Uses the *exact same chunk spans* as the fixed-size baseline (8,627 of them) — this is deliberate, so "late" and "fixed" are a perfectly matched pair that differ *only* in whether context was seen during embedding.
- **Result:** 1,804 encoder windows built, maximum 1,794 tokens (safely under the 2,048 cap — no budget fallback was ever needed). Total planned input across all windows was 3,022,356 tokens, about 1.37× the token cost of embedding the chunks independently (the price we pay for context-awareness).
- **Sanity check that matters:** when the window's full text and the target chunk were pooled "early" (whole window, no gaps) vs. pooled in the final late-chunking pipeline, the two methods' vectors differed by at most 1.49×10⁻⁸ — i.e., provably the same computation.
- A more interesting number: when the *same* chunk appeared in multiple overlapping windows (different neighboring context each time), the resulting vectors were highly but not perfectly similar — mean cosine similarity 0.975, minimum 0.888 across 3,270 comparisons. This is expected and informative: it shows late chunking's vectors genuinely shift depending on neighboring context, which is the whole point of the method, but the shift is moderate, not wild.
- Checks: 105 pipeline + 17 cleaning = 122 tests passed.

**At this point, all five chunking methods existed with embeddings. This is roughly the halfway point of the engineering work.**

---

## 8. Phase 9 — Retrieval and indexing (27 September 2026)

Now we needed to actually *search* each method's chunks for a given question.

- Built **5 separate FAISS exact search indexes** (one per method), using `IndexFlatIP` (exact inner-product search, no approximation) — **53,977 indexed vectors total** across all five.
- Every method searches the *entire* 60-document corpus with no filters or shortcuts — there is no "cheating" by narrowing the search space per question.
- Ranking: highest score wins; exact ties broken by chunk ID — deterministic, reproducible, no randomness.
- Retrieved up to K=1, 3, 5, 10 chunks; assembled into a final context under a shared **1,536-token evidence budget** (measured in Jina content tokens), skipping any candidate too big to fit and trying the next one instead of truncating mid-sentence.
- Ran this on the 20-question pilot: 100 searches, 100 assembled contexts.
- **Independent numerical audit:** recomputed every FAISS score independently using plain float64 dot products; the largest disagreement found was 1.62×10⁻⁷ (tolerance was set at 2×10⁻⁶) — confirms the search results aren't an artifact of floating-point shortcuts.
- A concrete example worth remembering: for one pilot question about UCSC credit definitions, structure-aware's #1 result was the directly relevant passage, while fixed-size's #1 result was an unrelated rule from a different university's handbook — an early, small hint that chunking method genuinely changes what gets retrieved first, which the final results (Section 14) later confirm at scale.
- Checks: 116 pipeline + 17 cleaning = 133 tests passed.

---

## 9. Phase 10 — Grounded answer generation, first attempt (27 September 2026)

With retrieval working, we needed an LLM to actually read the retrieved evidence and answer the question — grounded strictly in the evidence, instructed to abstain if the evidence didn't support an answer.

- **Original generator:** GroqCloud, model `qwen/qwen3.8-27b`, free tier, temperature effectively 0 (1×10⁻⁸), seed 42.
- Deliberately conservative local rate limits (24 requests/min, 900/day) — stricter than Groq's own published free-tier limits, to avoid ever hitting a hard quota wall mid-experiment.
- **First live test:** only 15 of the prepared 100 requests were actually sent (3 pilot questions × 5 methods), to sanity-check the whole pipeline before spending the rest of the free quota. All 15 succeeded, 0 errors, 0 retries. Token usage: 35,888 input + 1,266 output = 37,154 tokens total.
- A concrete finding from this tiny sample: for one question, fixed-size chunking's retrieval found the correct passage only at rank 10 and then *dropped it from the context* (because of the token budget), so the model correctly abstained — while structure-aware retained the same passage and gave a correct, specific answer (40% continuous assessment, 80% attendance). This is exactly the kind of chunking-causes-failure story the whole project is trying to quantify.
- Checks: 129 pipeline + 17 cleaning = 146 tests passed.

**At this point (27 September), we had a fully working pipeline but only a 20-question draft pilot with zero human-approved questions — nowhere near a reportable result yet.**

---

## 10. The benchmark problem — why this took the longest

Between Phase 10 and what we now call "Phase 11," most of the real-world time went into getting an actually *trustworthy, reviewed* set of benchmark questions — this is the part that can't be rushed by writing more code, because it needs real human judgment about what's a fair, answerable, correctly-evidenced question.

### 10.1 The team's 153-question import (6 October 2026)
The team had collected 160 candidate questions in a spreadsheet. We built an importer that:
- Excluded 7 problematic rows (3 duplicate pairs + 4 individually flagged issues — e.g. a medical-certificate question whose "evidence" was actually about something unrelated, a duration question answered with study-mode info instead, a page reference pointing to page 220 of a 41-page document).
- Kept **153 retained drafts**.
- Automatically matched candidate evidence passages for 56 records, recovered 12 blank-evidence records from their stated reference answer, and corrected 24 page references.
- Found that **127 of the 153 records** still had at least one unresolved issue needing a human decision (scope too vague, answerability label uncertain, or no exact matching passage found).
- Exported a clean correction worksheet: 77 rows for Aysha, 50 for Haleema, to fix by hand.

### 10.2 The assistant-authored 100-question benchmark (6 October 2026)
While waiting on the team's correction queue, at Vithusan's explicit request, the assistant authored a *separate* 100-question answerable benchmark directly from 10 source PDFs (60 handbook + 40 lecture-style questions, 10 per source document), with exact character-offset evidence spans and rubrics, visually checking 50 original PDF pages for accuracy. This was explicitly labeled **AI-authored, not independently human-reviewed** at the time of creation, kept in a separate workspace from the team's 153 drafts, and never allowed to masquerade as a "held-out test."

### 10.3 Reconciliation into the final 120-question benchmark
These threads converged into the final frozen benchmark used for all reported results: **120 questions** — 100 answerable (the assistant-authored set, subsequently reviewed and approved) plus 20 newly authored unanswerable/ambiguous questions designed to test whether the system correctly *refuses* to answer when it shouldn't. **All 120 rows now carry `review.status: "approved"`** with a named independent human reviewer, PDF verification, and a content hash binding the approval to the exact question text/evidence/rubric — confirmed directly by inspecting the benchmark files, not just taken on faith.

---

## 11. The provider switch: Groq → OpenAI (7 October 2026)

When it came time to run the full 600-request benchmark (120 questions × 5 methods), we discovered **Groq's free tier could not be upgraded to a paid billing plan** — it was simply unavailable at the time. After checking whether the Cursor platform's own API access could substitute (it couldn't, for this use case) and confirming Groq's paid tier wasn't enabling billing, Vithusan made the call: switch to **OpenAI's paid, pay-as-you-go API**, self-funded, with no hidden "zero-budget" claim anymore.

This required real code changes, not just a config flag:
- Replaced the Groq client with an OpenAI client (`gpt-4o-mini`).
- **Found and fixed a real bug** in the process: OpenAI's API returns a *dated snapshot name* for the model (e.g. `gpt-4o-mini-2024-07-18`) instead of echoing back the exact model name requested. Our safety check was treating this as "wrong model returned" and rejecting every single response — which meant the *first* live smoke test made 30 real, paid API calls that all got silently discarded as failures before we caught and fixed this. Fixed by accepting any returned name that matches the requested name or that name plus a dated suffix.
- Swapped the tokenizer from a pinned Hugging Face tokenizer to a pinned, hash-verified `tiktoken` bundle (OpenAI's own encoding), downloaded once and stored locally so no runtime network dependency exists.
- Updated the wire format (OpenAI uses `max_completion_tokens` instead of `max_tokens`, no `reasoning_effort` field, different endpoint URL).
- Re-ran the smoke test after the fix: 30/30 succeeded.
- Updated `requirements-phase2.txt` to swap Groq packages for OpenAI/tiktoken packages.
- Full test suite re-verified at 155/155 passing after all of this.

---

## 12. The full 600-request generation run (7 October 2026)

With the provider switch validated, we ran the complete experiment: **120 questions × 5 chunking methods = 600 generation requests.**

- **Result: 600/600 successful, 0 failed, 0 retries needed.**
- Took 17.2 minutes.
- Token usage: 1,313,239 prompt tokens + 38,165 completion tokens.
- **Estimated cost: approximately $0.22** — this is the entire paid budget spent on the primary generation run.
- Ran the offline evaluation pass immediately after: all 5 methods showed 120/120 answers complete, 0 missing, 0 failed.

---

## 13. RAGAS — the automated LLM judge (7 October 2026)

Retrieval metrics (did the right text get retrieved?) don't tell you whether the *generated answer* is actually faithful to the evidence or relevant to the question. For that we used **RAGAS**, an established open-source LLM-judge framework, specifically four of its metrics: Faithfulness, Answer Relevancy, Context Precision, and Context Recall.

- Judge model: OpenAI `gpt-4o-mini` for the judgments, `text-embedding-3-small` for the judge's own internal embedding needs — **deliberately different from and separate from** the Jina embedding used for retrieval, so there's no risk of the judge's embedding choice accidentally confounding the retrieval comparison.
- First ran over **all 100 answerable questions × 5 methods = 500 samples**: all 5 methods completed, 100/100 scored each, 0 failures, ~14.25 minutes total (after we found that the default single-threaded setting would have taken over 6 hours, and added a configurable concurrency option, verified safe with a small smoke test first before committing to the full run).
- Headline pilot-wide result at this stage: `fixed_size` scored lowest on nearly every metric; `semantic` and `structure_aware` scored consistently higher — a first hint of the pattern the final, properly-split test-set result later confirms with statistical backing (Section 16).

---

## 14. Finalizing the benchmark: the development/test split and freeze (7 October 2026)

Having a reviewed set of questions isn't the same as having a *valid* held-out test set — the question/test-set design has its own strict rules, defined back in the original Phase 2 protocol, specifically to prevent us from accidentally cheating (e.g. tuning on data we later "test" on).

### 14.1 The blocker we found
Checking the benchmark manifest, we discovered that **all 17 real document groups were already marked as "reserved for development"** — a leftover placeholder from before any test-split work had begun. This made it *impossible* to create any test split at all without first deciding which document groups to free up. We surfaced this explicitly to Vithusan rather than silently working around it, since it's a dataset-validity decision, not a code bug.

### 14.2 The split
Vithusan authorized freeing up a subset of document groups for testing, and the team self-approving the experiment protocol (the originally planned external reviewer wasn't available). We then computed a split using **only document-group identity and size — never any model score or result** — to avoid any appearance of result-driven tuning:

- **40 development questions** (30 answerable, 10 unanswerable).
- **78 test questions** (70 answerable, 8 unanswerable).
- **2 diagnostic questions** (the ambiguous ones, excluded from primary scoring per protocol).

This matches the protocol's pre-declared target of "40 development + 80 held-out" almost exactly (the 2 ambiguous ones come out of the "80" side as diagnostic). One rebalancing pass was done: the first draft put all 3 `table_lookup` questions in development (0 in test), which fights the protocol's own category-coverage goal — swapped two equal-sized document groups between dev/test to move all 3 into test instead, keeping the 40/78/2 totals exactly the same.

### 14.3 Honest corrections before calling it "approved"
The protocol document still said "Vithusan selected GroqCloud's free tier" and "the paid budget is zero" — both false after Section 11's switch. We corrected this explicitly before marking the protocol `approved`, rather than approving a document containing a known-false statement.

### 14.4 The freeze
Every re-split question's review hash was refreshed (since changing a question's `split` field legitimately invalidates its prior approval hash, by the system's own design — this caught a real bug in our first hash-recomputation attempt, which we found and fixed by using the exact project function instead of reimplementing it). Then:

```
rag_pipeline.evaluation.validate_benchmark --freeze
```

This wrote a **permanent, tamper-evident snapshot** (`frozen_snapshot.json`) of the exact benchmark content, using an exclusive file-create so it's structurally impossible to silently re-freeze over it later. From this point forward, the 78-question test split is locked.

---

## 15. Building the "final test scoring" capability (7 October 2026)

Here's a subtlety worth explaining clearly: up to this point, the evaluation code had a **hard-coded safety rule** that only let you score the `pilot` split, with an explicit error message: *"final test scoring is not enabled."* This was deliberate — it's a guardrail against accidentally reporting "final" numbers before the benchmark was actually frozen. Since the benchmark was now properly frozen, this code path needed to be built for the first time:

- Added a `--final-test-scoring` flag (mutually exclusive with the old `--provisional` flag) that **requires** `frozen_snapshot.json` to exist before it will run at all.
- Generalized the internal data-loading function to read from the frozen `test.jsonl` file instead of `pilot.jsonl`, while still re-using the *same* underlying retrieval/generation traces (filtering down to just the 78 relevant question IDs).
- **A real discovery along the way:** editing the benchmark manifest (for the freeze) had changed its file hash, which correctly tripped an existing safety check ("Benchmark changed since retrieval") — because the existing retrieval/generation runs' recorded provenance pointed at the *old* manifest hash. Rather than weakening that check, we re-ran retrieval and generation in **cache-only mode** — meaning it replayed the exact same previously-computed vectors and answers, with **zero new model inference and zero new paid API calls** — to produce fresh runs stamped against the current, frozen manifest. Confirmed: 120/120 retrieval cache hits, 600/600 generation cache hits, **0 live API requests**, **$0 additional cost**.
- Ran the real thing end to end: **78 test questions × 5 methods = 390 scored records, 0 missing, 0 failed.**

---

## 16. The final results

With the frozen test split properly scored, we computed two independent sets of metrics and tested whether the differences between methods were statistically real or just noise.

### 16.1 Retrieval metrics (exact character-span matching, no LLM involved)

| Method | Recall@10 | MRR@10 | Context Recall | Context Complete |
|---|---|---|---|---|
| fixed_size | 0.493 | 0.370 | 0.474 | **0.457** |
| sliding_window | 0.633 | 0.440 | 0.550 | 0.529 |
| semantic | 0.581 | 0.416 | 0.567 | 0.557 |
| late | 0.600 | 0.413 | 0.600 | 0.600 |
| structure_aware | 0.652 | 0.417 | 0.638 | **0.629** |

`context_complete` is our strictest, primary metric: did *every single piece* of required evidence make it into the final context actually shown to the generator?

### 16.2 Statistical significance (not just averages)

We used a paired cluster-bootstrap test (2,000 resamples, seed 42, resampling by document/question family so related questions aren't treated as independent) to check which differences are real:

| Comparison | Difference | 95% Confidence Interval | Statistically significant? |
|---|---|---|---|
| fixed_size vs structure_aware | −0.171 | [−0.272, −0.043] | **Yes** |
| fixed_size vs late | −0.143 | [−0.257, −0.043] | **Yes** |
| sliding_window vs structure_aware | −0.100 | [−0.186, −0.014] | **Yes** |
| fixed_size vs sliding_window | −0.071 | [−0.143, −0.014] | **Yes** |
| fixed_size vs semantic | −0.100 | [−0.271, 0.043] | No |
| (all other pairs) | small | crosses zero | No |

**`fixed_size` is significantly worse than three of the other four methods.** `structure_aware` has the best point estimate and is significantly ahead of both `fixed_size` and `sliding_window`.

### 16.3 RAGAS judge metrics, scored on the same 78-question frozen test split

| Method | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|
| fixed_size | 0.773 | 0.693 | 0.660 | 0.715 |
| sliding_window | 0.796 | 0.686 | 0.666 | 0.807 |
| semantic | 0.835 | 0.725 | 0.651 | 0.817 |
| late | 0.848 | 0.704 | 0.655 | 0.768 |
| structure_aware | 0.803 | 0.754 | 0.651 | 0.825 |

Running the same significance test on these: `fixed_size` is significantly worse than `semantic` on faithfulness and significantly worse than `semantic`/`structure_aware` on context recall. `structure_aware` is significantly ahead of `sliding_window`/`late` specifically on answer relevancy. **Context precision shows no significant difference between any pair of methods** — we report this honestly as inconclusive rather than pretending it supports a ranking.

### 16.4 The consolidated finding

Two independent measurements (exact-span retrieval metrics and an LLM judge) **converge on the same conclusion**: `fixed_size` chunking is the measurable lower bound, and `structure_aware` (with `semantic`/`late` close behind) is the upper bound. With only 10 independent question-family clusters in the 78-question test split, we are careful to say "fixed_size measurably underperforms the other four" rather than claiming a strict 1-2-3-4-5 ranking — the non-significant comparisons are genuinely inconclusive, not secretly in anyone's favor.

### 16.5 Why it fails: the automated failure taxonomy

We also built an automated (not manual/subjective) breakdown of *why* retrieval failed when it did, using diagnostic data already recorded during context assembly — no new model calls needed:

- Across 500 answerable question/method pairs (the full 120-question pilot, all 5 methods), **266 (53%) were fully successful**.
- Of the failures: **285 were "retrieval miss"** (the right evidence never appeared in the top-10 results at all — by far the dominant failure mode for every method), **19 were "overflow"** (the right chunk *was* retrieved but got dropped because the shared 1,536-token context budget ran out), and just **1 was genuine "boundary fragmentation"** (evidence split across chunks where no single chunk alone was complete).
- `sliding_window` stood out with 9 overflow cases vs. 2–3 for every other method — its overlapping windows cost more of the shared token budget per chunk, so it runs out of room more often even when it retrieves correctly.
- For the 18 unanswerable questions × 5 methods = 90 pairs, we checked whether the model answered confidently using a *different institution's* handbook (or hallucinated from unrelated real documents) without flagging the mismatch: **9 of 90 (10%) did this**, spread across all five methods roughly equally — a shared system limitation, not something specific to one chunking method.

---

## 17. A decision we explicitly declined, and why

At one point, after the per-answer *human* review step was deprioritized by the team, the question came up of whether the assistant could simply fill in the human-review template itself. **This was declined.** The reason is not a technical limitation — it would have been easy to write the JSON. The reason is that the review schema requires a field (`reviewer_kind: "human"`) specifically so that anyone reading the results later knows whether a real person or a model made the judgment call. Writing `"human"` into that field while an AI did the actual grading would be **fabricating provenance in a dataset meant to support an academic thesis** — the kind of thing that, if ever discovered later (by an examiner, or a future student reusing this benchmark), would call the whole project's integrity into question. The team agreed, and the project instead relies on **RAGAS** for answer-quality scoring, which is always transparently labeled as an LLM judge in every output file — never disguised as anything else. This is documented explicitly in the methodology notes so a reader never mistakes the resulting `null` human-correctness fields for an oversight.

---

## 18. Independent reproduction

Per the project's own completion checklist (in `PLAN.md`), a key integrity requirement is that **someone other than the person who ran the main experiment reproduces a result independently**. On 7 October 2026, both Aysha and Haleema independently reproduced results and confirmed they matched. (For the thesis write-up, it would strengthen the reproducibility section to record exactly which command each of them ran and which specific numbers they checked — worth a quick one-line note from them.)

---

## 19. Where things stand right now

**Done and verified:**
- All five chunking methods implemented, tested, and run on the full 60-document corpus.
- Shared embedding model held identical across all methods (the controlled variable).
- 120-question benchmark, independently human-reviewed and approved, **frozen** into a tamper-evident snapshot.
- Full 600-answer generation run (0 failures, ~$0.22).
- Retrieval metrics + RAGAS automated judge, both computed on the frozen 78-question test split specifically (not mixed with development data).
- Paired statistical significance testing on both metric families.
- Automated, evidence-based failure-mode breakdown.
- Independent reproduction by two other team members, confirmed.
- 155/155 automated code tests passing throughout every change made in this process.

**Explicitly not done, and genuinely needing human/team input (not something the assistant can substitute for):**
- Per-answer human correctness/faithfulness rubric scoring was deliberately skipped in favor of RAGAS (Section 17) — this is a documented methodological choice, not an oversight, but should be stated plainly in the thesis methodology section.
- "Extraction quality" failure analysis (did each chunker preserve the right atomic facts) needs a human reading actual chunk text — not automatable from existing logs.
- A single, consolidated "Limitations" section for the thesis (currently the honest caveats are scattered correctly across several technical docs, not yet merged into one place).
- **The actual thesis document itself** — everything described above lives in technical project documentation (`PLAN.md`, `rag_pipeline/docs/phase11_final_results.md`, `phase11_failure_taxonomy.md`, `experiment_protocol.md`) and is accurate, but it is not yet written in the university's required thesis format (Introduction, Literature Review, Methodology, Results, Discussion, Conclusion, references).
- Supervisor review and institutional submission logistics.

---

## 20. Where to find everything (for your own verification)

| What | Where |
|---|---|
| The original plan and full phase-by-phase work log | `PLAN.md` |
| Final results (retrieval + RAGAS + significance, test split) | `rag_pipeline/docs/phase11_final_results.md` |
| Failure-mode breakdown | `rag_pipeline/docs/phase11_failure_taxonomy.md` |
| The experiment protocol (pre-declared rules, now with the 7 Oct corrections) | `rag_pipeline/docs/experiment_protocol.md` |
| The frozen benchmark itself | `data/benchmark/benchmark_120_20261007/` (`development.jsonl`, `test.jsonl`, `diagnostic.jsonl`, `frozen_snapshot.json`) |
| The full 600-answer generation run | `data/outputs/rag/phase11_benchmark120_full_openai_20261007/` |
| The RAGAS judge results (test split) | `data/outputs/rag/phase11_benchmark120_ragas_judge_test_split_20261007/` |
| The final test-scoring run | `data/outputs/rag/phase11_final_test_scoring_20261007/` |
| Per-phase detailed completion reports | `rag_pipeline/docs/phase1_completion.md` through `phase10_completion.md`, `phase2_status.md`, `phase11_status.md` |




# Documentation Extraction Report

This report extracts and organizes factual details from 18 markdown files in `rag_pipeline/docs/`. Each section follows the requested structure: (1) phase/topic & objective, (2) what was implemented, (3) exact numeric results, (4) problems/bugs and fixes, (5) key decisions and rationale, (6) limitations/caveats.

---

## 1. `rag_pipeline/docs/phase1_completion.md`

**Date:** 19 September 2026

### 1. Phase/topic and objective
Phase 1: shared foundation and corpus-inspection command. Explicitly states no chunking, tokenizer selection, embeddings, retrieval, generation, or benchmark work has run yet.

### 2. What was implemented
- `common/load_corpus.py` — consistency checks, exact text, safe metadata loading.
- `common/provenance.py` — document/page/heading/region types and span mapping.
- `common/chunk_schema.py` — deterministic IDs and validated source-slice records.
- `common/config.py` — strict configuration and path rules.
- `common/run_manifest.py` — environment, source hashes, run status, output hashes.
- `cli.py` — complete inspection workflow and readable reports.
- `02_data_cleaning/validate_corpus.py` — optional external report path for read-only validation (original default behavior preserved).
- `requirements-phase1.txt` — four pinned runtime dependencies, included by full requirements file.
- CLI command: `.venv/bin/python -m rag_pipeline inspect --run-id phase1_my_review`

### 3. Exact numeric results
| Metric | Value |
|---|---|
| Default corpus | 60 documents: Aysha 12, Haleema 24, Vithusan 24 |
| Physical page records | 6,215 |
| Explicitly excluded pages | 44 |
| Other blank prepared pages | 172 |
| Eligible continuous text regions | 169 |
| Foundation tests | 17 passed |
| Existing cleaning tests (incl. external-report regression) | 17 passed |
| Original PDFs and prepared files checked | 187 file hashes, all unchanged |
| `uv pip check` | 133 packages, passed |
| Isolated run Python version | 3.11.16 on Linux |

Verified run location: `data/outputs/rag/phase1_clean_environment_20260919/` with `inspection_report.md`, `inspection_examples.json`, `validation_report.json`.

### 4. Problems/bugs found
None reported; the run passed in both development and isolated environments with identical corpus identity.

### 5. Key decisions and rationale
- No commit/push performed; outputs are git-ignored.
- Isolated runtime used only the four pinned Phase 1 packages to validate minimal dependency footprint.
- Token fields left nullable in schema since tokenizer decision belongs to Phase 2.

### 6. Limitations/caveats
- Checks verify integrity/provenance only; they do not reconstruct missing images or verify semantic correctness of extracted tables/formulas.
- All later methods must apply the same blank/excluded-page policy.
- Next phase (Phase 2) will cover experiment protocol, benchmark, model feasibility.

---

## 2. `rag_pipeline/docs/phase2_status.md`

**Date:** 20 September 2026

### 1. Phase/topic and objective
Phase 2 handover: status "ready for Phase 3 development." LangChain and GPU preparation complete; final benchmark review/freezing deferred until before Phase 11.

### 2. What was implemented
- `PreparedCorpusLoader` — LangChain `BaseLoader` returning 169 `Document` regions from 60 sources, preserving exact text, offsets, pages, exclusions.
- `JinaEmbeddings` — implements LangChain `embed_documents()` and `embed_query()`, plus token-output helper for later late-chunking.
- `ChatGroq` factory + compatible pinned dependencies (no API requests made; exact free model unset).
- Model configuration, offline teaching example, readable guide.
- Commands: `.venv/bin/python -m rag_pipeline.examples.inspect_langchain`, `.venv/bin/python -m rag_pipeline inspect-langchain --run-id my_langchain_review`

### 3. Exact numeric results
- Resource decision: local embeddings; CPU tested; RTX 3050 4 GB CUDA access confirmed outside sandbox.
- Embedding model: Jina v2 small English; passed CUDA token-output, pooling, padding, query-consistency, overflow checks at 2,048 total tokens.
- 2,048-token model test: ~1.1 seconds per synthetic forward pass; ~1.37 GiB peak process RAM.
- Pilot: 20 drafts — 18 answerable, 1 unanswerable candidate, 1 ambiguous. 7 source PDFs, 13 visually inspected reference pages.
- Verification: **39 pipeline tests + 17 cleaning tests = 56 distinct tests passed**.
- All 20 draft records passed integrity validation.
- 122 prepared input files in Phase 1 snapshot — unchanged hashes.
- CUDA probe (RTX 3050): peak PyTorch allocation ~667 MiB, 798 MiB reserved, batch size 1 default.

Artifacts: `data/outputs/rag/model_probe_20260920/`, `phase2_probe_20260920/`, `phase2_preparation_20260920/`, `phase2_langchain_cuda_20260920/`, `phase2_langchain_cpu_20260920/`, `phase2_langchain_loader_20260920/`, `phase2_langchain_checks_20260920/`.

Local benchmark files: `data/benchmark/pilot.jsonl`, `pilot_review.md`, `benchmark_manifest.json`, `annotation_reviews.csv`, `pilot_source_pages/`.

Commands:
```bash
.venv/bin/python -m rag_pipeline.evaluation.validate_benchmark
.venv/bin/python -m unittest discover -s rag_pipeline/tests -v
```

### 4. Problems/bugs found
None described as bugs; `--require-ready` and `--freeze` correctly refused the unfinished benchmark (expected behavior, not a bug).

### 5. Key decisions and rationale
- **Workflow update from Vithusan**: use 20 questions for development through answer generation (Phases 3–10); larger reviewed benchmark supplied later before Phase 11. Draft pilot scores are debugging observations, not final findings.
- Jina v2 small English selected for embeddings.
- Groq free tier preferred for generation.
- Provisional target: 40 development + 80 test (both currently zero records).
- Reviewed pilot questions may move to development, never to test.

### 6. Limitations/caveats
- A validation success is not a human quality certification.
- Human approvals currently: **zero**.
- Phase 3 has not started as of this doc; prerequisites ready.

---

## 3. `rag_pipeline/docs/phase3_completion.md`

**Date:** Implemented 20 September 2026 — Fixed-size chunking (baseline, method 1 of 5)

### 1. Phase/topic and objective
Baseline token-window chunker: divide each eligible region into consecutive windows of **256 content tokens max, zero overlap**.

### 2. What was implemented
| File | Responsibility |
|---|---|
| `common/langchain_loader.py` | Load prepared text as LangChain `Document` regions |
| `common/tokenization.py` | Verify pinned Jina tokenizer; count tokens, retain offsets |
| `chunking/fixed_size/chunker.py` | `FixedSizeSplitter`, LangChain `TextSplitter` subclass |
| `chunking/fixed_size/run.py` | Process regions, create chunk records, save reports |
| `configs/fixed_size.json` | Versioned baseline settings |
| `tests/test_fixed_size.py` | Boundary, Unicode, metadata, token-limit checks |

Commands:
```bash
.venv/bin/python -m rag_pipeline chunk-fixed --run-id my_fixed_sample --sample
.venv/bin/python -m rag_pipeline chunk-fixed --run-id my_fixed_full
```
Dependency added: `langchain-text-splitters==0.3.8` in `requirements-phase2.txt`.

### 3. Exact numeric results
| Check | Result |
|---|---|
| Three-member sample | 3 documents, 555 chunks |
| Full corpus | 60 documents, 169 eligible regions, **8,627 chunks** |
| Content tokens per chunk | 2–256; mean **253.18**; median 256 |
| Max encoder input | 258 tokens (below tested 2,048 limit) |
| Eligible character coverage | 11,424,026 characters — 100% |
| Overlap / empty chunks / overflows | 0 / 0 / 0 |
| Cross-page chunks | 4,994 |
| Boundary backoff operations | 143 |
| First full-run processing time | ~56 seconds |
| Tests | 51 pipeline + 17 cleaning = passed |
| Prepared corpus total characters | 11,424,676 (650-char difference is whitespace outside eligible regions) |

Full/repeat run directories: `phase3_fixed_full_20260920`, `phase3_fixed_repeat_20260920`. Verification artifacts: `phase3_verification_20260920/boundary_review.md`, `verification.json`.

Full chunk has 258 encoder input tokens (256 content + 2 special tokens).

### 4. Problems/bugs found
- Boundary backoff needed 143 times when cutting a WordPiece word would change its independent tokenization — handled by moving boundary back until the slice retokenizes within the cap.
- No overlap/empty/overflow chunks occurred.

### 5. Key decisions and rationale
- Custom override of LangChain's splitter needed because token decoding can change case/spacing and first-match text search could select wrong occurrence; chunk text must be exact substring of source.
- Leading whitespace → first chunk; inter-token whitespace → chunk on left; trailing whitespace → last chunk.
- Excluded/blank pages remain barriers; normal page boundaries may be crossed.
- Chunk IDs and canonical-record hash exclude `run_id` for reproducibility checks.

### 6. Limitations/caveats
- Boundaries may fall inside sentences/words/lists/tables — this is "part of the method we will evaluate," not corrected.
- 256-token setting is a starting baseline, not proven optimum.
- Results establish segmentation integrity, not semantic extraction accuracy or answer quality.
- Next: Phase 4 — sliding-window chunking.

---

## 4. `rag_pipeline/docs/phase4_completion.md`

**Date:** Implemented 20 September 2026 — Sliding-window chunking (method 2 of 5)

### 1. Phase/topic and objective
Sliding-window chunking using Phase 2's starting setting: **256 content tokens, 64 source tokens overlap**, nominal stride 192.

### 2. What was implemented
| File | Purpose |
|---|---|
| `chunking/sliding_window/chunker.py` | `SlidingWindowSplitter`, settings checks, overlap metadata |
| `chunking/sliding_window/run.py` | Entry point |
| `configs/sliding_window.json` | 256-token / 64-overlap config |
| `common/token_windows.py` | Shared exact token/character boundary rules (also used by fixed-size) |
| `common/chunk_run.py` | Shared output writer, coverage checks, stats, manifests |
| `tests/test_sliding_window.py` | Overlap, tail, Unicode, mapping, zero-overlap, failure checks |

Commands:
```bash
.venv/bin/python -m rag_pipeline chunk-sliding --run-id my_sliding_sample --sample
.venv/bin/python -m rag_pipeline chunk-sliding --run-id my_sliding_full
```

### 3. Exact numeric results
| Check | Result |
|---|---|
| Deterministic member sample | 3 documents, 739 chunks |
| Full corpus | 60 documents, 169 regions, **11,419 chunks** |
| Unique eligible text covered | 11,424,026 characters — 100% |
| Repeated characters from overlap | 3,764,857 |
| Content tokens in all chunks | 2,904,243 |
| Max content / encoder input tokens | 256 / 258 |
| Adjacent pairs within regions | 11,250, all sharing exactly 64 source tokens |
| Empty / overflow / redundant-tail chunks | 0 / 0 / 0 |
| Observed full-run time | ~82 seconds |
| Automated tests | **62 pipeline + 17 cleaning = 79 passed** |

Comparison table (Fixed vs Sliding):
| Measure | Fixed-size | Sliding window |
|---|---:|---:|
| Chunks | 8,627 | 11,419 |
| Content tokens to embed | 2,184,219 | 2,904,243 |
| Unique eligible characters | 11,424,026 | 11,424,026 |

Sliding windows create **2,792 additional chunks** and ~**33% more content tokens**.

Verification: All checks passed; two complete sliding runs have identical canonical records; all 8,627 Phase 3 baseline spans match; all 122 prepared-input hashes unchanged; fixed-size CLI rerun on 3-member sample matched 555 original canonical records.

Output directories: `phase4_sliding_full_20260920`, `phase4_verification_20260920`.

### 4. Problems/bugs found
No overlap reductions occurred in the full run (zero), meaning the "no new evidence" safety reduction rule was never triggered.

### 5. Key decisions and rationale
- Overlap measured in original region's token stream.
- Both starts and ends must advance; if a candidate adds no new evidence, overlap is reduced (none needed in this run).
- `overlap` must satisfy `0 <= overlap < chunk_size`.
- Overlap never crosses region/document boundaries.
- `method_metadata` fields recorded: `requested_overlap_tokens`, `nominal_stride_tokens`, `overlap_with_previous_region_tokens`, `overlap_with_previous_characters`, `overlap_text_content_tokens`, `overlap_reduced`.

### 6. Limitations/caveats
- Total chunk characters ≈1.33× unique eligible characters.
- Overlap can repeat irrelevant material and crowd out other evidence from generator context — to be measured later.
- Timings are observed runs, not controlled performance study.
- Next: Phase 5 — structure-aware chunking.

---

## 5. `rag_pipeline/docs/phase5_completion.md`

**Date:** Completed 21 September 2026 — Structure-aware chunking (method 3 of 5)

### 1. Phase/topic and objective
Chunking driven by document headings and paragraph/list/table boundaries, same 256-content-token cap, **no overlap**.

### 2. What was implemented
| File | Responsibility |
|---|---|
| `chunking/structure_aware/headings.py` | Validate/combine heading hints |
| `chunking/structure_aware/units.py` | Locate paragraph, list, table, sentence units |
| `chunking/structure_aware/chunker.py` | LangChain `TextSplitter`, section paths, packing, fallbacks |
| `chunking/structure_aware/reporting.py` | Heading decisions and fallback reports |
| `chunking/structure_aware/run.py` | Entry point |
| `configs/structure_aware.json` | Settings, 256-token cap |
| `tests/test_structure_aware.py` | Boundary, hierarchy, table, fallback, glyph checks |

Heading sources: prepared heading (validated vs. source line), font candidate (large font ≥15% larger than median candidate font size), PDF outline entry (matched to destination page).

Commands:
```bash
.venv/bin/python -m rag_pipeline chunk-structure --run-id my_structure_sample --sample
.venv/bin/python -m rag_pipeline chunk-structure --run-id my_structure_full
```

### 3. Exact numeric results
| Check | Result |
|---|---:|
| Documents / eligible regions | 60 / 169 |
| Structure-aware chunks | **13,377** |
| Content-token range | 1–256 |
| Mean / median content tokens | 163.28 / 188 |
| Unique eligible characters covered | 11,424,026 — 100% |
| Overlap / empty / overflow chunks | 0 / 0 / 0 |
| Accepted headings / candidate hints | 8,448 / 89,071 |
| Sections | 6,209 |
| Chunks without a heading | 86 |
| Chunks using table-row / sentence / token fallback | 1,372 / 940 / 261 |
| Observed full-run processing time | ~83 seconds |
| Automated checks | **78 pipeline + 17 cleaning tests passed** |
| Chunks below 64 content tokens | 2,375 |

Size comparison across methods so far:
| Method | Chunks | Mean content tokens | Total content tokens |
|---|---:|---:|---:|
| Fixed-size | 8,627 | 253.18 | 2,184,219 |
| Sliding window | 11,419 | 254.33 | 2,904,243 |
| Structure-aware | 13,377 | 163.28 | 2,184,164 |

Output dirs: `phase5_structure_full_v2_20260920` (usable), `phase5_structure_full_20260920` (earlier failed glyph-only unit case, retained for diagnosis), `phase5_structure_repeat_20260921` (independent repeat), `phase5_verification_20260920` (audit).

### 4. Problems/bugs found
- Earlier run `phase5_structure_full_20260920` recorded a **failed glyph-only unit case** (tokenizer-ignored decorative glyphs with no encodable text) — fixed in `full_v2` by attaching such glyphs to a neighboring unit without deletion.
- Example of **questionable heading hierarchy**: a contents-like label inferred as ancestor of a history section — flagged as a limitation, not fixed (heading paths are diagnostic metadata, not corrected).

### 5. Key decisions and rationale
- Heading acceptance rules: reject common prose, bullet items, column-like lines, contents entries with dot leaders, contact lines, unsupported numbered list entries.
- Outline levels > numbering depth > prepared inferred levels for hierarchy; unknown level becomes level 2.
- Repeated current heading at same level = continuation, not new section.
- Oversized units split via table rows → sentences → token windows (in that fallback order).
- All acceptance/rejection/duplicate decisions saved for review.

### 6. Limitations/caveats
- These are deterministic heuristics, not perfect document understanding.
- Short sections retained rather than merged to fill token budget.
- Table reading order and missing image content remain preparation limitations.
- Heading paths are not added to embeddings in this phase.
- Next: Phase 6 — semantic chunking.

---

## 6. `rag_pipeline/docs/phase6_completion.md`

**Date:** Completed 21 September 2026 — Semantic chunking (method 4 of 5)

### 1. Phase/topic and objective
Chunking using meaning-based boundaries: embed sentence/list/table-row units with Jina v2 small English, compare adjacent units by cosine distance, cut at a threshold.

### 2. What was implemented
| File | Purpose |
|---|---|
| `chunking/semantic/units.py` | Locate scoring units; preserve glyphs, split oversized units |
| `chunking/semantic/chunker.py` | LangChain splitter, threshold calc, packing, boundary reasons |
| `chunking/semantic/cache.py` | LangChain embedding cache with checksums |
| `chunking/semantic/reporting.py` | Score examples, boundary counts, cache/GPU measurements |
| `chunking/semantic/run.py` | Connects model, tokenizer, cache, writer |
| `configs/semantic.json` | Versioned settings |
| `tests/test_semantic.py` | Topic changes, caps, offsets, short regions, cache corruption |

Rules: final chunk ≤256 content tokens + 2 special tokens; minimum target 64 tokens before semantic cut; no overlap; threshold = region's 90th percentile (linear interpolation); ties require strictly greater distance; <3 units → pack without threshold; oversized unit split via tokenizer offsets before embedding; CUDA FP32, batch size 1, content-only mean pooling, L2 normalization.

Commands:
```bash
.venv/bin/python -m rag_pipeline chunk-semantic --run-id my_semantic_sample --sample
.venv/bin/python -m rag_pipeline chunk-semantic --run-id my_semantic_full
.venv/bin/python -m rag_pipeline chunk-semantic --run-id my_semantic_repeat --cache-only
```

### 3. Exact numeric results
Full-corpus run: `data/outputs/rag/phase6_semantic_full_20260921/`. Sample: `phase6_semantic_sample_20260921`.

| Measurement | Result |
|---|---:|
| Documents / eligible regions | 60 / 169 |
| Semantic chunks | **11,927** |
| Content-token min / max | 1 / 256 |
| Mean / median content tokens | 183.13 / 205 |
| Eligible characters covered | 11,424,026 — 100% |
| Overlap / empty / overflow | 0 / 0 / 0 |
| Scoring units (incl. short regions) | 103,574 |
| Chunks ending at semantic cut | 5,337 |
| Chunks ending because next unit would exceed cap | 6,421 |
| Region-ending chunks | 169 |
| Chunks below 64-token target | 181 |
| Regions too short for similarity scoring | 12 |
| Fragments from pre-splitting oversized units | 736 |

Size comparison (all 4 methods so far):
| Method | Chunks | Mean content tokens | Total content tokens |
|---|---:|---:|---:|
| Fixed-size | 8,627 | 253.18 | 2,184,219 |
| Sliding window | 11,419 | 254.33 | 2,904,243 |
| Structure-aware | 13,377 | 163.28 | 2,184,164 |
| Semantic | 11,927 | 183.13 | 2,184,166 |

Resources: sample ~55 seconds, encoding 6,583 new units; full run ~**642 seconds (10.7 minutes)**, 81,343 new units, 22,213 cache hits (not fully cold). Boundary inference ~376 seconds; model loading 2.1 seconds; cache operations ~151 seconds. Torch peak GPU memory ~294.7 MiB allocated / 304 MiB reserved. SQLite boundary cache ~351.3 MiB.

Verification: **89 pipeline + 17 cleaning = 106 tests passed**. Audit recomputed 103,399 scored boundaries from saved vectors; all 122 canonical prepared-file hashes matched Phase 1.

Cache-only repeat `phase6_semantic_repeat_20260921`: ~113 seconds, 103,556 cache hits, zero misses, no model load/inference. Canonical record fingerprint:
```
c00e17fc81051f80cd92b43788e59b3efd153d1c583e547e68b074966ba07d6b
```
Regression samples matched: fixed-size 555, sliding-window 739, structure-aware 775.

Example boundary scores: scholarship-to-network-services cut at cosine distance **0.310351**, above threshold **0.297359**; related Network Operating Centre sentences stayed together at distance **0.129897**.

### 4. Problems/bugs found
- An isolated extraction fragment produced a misleading similarity cut (example documented, not "fixed" — preserved as a known limitation/diagnostic).

### 5. Key decisions and rationale
- 90th percentile with linear interpolation chosen as predeclared starting rule, not optimal — "No held-out benchmark questions or answers were used."
- Content-only pooling chosen as the shared experimental adaptation (Phase 2 decision), differing from stock Jina pooling.
- Boundary vectors kept in separate cache from Phase 7's final retrieval embeddings.

### 6. Limitations/caveats
- Similarity is a model estimate, not proof of topic change.
- More than half of chunk endings are size-driven, not semantic.
- Next: Phase 7 — shared embeddings for the first four methods.

---

## 7. `rag_pipeline/docs/phase7_completion.md`

**Date:** Completed 21 September 2026 — Shared chunk embeddings for the four independently encoded methods

### 1. Phase/topic and objective
Create 512-dimensional embedding vectors for all chunks from the four independent chunking methods (fixed, sliding, structure, semantic) using the same Jina model.

### 2. What was implemented
| File | Responsibility |
|---|---|
| `embedding/model_adapter.py` | Pinned Jina implementation behind LangChain |
| `embedding/vector_cache.py` | Shared batched encoder, cache keys, vector checks |
| `embedding/inputs.py` | Validate finished chunks against manifests/corpus |
| `embedding/encode_chunks.py` | Ordered vector export, row maps, resume, locks, manifests |
| `configs/embedding.json` | Four input artifact paths, common inference batch size |
| `tests/test_embeddings.py` | Batching, corruption, overflow, mapping, recovery checks |

Shared settings: Jina `jinaai/jina-embeddings-v2-small-en`, FP32, NVIDIA GPU, 512 dims, content-only mean pooling, L2 normalization, empty doc/query prefixes, exact chunk text, retokenize/validate (never silently truncate), seed 42, deterministic Torch.

Commands:
```bash
.venv/bin/python -m rag_pipeline embed-chunks --run-id my_embedding_sample --sample
.venv/bin/python -m rag_pipeline embed-chunks --run-id my_embeddings
.venv/bin/python -m rag_pipeline embed-chunks --run-id my_embeddings --resume
.venv/bin/python -m rag_pipeline embed-chunks --run-id my_embedding_repeat --cache-only
```

### 3. Exact numeric results
Batch size = **8** (overrides adapter default; batch-one probe retained separately). Largest element difference across 2,845 batch-one/batch-eight sample vectors: below **1.8e-7**; peak reserved GPU memory in batch-eight probe ~412 MiB.

Full run: `data/outputs/rag/phase7_embeddings_full_20260921/`. Encoder identity: `d32e9c49b9612c8029b9cf12baf2954b2f969ed465a96511a71b844529cd3e83`.

| Method | Matrix shape | Matrix MiB | Observed stage seconds | Cache hits / new vectors |
|---|---:|---:|---:|---:|
| Fixed-size | 8,627 × 512 | 16.85 | 46.56 | 568 / 8,059 |
| Sliding window | 11,419 × 512 | 22.30 | 53.84 | 1,934 / 9,485 |
| Structure-aware | 13,377 × 512 | 26.13 | 56.65 | 1,570 / 11,807 |
| Semantic | 11,927 × 512 | 23.30 | 48.01 | 2,685 / 9,242 |
| **Total** | **45,350 × 512** | **88.57** | **205.06** | **6,757 / 38,593** |

New-vector inference ~165 seconds; preflight validation ~44 seconds; total ~4.2 minutes including preflight. Torch peak GPU memory ~361.8 MiB allocated / 416 MiB reserved; peak process RAM ~1.97 GiB. Final-chunk cache contains **41,284 unique text vectors**, ~164.9 MiB (matrices still have 45,350 rows for repeated-text occurrences).

Tests: **99 pipeline + 17 cleaning = 116 total passed**. All 122 canonical prepared files unchanged.

Repeat `phase7_embeddings_repeat_20260921`: 45,350 cache hits, zero misses/inference/model load; export/validation ~23.2 seconds + 43.6 seconds preflight; byte-identical matrices.

Earlier feasibility records: `phase7_embeddings_sample_20260921` (batch-one), `phase7_embeddings_batch8_sample_20260921` (batch-eight).

### 4. Problems/bugs found
None reported as bugs; the new tests specifically cover batch order, duplicate text, oversized inputs, wrong vector dims, NaN/zero vectors, corrupted caches, changed input text, interrupted jobs, concurrent writers.

### 5. Key decisions and rationale
- Batch size 8 chosen for Phase 7 (vs batch 1 default) since numeric difference was negligible (<1.8e-7).
- Identical text may reuse the same vector in cache, but each chunk occurrence keeps its own matrix row/source mapping.
- Cache key depends on exact text + encoder identity, not method name (methods treat identical text identically).
- Resume validates code/config/input identity, verifies completed methods, rebuilds incomplete outputs.

### 6. Limitations/caveats
- Observed timings are not a controlled cold-speed ranking since later methods may reuse earlier cache.
- Only a method with a complete stage manifest is ready for downstream use.
- Next: Phase 8 — late chunking.

---

## 8. `rag_pipeline/docs/phase8_completion.md`

**Date:** Implemented 21 September 2026 — Late chunking (method 5 of 5)

### 1. Phase/topic and objective
Contextualize longer source windows before pooling to produce embeddings for the fixed-size baseline's chunk spans ("late chunking").

### 2. What was implemented
| File | Responsibility |
|---|---|
| `chunking/late/chunker.py` | Window ownership, token alignment, contextual pooling |
| `chunking/late/cache.py` | Atomic context-aware cache, whole-window parity probe |
| `chunking/late/run.py` | Baseline validation, GPU execution, vectors, mappings, reports |
| `chunking/late/audit.py` | Independent re-reading/verification |
| `configs/late.json` | Baseline location, 5 owned chunks, 1 neighbor per side, input cap |
| `tests/test_late.py` | Ownership, gap, Unicode, pooling, cache-recovery checks |

Window policy: usually 5 consecutive chunks own a window, +1 neighbor chunk each side; windows stop at document boundaries, excluded pages, blank-page gaps; max 2,048 total tokens; planner reduces owned chunks then neighbors if needed, never cuts a target chunk.

Commands:
```bash
.venv/bin/python -m rag_pipeline chunk-late --run-id my_late_sample --sample
.venv/bin/python -m rag_pipeline chunk-late --run-id my_late_full
.venv/bin/python -m rag_pipeline chunk-late --run-id my_late_repeat --cache-only
.venv/bin/python -m rag_pipeline.chunking.late.audit --run-id my_late_full --repeat-id my_late_repeat
```

### 3. Exact numeric results
| Check | Result |
|---|---|
| Exact matched baseline chunks | **8,627**, each 512-dim normalized FP32 vector |
| Documents / eligible regions | 60 / 169 |
| Encoder windows | **1,804**; max **1,794** total tokens; no budget fallback needed |
| Planned total window input | **3,022,356 tokens**, ~1.373× independent fixed-chunk input |
| Independent/contextual token-count differences | 210 chunks |
| Whole-window early/late parity | max element difference **1.49e-8** (tolerance 1e-6) |
| Same-chunk window-placement comparisons | 3,270; mean cosine **0.97538**, min **0.88825** |
| Early/late vector comparisons | mean cosine **0.95412**, min **0.73017** |
| GPU peak allocated/reserved | ~597 / 848 MiB |
| Full process peak host memory | ~2.07 GiB |
| Final vector matrix | ~16.85 MiB |
| Full run elapsed / actual inference | ~149.5 / 113.6 seconds; includes 117 cached windows |
| Cached repeat | 1,804 hits, zero inference; ~29.2 seconds including validation |
| Tests | **105 pipeline + 17 cleaning = 122 passed** |

Artifacts: `phase8_late_full_20260921/`, `phase8_late_repeat_20260921/`, `phase8_late_full_20260921_audit.json`. Audit checked all 122 prepared files against Phase 1 and 6 paired early artifacts against preflight hashes.

### 4. Problems/bugs found
None reported as bugs — all checks (parity, matching, byte-identical repeat) passed.

### 5. Key decisions and rationale
- Phase 8 uses batch **one** for longer inputs (vs. Phase 7's batch 8); numerical compatibility between batch sizes documented in Phase 7 (<1.8e-7).
- Window-local token offsets determine pooling membership; tokens crossing a target boundary cause explicit failure.
- Phase 8 deliberately uses a new-run-ID recovery workflow instead of Phase 7's `--resume` command.
- Separate SQLite cache at `data/outputs/rag/late_window_cache/`; cache key includes complete window text and pooling spans (changed context ⇒ different key).

### 6. Limitations/caveats
- Cosine similarities are not accuracy percentages.
- `window_seams.jsonl` exposes context-window-placement sensitivity but does not remove it.
- GPU peaks exclude other applications/driver memory.
- Next: Phase 9 indexing and retrieval.

---

## 9. `rag_pipeline/docs/phase9_completion.md`

**Date:** Completed 27 September 2026 — Indexing and retrieval

### 1. Phase/topic and objective
Build five separate searchable FAISS indexes (one per chunking method) and run the 20-question development pilot through retrieval and context assembly.

### 2. What was implemented
| File | Role |
|---|---|
| `retrieval/inputs.py` | Validate five source sets and model compatibility |
| `retrieval/build_index.py` | Build, save, reload LangChain/FAISS stores |
| `retrieval/search.py` | Common exact ranking and tie rule |
| `retrieval/assemble_context.py` | Source-span dedup, budget, citations |
| `retrieval/diagnostics.py` | Pilot-only evidence inspection |
| `retrieval/run.py` | Query encoding, five-method execution, manifests |
| `retrieval/audit.py` | Saved-output checks, independent dot-product/evidence checks |
| `configs/retrieval.json` | Method inputs, pilot location, shared settings |
| `tests/test_retrieval.py` | Synthetic ranking, corruption, overlap, evidence tests |

Index: FAISS `IndexFlatIP` per method. Query encoder: same pinned Jina model, CUDA batch 1. Search scope: all 60 docs, no filters. Ranking: descending inner product; ties by ascending chunk ID. K = 1, 3, 5, 10. One FAISS thread. Context: top 10 candidates, 1,536 Jina content-token budget.

Commands:
```bash
uv pip install --python .venv/bin/python -r requirements-phase9.txt
.venv/bin/python -m rag_pipeline retrieve-pilot --run-id my_retrieval_pilot
.venv/bin/python -m rag_pipeline retrieve-pilot --run-id my_retrieval_repeat --cache-only
.venv/bin/python -m rag_pipeline.retrieval.audit --run-id my_retrieval_pilot --repeat-id my_retrieval_repeat
```

### 3. Exact numeric results
| Check | Result |
|---|---|
| Indexed vectors | **53,977** across 5 indexes; each searches all 60 docs |
| Per method | Fixed 8,627; sliding 11,419; structure 13,377; semantic 11,927; late 8,627 |
| Pilot execution | 20 shared query vectors, 100 searches, 100 assembled contexts |
| Saved evidence budget | ≤1,536 Jina content tokens (excl. citation wrappers) |
| Repeat | Byte-identical queries/indexes/rankings/contexts/diagnostics |
| Numerical audit | Max FAISS/float64 score difference **1.62e-7** (tolerance 2e-6) |
| Tests | **116 pipeline + 17 cleaning = 133 passed** |
| Inputs | 122 prepared files + 26 source/benchmark artifacts unchanged |
| Local resources | ~105.42 MiB total index storage; ~302 MiB peak reserved query GPU memory; ~2.21 GiB peak process memory |
| Timings | Initial run **72.2 s** (incl. **57.6 s** input validation); cached repeat **69.4 s** |
| Human review | **0 of 20** pilot questions approved |

Main output: `data/outputs/rag/phase9_retrieval_pilot_20260927/`. Repeat: `phase9_retrieval_repeat_20260927/`.

Examples cited:
- **pilot_001**: UCSC credit definition is first structure-aware result; first fixed-size result is unrelated-university rule.
- **pilot_004**: Fixed/sliding retrieval find reference at rank 10, but budgeted context omits it; structure-aware finds it at rank 2 and retains it.

### 4. Problems/bugs found
None reported; all checks passed.

### 5. Key decisions and rationale
- "Request all exact scores before sorting and selecting K" chosen to resolve ties even at the cutoff (manageable up to largest index of 13,377 chunks; acknowledged as not scalable for much larger corpora).
- Only `question_id` and question text/hash enter query records — reference answers/evidence/scope never enter search interface.
- Overlap-removal context assembly: repeated spans counted once; separate pieces keep separate source spans (no false concatenation claim).

### 6. Limitations/caveats
- No answer-generation model called in this phase.
- Pilot results are diagnostic, "must not become final research claims."
- Reviewed development/test questions and final quantitative evaluation remain Phase 11.

---

## 10. `rag_pipeline/docs/phase10_completion.md`

**Date:** Implemented and pilot-validated 27 September 2026 — Grounded answer generation

### 1. Phase/topic and objective
Generate answers from Phase 9's saved evidence using a free LLM (GroqCloud), applying the same instructions/limits across all five chunking methods.

### 2. What was implemented
| File | Purpose |
|---|---|
| `generation/answer.py` | LangChain messages, full-prompt counting, request identity, citation diagnostics |
| `generation/client.py` | Shared ChatGroq transport, sanitized metadata |
| `generation/state.py` | Persistent answer cache, quota ledger, retries, writer lock |
| `generation/run.py` | Input checks, prompt prep, pilot selection, resume, export |
| `generation/audit.py` | Prompt/source/token/response integrity, replay comparison |
| `generation/prompts/grounded_answer.txt` | Grounding/abstention instructions |
| `configs/generation.json` | Model, token limits, pilot selection, tokenizer pins |
| `tests/test_generation.py` | Offline request, error, quota, replay tests |

Shared generator settings:
| Setting | Value |
|---|---|
| Provider | GroqCloud via LangChain `ChatGroq` |
| Model | `qwen/qwen3.8-27b` |
| Reasoning mode | `none` |
| Temperature / top-p / seed | 1e-8 / 1.0 / 42 |
| Max answer | 512 generator tokens |
| Total budget | 8,192 tokens (incl. 256-token margin + answer reserve) |
| Evidence | ≤1,536 Jina content tokens |
| First live sample | pilot_001, pilot_004, pilot_019 × 5 methods |

Local rate limits (more conservative than Groq's published free tier of 30 req/min, 1,000/day, 8,000 tokens/min, 200,000/day): 24 requests/minute, 900/day, 7,000 tokens/minute, 180,000/day.

Tokenizer: `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

Commands:
```bash
uv pip install --python .venv/bin/python -r requirements-phase10.txt
.venv/bin/python -m rag_pipeline.download_generator_tokenizer
.venv/bin/python -m rag_pipeline generate-answers --run-id phase10_prepare_new --dry-run
.venv/bin/python -m rag_pipeline generate-answers --run-id phase10_pilot_new
.venv/bin/python -m rag_pipeline generate-answers --run-id phase10_replay_new --cache-only
.venv/bin/python -m rag_pipeline generate-answers --run-id phase10_all_new --all-pilot
.venv/bin/python -m rag_pipeline generate-answers --run-id phase10_all_new --all-pilot --resume
```

### 3. Exact numeric results
| Check | Result |
|---|---|
| Prepared requests | **100**: 20 pilot questions × 5 methods |
| Live pilot | **15/15 completed**: 3 questions × 5 methods; no API errors/retries |
| Remaining live requests | 85 not executed |
| Actual usage | **35,888 input + 1,266 output = 37,154 tokens** |
| Full prepared input | **242,365 estimated input tokens**; 319,165 incl. output reservations/margins |
| Largest prepared prompt | **3,020 Qwen tokens**; all 100 fit without truncation |
| Prompt-count comparison | Provider/local counts match exactly for all 15 live requests |
| Answer limits | No answer reached 512-token output limit |
| Cached repeat | 15/15 cache hits, zero new API calls |
| Tests | **129 pipeline + 17 cleaning = 146 passed** |
| Inputs | 122 prepared files + full Phase 9 inventory unchanged |
| Timings | **428.5 seconds** recorded live run (mostly quota pacing); **13.46 seconds** summed client latency |
| Human review | Pilot annotations remain drafts |

Artifacts: `phase10_generation_pilot_20260927/`, `phase10_generation_repeat_20260927/`, `phase10_prepare_20260927/` (no-call snapshot, superseded by main 100-request run).

Example observations:
- **Structure-aware/pilot_001**: Answer cited 15 lecture hours or 30 practical hours, or equivalent combination.
- **Fixed/pilot_004**: Phase 9 found gold passage at rank 10 but omitted it from context; generation abstained. Structure-aware retained passage and answered: 40% continuous assessment, 80% attendance.
- **Late/pilot_019**: Model correctly declined to invent a personal mark.
- **Fixed/pilot_001**: Abstention included unsupported university-name expansions.
- Count: 4 responses gave direct answers, 11 began with abstention (of the 15 live).

### 4. Problems/bugs found
- Earlier proposed Llama model was not in the account's model list on 27 September 2026 — Qwen used instead (not exactly a bug, but a provider-availability issue resolved by model substitution).
- Groq returned **three different system fingerprints** during the pilot (preserved per response; cause unknown).
- The first citation diagnostic recognizes individual `[S1]` labels only; grouped forms like `[S1, S2]` are not fully parsed — flagged for Phase 11 to fix.

### 5. Key decisions and rationale
- Groq free tier chosen deliberately conservative to avoid exceeding published limits; "intentionally conservative scheduling can leave some free quota unused."
- Shared persistent quota ledger reserves estimated prompt + max answer + margin before every attempt, including failed attempts.
- Resume requires unchanged inputs, code, settings, selection, trial ID.

### 6. Limitations/caveats
- Groq does not expose billing-tier verification; free-tier setting is a policy record, not a technical guarantee.
- Seed/low temperature do not guarantee identical fresh answers.
- 1,536-token packing limit still measured with Jina tokenizer, not Qwen's — needs reconciling before benchmark scoring.
- Next: Phase 11 — controlled evaluation (pending team benchmark/protocol freeze).

---

## 11. `rag_pipeline/docs/phase11_status.md`

**Dates:** "In progress" status note; new benchmark preparation dated 6 October 2026; earlier phase content dated throughout.

### 1. Phase/topic and objective
Phase 11: provisional evaluation implementation. Status: "in progress." Offline evaluation code works; final benchmark evaluation and thesis findings still pending.

### 2. What was implemented
Evaluation code modules:
| Module | Responsibility |
|---|---|
| `evaluation/inputs.py` | Verify checksums, questions, spans, context assembly, prompt lineage |
| `evaluation/retrieval_metrics.py` | Evidence-unit coverage, ranking metrics |
| `evaluation/answer_metrics.py` | Validate human judgments, parse citation labels |
| `evaluation/compare_runs.py` | Aggregate coverage, paired grouped bootstrap differences |
| `evaluation/ragas_adapter.py` | Optional explicit-client RAGAS interface (stub-tested, not live-validated) |
| `evaluation/run.py` | Run offline checks, write reviewable outputs |
| `evaluation/audit.py` | Independently recompute metrics, verify repeat outputs |

Commands:
```bash
.venv/bin/python -m rag_pipeline evaluate --run-id phase11_check_new --provisional
.venv/bin/python -m rag_pipeline evaluate --run-id phase11_check_repeat --provisional
.venv/bin/python -m rag_pipeline.evaluation.audit --run-id phase11_check_new --repeat-id phase11_check_repeat
.venv/bin/python -m rag_pipeline evaluate --run-id phase11_reviewed_new --provisional --answer-reviews data/benchmark/pilot_answer_reviews.jsonl
```
Dependencies in `requirements-phase11.txt` (extends Phase 10 env); RAGAS 0.2.6 optional.

Metrics implemented: Evidence Recall@K (K=1,3,5,10); Complete evidence@K; MRR@K; Context recall/complete evidence; Alternative evidence; Answer correctness; Faithfulness/citation support; Relevance (0/1/2); Unanswerable behavior.

Paired bootstrap: 2,000 samples, seed 42, 95% percentile intervals, resampling connected source/question families.

### 3. Exact numeric results
**New benchmark preparation (6 Oct 2026):**
- 100-question AI-authored benchmark: **138 evidence units** from **10 PDFs**; **50 physical PDF pages** visually checked by the assistant.
- New run: **500 retrieval evaluations**, **500 generation prompts prepared**, zero generated answers, no API calls.
- Retrieval and prompt/source audits passed; **32 focused tests passed**.

**Earlier 20-question pilot run (described in same doc):**
- Scope: 20 draft pilot questions (18 answerable, 1 unanswerable, 1 ambiguous), not the 153 imported team questions.
- Five methods × 20 questions = **100 records**.
- Only **15 answers** generated in Phase 10; **85 remain unexecuted**.
- Main output: `data/outputs/rag/phase11_provisional_20261006_v2/`; repeat: `..._v2_repeat`; earlier dev run `phase11_provisional_20261006/` preserved.

Team-handoff correction queue: **73 evidence-confirmation cases**, **25 answerability decisions**, **29 wording-only cases**; **26 other annotations** with no automated intake flag.

Tests: **All 154 pipeline tests passed**, including **19 new evaluation tests**.

Audit: independently recomputed **all 90 answerable question–method records** using non-whitespace character positions; matched every saved retrieval metric; confirmed identical repeat artifacts (excl. environment metadata); checked **122 unchanged prepared files**. Saved as `phase11_provisional_20261006_v2_audit.json`.

### 4. Problems/bugs found
None reported as code bugs; the main gaps are process/review gaps (missing human review, missing benchmark freeze).

### 5. Key decisions and rationale
- User requested assistant-led authoring of a new 100-question benchmark instead of waiting for team correction queue.
- Evaluation command requires explicit `--provisional` flag; omitting it fails before creating output — a deliberate guard against accidental "final" results.
- JSON `null` / blank CSV always means "not scored," never automatic zero.
- Missing retrieval/context for answerable questions is scored as a conservative zero.
- Ambiguous and extraction-diagnostic questions excluded from primary retrieval scores.

### 6. Limitations/caveats
- No model weights loaded, no API contacted in this provisional run.
- Judge status for RAGAS: `not_run`.
- Six remaining prerequisites listed before final scoring: final benchmark selection/review; protected dev/test splits; frozen method settings; comparable retrieval/generation across benchmark; human answer review + validated judge workflow; final scored outputs with uncertainty reporting, followed by Phases 12–13.

---

## 12. `rag_pipeline/docs/decisions.md`

**Dates:** Spans 19 September 2026 through 6 October 2026 (chronological decision log across all phases).

### 1. Phase/topic and objective
A running log of implementation decisions by phase, each with rationale/consequence.

### 2. What was implemented
Summarizes implementation choices already covered in the completion docs; no new code beyond cross-referencing (e.g., `PreparedCorpusLoader`, `JinaEmbeddings`, `ChatGroq` client factory added "pre-Phase 3").

### 3. Exact numeric results (decision-log specific numbers)
**Phase 1 corpus boundaries:**
- 60 documents; 6,215 physical page records.
- 44 explicit exclusions.
- 172 included pages with empty/whitespace-only text.
- 169 continuous eligible regions.

**Phase 2 follow-ups (20 Sept 2026):**
- `PreparedCorpusLoader` full run: 169 eligible regions from 60 sources, unchanged text.
- CUDA probe: peak allocation ~667 MiB.
- Near-zero temperature `1e-8` recorded for Groq integration.

**Phase 3 (20 Sept 2026):** 8,627 chunks; 122 unchanged prepared-input hashes.

**Phase 4 (20 Sept 2026):** 11,419 chunks; 11,250 adjacent pairs sharing exactly 64 source tokens; zero overlap reductions.

**Phase 5 (21 Sept 2026):** 13,377 chunks; mean 163.28 content tokens; "95 tests pass" (differs slightly from completion doc's "78 pipeline + 17 cleaning").

**Phase 6 — table of boundary decisions** (region-local 90th percentile, 64-token minimum, 256-token hard cap, no overlap, SQLite caching, FP32/seed 42/deterministic Torch).

**Phase 7 (21 Sept 2026):** batch size 8; largest element difference across 2,845 sample vectors below 1.8e-7.

**Phase 8 (21 Sept 2026):** reuse Phase 7 fixed-size chunks exactly; 5 chunks per window + 1 neighbor each side; 2,048-token cap.

**Phase 9 (27 Sept 2026):** FAISS IndexFlatIP per method; K=1,3,5,10; 1,536-token budget.

**Phase 10 (27 Sept 2026):** Groq `qwen/qwen3.8-27b`; `reasoning_effort=none`; 512 output tokens, 256-token margin, 8,192-token total; 100 prompts prepared; first sample pilot_001/004/019 = 15 requests.

**Streamlit explorer (28 Sept 2026):** added inspection UI; artifacts under `data/outputs/rag/ui_verification_20260928/`.

**Phase 11 provisional (6 Oct 2026):** offline command requires `--provisional`; preserves separate 153-question intake as drafts.

### 4. Problems/bugs found
None distinctly new; references the Phase 5 "earlier full directory retains the failed tokenless-glyph diagnosis" (same issue noted in Phase 5 completion doc).

### 5. Key decisions and rationale (notable entries not already captured above)
- JSONL prepared text chosen as input contract because "it matches TXT and carries physical page provenance."
- Both Git revision and source hashes recorded, since "experimental code may be uncommitted; a commit ID alone would not identify the code actually run."
- Vithusan selected this computer only (no Colab/GPU assumption); free LLMs chosen (zero paid budget) — **as of this doc's Phase 2 section**, later superseded (see `experiment_protocol.md` 7 Oct 2026 update).
- GPU use allowed: `lspci` identifies RTX 3050 Mobile; `nvidia-smi` reports 4,096 MiB and driver **595.91.07**.
- Code style decision: "clear names, short functions, useful comments and teaching examples... Code remains AI-assisted; readability and team understanding are the standard."

### 6. Limitations/caveats
- "Decisions still pending" (as of Phase 2 section): confirmation of development retrieval quality and final chunk configuration; reviewed benchmark composition; lead review of draft metrics; exact free generator/judge models.
- Phase 6: "The 90th-percentile choice remains untuned."
- Phase 11: exact source evidence defines retrieval relevance; human rubric judgments define answer correctness; citation label existence alone does not establish support.

---

## 13. `rag_pipeline/docs/model_feasibility.md`

**Date:** 20 September 2026 — Local embedding feasibility

### 1. Phase/topic and objective
Decide and validate the shared embedding model for the five-method study.

### 2. What was implemented
- GPU/CPU capability probes for Jina embeddings.
- Pooling: content-token mean only (excludes special tokens/padding), then L2 normalization — a deliberate deviation from stock pooling.
- ALiBi tensor allocation override to 2,048 positions; explicit rejection of longer inputs before inference.
- Pinned custom code files inspected (imports/loading/pooling/ALiBi paths), hashes/revisions pinned.
- Reproduce commands:
```bash
uv pip install --python .venv/bin/python -r requirements-phase2.txt
.venv/bin/python -m rag_pipeline.embedding.download_probe --bundle data/outputs/rag/model_probe_20260920
.venv/bin/python -m rag_pipeline.embedding.probe --bundle data/outputs/rag/model_probe_20260920 --output data/outputs/rag/phase2_probe_repeat --device cuda
```

### 3. Exact numeric results
| Item | Value |
|---|---|
| Model | `jinaai/jina-embeddings-v2-small-en` |
| Model/tokenizer revision | `44e7d1d6caec8c883c2d4b207588504d519788d0` |
| Custom implementation | `jinaai/jina-bert-implementation` |
| Implementation revision | `f3ec4cf7de7e561007f27c9efc7148b0bd713f81` |
| Weights | Safetensors, **65,388,544 bytes**; SHA256 in `embedding/probe_bundle.json` |
| Runtime | Python 3.11.16, torch 2.5.1, transformers 4.57.6; CPU FP32, 4 threads |
| Vector dimension | 512 |
| Tested maximum | 2,048 total tokens = 2,046 content + CLS/SEP |

CPU timing measurements (`data/outputs/rag/phase2_probe_20260920/probe_result.json`):
| Total input tokens | Forward + content pooling seconds |
|---|---:|
| 130 | 0.048 |
| 258 | 0.040 |
| 514 | 0.094 |
| 1,026 | 0.305 |
| 2,048 | 1.095 |

Other measurements:
- Model loading took **7.16 seconds**.
- Process peak RSS: **~1,402 MiB**.
- Exact-window early/late pooling max absolute difference: **0.0**.
- Contextual vs. independent sentence encoding cosine: **0.9449**.
- A **2,049-token input was rejected before inference** (no truncation).
- GPU follow-up: RTX 3050 Laptop GPU, 4,096 MiB VRAM; LangChain CUDA probe at 2,048 tokens: ~667 MiB peak allocated, 798 MiB reserved, batch size 1, FP32.
- LangChain CUDA single-op timing (after warmup): **0.089 seconds**.
- Padded two-text batches matched independent calls within **4.5e-8** max absolute difference.
- Full-window early/late pooling differed by less than **1.5e-8**.

### 4. Problems/bugs found
None — all probes passed (no missing/unexpected/mismatched/error loading keys; finite normalized vectors; overflow correctly rejected).

### 5. Key decisions and rationale
- Jina v2 small English selected by Vithusan for the five-method study.
- Content-only mean pooling chosen deliberately to differ from stock `model.encode()`, applied consistently to queries, early, and late paths.
- ALiBi buffer explicitly capped at 2,048 to prevent dynamic rebuild of a larger buffer.
- No projection applied — uses `last_hidden_state`, not BERT's pooler output.

### 6. Limitations/caveats
- CPU timings are single forwards on synthetic inputs, not averaged benchmarks or full-corpus estimates.
- Advertised 8,192-token limit, production-scale batching, window seams, full-corpus speed, and retrieval quality remain untested (deferred to Phase 7/8).
- Not a comprehensive security audit of the custom code.
- GroqCloud free tier selected for generation with zero paid budget (model TBD at this stage) — later changed per `experiment_protocol.md`.

---

## 14. `rag_pipeline/docs/langchain_guide.md`

### 1. Phase/topic and objective
Short team guide explaining the LangChain interfaces/classes added across phases and how to use them.

### 2. What was implemented
- `rag_pipeline/configs/models.json` holds model settings.
- `PreparedCorpusLoader` (LangChain `BaseLoader`) — returns **169 continuous text regions from 60 PDFs**.
- `JinaEmbeddings` (LangChain `Embeddings`) with `embed_documents()`, `embed_query()`, `encode_tokens()` helper.
- Per-phase splitter classes: `FixedSizeSplitter` (Phase 3), `SlidingWindowSplitter` (Phase 4), `StructureAwareSplitter` (Phase 5), `SemanticSplitter` (Phase 6).
- `ChunkEmbeddings` with tested batch size of 8 (Phase 7).
- Late chunking reuses `JinaEmbeddings.encode_tokens()` (Phase 8).
- Retrieval uses LangChain `FAISS`, `Document`, `InMemoryDocstore` (Phase 9).
- Generation uses `ChatPromptTemplate` and `ChatGroq` (Phase 10); `GroqAnswerClient` wraps `ChatGroq`.

Commands:
```bash
.venv/bin/python -m rag_pipeline.examples.inspect_langchain
.venv/bin/python -m rag_pipeline inspect-langchain --run-id my_langchain_review
.venv/bin/python -m rag_pipeline generate-answers --run-id answers_prepare_new --dry-run
.venv/bin/python -m rag_pipeline generate-answers --run-id answers_pilot_new
uv pip install --python .venv/bin/python -r requirements-phase2.txt
.venv/bin/python -m rag_pipeline.embedding.download_probe --bundle data/outputs/rag/model_probe_20260920
.venv/bin/python -m unittest discover -s rag_pipeline/tests -v
```

### 3. Exact numeric results
- 169 regions from 60 PDFs (`PreparedCorpusLoader`).
- CUDA probe: ~667 MiB peak PyTorch allocation / 798 MiB reservation at 2,048 tokens.
- Phase 10: first command prepares **100 requests**; second command selects **3 questions × 5 methods = 15 answers**.
- Pinned packages: `langchain-core==0.3.63`, `langchain-groq==0.2.3`, `groq==0.13.1`, `langchain-text-splitters==0.3.8`.

### 4. Problems/bugs found
None; this is a reference guide.

### 5. Key decisions and rationale
- `load_and_split()` deliberately left unavailable: a generic implementation would copy region offsets onto smaller chunks without updating them.
- `lazy_load()` yields regions only after full corpus validation (not constant-memory).
- Device/precision policy must be recorded consistently per run; CPU vs GPU timings should not be compared as if chunking alone caused differences.
- `GROQ_API_KEY` must be set via environment/`.env`, never embedded in code.

### 6. Limitations/caveats
- RTX 3050 "works outside the restricted agent sandbox" — GPU commands must run in a normal GPU-enabled terminal.
- Free-tier configuration records a policy choice; it cannot verify an account's billing tier.
- Low temperature/fixed seed do not guarantee identical fresh hosted responses.

---

## 15. `rag_pipeline/docs/annotation_guide.md`

### 1. Phase/topic and objective
Instructions for team members (Aysha, Haleema, Vithusan) to annotate/review the benchmark.

### 2. What was implemented
- `data/benchmark/pilot_review.md` — 20 drafts with answers/evidence, links to 13 rendered physical PDF pages from 7 PDFs.
- Record schema (version 1): question identity, corpus hash, split, language, category, answerability, question, scope, reference answer, rubric, `group_id`, author, `evidence_units`, rationale (if non-answerable), `review`.
- Each evidence unit: `unit_id`, claim, list of alternative source spans (document/source identity, original PDF SHA256, prepared-text SHA256, document-level character start/end, exact text, one-based physical `page_numbers`).
- `annotation_hash` function in `rag_pipeline.evaluation.validate_benchmark`.
- Commands:
```bash
.venv/bin/python -m rag_pipeline.evaluation.validate_benchmark
.venv/bin/python -m rag_pipeline.evaluation.validate_benchmark --require-ready
.venv/bin/python -m rag_pipeline.evaluation.validate_benchmark --freeze
```

### 3. Exact numeric results
- 20 drafts: 18 answerable, 1 unanswerable, 1 ambiguous.
- 18 answerable drafts reference six questions from each member's documents.
- 13 rendered physical PDF pages from 7 PDFs.
- Development and test currently have **zero records**.
- Example: UCSC PDF physical page 71 = printed page 57.

### 4. Problems/bugs found
Not applicable (process guide, not implementation report).

### 5. Key decisions and rationale
- Approved review requires: `status: "approved"`, `reviewer` (actual independent teammate name), `reviewer_kind: "human"`, `pdf_verified: true`, `reviewed_content_hash` (SHA256 via `annotation_hash`), `notes`.
- Approval field itself does not enter the content hash, but wording/evidence/rubric/group/split do.
- Review by a second team member required when a teammate authored/revised the question.
- All versions/near-duplicates of a source family must stay on one side of dev/test split.
- Reviewed pilot questions may become development questions; they never become test questions.
- `--freeze` creates `frozen_snapshot.json` (corpus, manifest, question-file, source PDF, protocol-document hashes) — exclusive-create, irreversible.

### 6. Limitations/caveats
- "None has human approval yet."
- The validator checks recorded review metadata, not the truth of someone's claimed review or semantic sufficiency of evidence.
- `--freeze` does not make local files physically read-only.

---

## 16. `rag_pipeline/docs/assistant_benchmark.md`

**Date:** 6 October 2026

### 1. Phase/topic and objective
Document the assistant-authored 100-question benchmark created at Vithusan's request to allow provisional experiments without waiting for manual team authoring.

### 2. What was implemented
- 100-question answerable set at `data/benchmark/assistant_100_20261006/`: 60 handbook questions + 40 lecture questions, 10 questions from each of 10 source PDFs.
- Files: `questions.csv`, `question_book.md`, `README.md` (scope/selection notes), `validation_report.json`.
- `build_records.py` preserved for rebuild/corrections.
- Local `retrieval_config.json` and `generation_config.json` pointing to new benchmark.

### 3. Exact numeric results
- 138 evidence units with **144 exact-span alternatives** total.
- 50 original PDF pages visually checked by the assistant.
- All 100 records pass schema, source-span, page, source-hash validation.
- 122 prepared corpus files unchanged.
- Completed provisional run: all **100 questions × 60 documents × 5 methods** → **500 retrieval/context records**.
- Generator dry run: **500 requests prepared, 0 answers**, no API calls.
- Prompt preparation estimate: **1,199,701 input tokens** for 500 requests; with answer allowance + margin, reserves **1,583,701 tokens**.
- Generation/evaluation code change: **32 focused tests passed**.
- Repeated offline evaluation: independent character-position audit recomputed all **500 question–method records**, matched every retrieval metric; repeat artifacts identical except environment metadata.

Artifacts:
- Retrieval: `data/outputs/rag/phase11_ai100_retrieval_20261006_gpu/` (earlier non-GPU attempt preserved as failed run at `phase11_ai100_retrieval_20261006`).
- Prompts: `data/outputs/rag/phase11_ai100_prompts_20261006/`.
- Report: `data/outputs/rag/phase11_ai100_provisional_20261006/report.md`, `metrics_summary.csv`.
- Audit: `data/outputs/rag/phase11_ai100_provisional_20261006_audit.json`.

### 4. Problems/bugs found
- An earlier retrieval attempt failed due to lack of GPU access (preserved as `phase11_ai100_retrieval_20261006`); fixed by rerunning with GPU access (suffix `_gpu`).

### 5. Key decisions and rationale
- Author recorded explicitly as **Codex (AI assistant)**; assistant checks recorded separately from pending human review ("not an independent review of the labels" since the same assistant authored and checked).
- All questions assigned to `pilot` split (no held-out test, no unanswerable questions in this set).
- Equal 10-per-document allocation explicitly called "a practical sampling choice, not a statistical-power justification."
- Dry-run approach chosen to make missingness auditable without silently generating/reusing answers.

### 6. Limitations/caveats
- "No human approvals are inferred."
- "Do not call these unseen test results or use them to assess abstention performance" (overlap with earlier pilots/team drafts).
- "No answer correctness, faithfulness or abstention conclusion is available yet."
- Final thesis reporting needs explicit account of AI authorship, independent validation performed, sampling limitations, and whether independent held-out evaluation was completed.

---

## 17. `rag_pipeline/docs/question_sheet_import.md`

**Date:** 6 October 2026 — Importing the team question sheet

### 1. Phase/topic and objective
Document the CSV importer that creates a draft intake workspace from the team's question sheet, without approving questions or modifying the benchmark manifest.

### 2. What was implemented
- Importer command and modules producing: `normalized_drafts.csv`, `retained_questions.csv`, `excluded_questions.csv`, `unresolved_questions.csv`, `draft_records.jsonl`, `decisions.json`, `import_summary.json`.
- Evidence matching rules (search supporting passage; interpret "same" as reference answer; permit whitespace/quote normalization; reject very short answers <20 chars or <4 words as sole evidence; reject spans crossing excluded/blank gaps).
- `evaluation/import_question_sheet.py` and `evaluation/prepare_correction_sheet.py`.
- Commands:
```bash
.venv/bin/python -m rag_pipeline.evaluation.import_question_sheet \
  --input 'QA_final_2026_10_03 - Sheet1.csv' \
  --decisions data/benchmark/import_decisions_20261006.json \
  --output data/benchmark/imports/team_153_repeat
.venv/bin/python -m unittest rag_pipeline.tests.test_question_sheet_import rag_pipeline.tests.test_benchmark -v
.venv/bin/python -m rag_pipeline.evaluation.prepare_correction_sheet --output data/benchmark/reviews/team_corrections_new
```

### 3. Exact numeric results
- Original CSV: **160 rows**, unchanged.
- Excluded: 4 named problem records + 1 copy of each of 3 duplicate pairs = **7 total excluded**:
  | ID | Reason |
  |---|---|
  | HALEEMA_033 | Medical-certificate question with unrelated credit evidence |
  | HALEEMA_143 | Duration question answered with study mode |
  | HALEEMA_146 | Role question answered with a person's name |
  | AYSHA_071 | Physical page 220 in a 41-page source |
  | HALEEMA_050 | Duplicate of retained HALEEMA_034 |
  | HALEEMA_051 | Duplicate of retained HALEEMA_038 |
  | HALEEMA_052 | Duplicate of retained HALEEMA_036 |
- **153 retained drafts**.
- **56 records** with aligned candidate evidence (55 answerable, 1 unresolved answerability label).
- **12 blank-evidence records** recovered via reference answer.
- **24 page references corrected**.
- **90 source filename cells resolved**.
- **26 records** without intake flags; **127 records** have ≥1 unresolved check (68 underspecified scope, 25 need unanswerable/ambiguous label, 73 answerable records lack full matching passage — these overlap categories).
- **8 retained "same" records** did not yield a sufficiently long full source match.
- Correction export: Aysha **77** correction rows, Haleema **50**, another **26** rows in review-only file (all 153 IDs preserved).
- **17 tests** cover Unicode offsets, quote/whitespace normalization, repeated text, excluded gaps, short answers, unresolved labels, independent-review gates, benchmark protections.
- All **41 resolved original source PDFs** hashed and compared to loaded corpus.

Output: `data/benchmark/imports/team_153_20261006/`.

### 4. Problems/bugs found
Not bugs per se, but data-quality issues identified in the sheet (excluded records above) — resolved by explicit exclusion decisions authorized by Vithusan, not by code fixes.

### 5. Key decisions and rationale
- `draft_records.jsonl` deliberately uses `record_type: benchmark_intake_draft`, `intake_schema_version: 1` — distinct from final benchmark schema so it cannot be fed to the scored validator by mistake.
- "Pending" status recorded (not invented reviewer names) since independent reviewer names/verification not supplied.
- Decisions file pins the CSV checksum; revised sheet requires revisiting exclusion decisions.

### 6. Limitations/caveats
- Matching evidence does not establish correctness.
- The 8 unmatched "same" records' answers are not proven false — paraphrase/formatting may simply prevent exact matching.
- `development_only` flag recorded for exposed source families; `false` does not mean eligible for held-out testing.
- Correction files need a correction-aware import and fresh evidence validation before use.

---

## 18. `rag_pipeline/docs/experiment_protocol.md`

**Dates:** Prepared 20 September 2026; updated 7 October 2026 (budget/provider change); Phase 9 implementation record 27 September 2026.

### 1. Phase/topic and objective
Phase 2 draft experiment protocol defining research scope, resources, embedding/generation decisions, chunking grid, benchmark design, retrieval/context rules, outcome metrics, and generation plan for comparing all five chunking methods. Status at top: "needs team review; no held-out benchmark has been frozen" (though later sections show it was substantially completed/updated by 7 Oct 2026).

### 2. What was implemented
- Protocol document itself (no code), plus references to `ChatOpenAI`-based `OpenAIAnswerClient` replacing originally planned `ChatGroq` factory for generation.
- Phase 9 implementation record confirms exact-search/context rules implemented for the 20 pilot drafts.

### 3. Exact numeric results
**Corpus identity hash:** `dc3eeded7225fc31c96a3d27a2231b241783a377d98cca3443dc3eb0ec7260cf`.

**Resources (original, 20 Sept):** This computer: 16 logical CPUs, ~14.93 GiB RAM, NVIDIA RTX 3050 Laptop GPU, 4,096 MiB VRAM.

**Budget update (7 October 2026) — major change:**
- Groq's free tier could not be upgraded to paid; Vithusan switched generation (and RAGAS judge) provider to **OpenAI's paid pay-as-you-go API**.
- Generation model: **`gpt-4o-mini`**.
- Full **600-request** benchmark run (120 questions × 5 chunking methods) cost approximately **$0.22**.
- RAGAS judge pass over **500 answerable samples** added "a comparable small per-token cost."
- RAGAS judge: `gpt-4o-mini` (LLM) + `text-embedding-3-small` (embeddings), both OpenAI paid, deliberately separate from the Jina retrieval embedding.
- Embeddings remain local (Jina v2-small, CPU/CUDA FP32) — not part of paid budget; the controlled retrieval variable.
- *(Superseded)* Original plan: GroqCloud free tier; Groq retires Llama 3.1 8B and Llama 3.3 70B for free/developer tiers from **16 August 2026**.
- Proposed pilot cap (original): 30 minutes per method, 8 GiB process-RAM ceiling.

**Development grid (proposed):**
| Method | Proposed development variants | Shared rules |
|---|---|---|
| Fixed | 128, 256, 512 tokens | No overlap; retain tail |
| Sliding | Same 3 caps; 32/64/128-token overlap (25%) | Zero-overlap equivalence as correctness check |
| Structure | Same 3 caps | Section/paragraph → sentence → token fallback |
| Semantic | Same 3 caps | 90th percentile cosine cut; min target cap/4, hard max cap |
| Late | Paired with each fixed variant | Same exact spans/model/pooling |

**Benchmark design — realized split (7 October 2026):**
- **40 development (30 answerable / 10 unanswerable)** and **78 test (70 answerable / 8 unanswerable)**, plus **2 ambiguous → `diagnostic`** split (excluded from scoring). Total 120 questions.
- Test-side answerable category counts: definition 11, eligibility 15, exception 9, explanation 14, multi_passage 5, procedure 13, table_lookup 3.
- `table_lookup` has only **3** answerable questions in the entire 120-question benchmark (all in test); the ">=6 per category" target was unachievable for this category.
- `multi_passage`: 5 in test, 2 in development — falls one short of the six-question target.
- Benchmark covers **17 real document groups** + 1 synthetic `out_of_corpus_probe` group.
- All 120 rows carry independent-reviewer `review.status == "approved"`, confirmed **7 October 2026**.
- `validate_benchmark --directory ...` reports `"freeze_ready": true` with **zero blockers**.
- Team self-approved `protocol_status: "approved"` since "the original external research-lead reviewer was unavailable."
- `--freeze` deliberately **not yet executed** (irreversible, requires separate explicit go-ahead).

**Phase 9 implementation record (27 Sept 2026):**
- Numerical audit uses 2e-6 absolute score tolerance against float64 dot products; observed max difference **1.62e-7**.

**Generation (updated 7 Oct 2026):**
- Generator: OpenAI `gpt-4o-mini`, temperature 0.
- Full run: **600 requests (120 × 5), 600/600 successful, 0 failures**.
- RAGAS run over **100 answerable questions × 5 methods = 500 samples**.
- Preregistered: 3 repeats for a balanced development-selected subset (if runtime permits).

### 4. Problems/bugs found
Not a bug but a provider constraint: "Groq's free tier could not be upgraded to a paid billing plan when the team attempted to do so" — resolved by switching to OpenAI paid API.

### 5. Key decisions and rationale
- Primary research scope: compare fixed-size, sliding-window, structure-aware, semantic, and late chunking on same 60-doc corpus; measure retrieval, grounded answers, failure causes, resource cost.
- Content-only mean pooling documented as adaptation, applied consistently across queries/early/late.
- CUDA FP32, batch size 1 configured as default device/precision policy; explicit CPU runs possible.
- `ChatOpenAI`-based `OpenAIAnswerClient` replaces originally planned `ChatGroq` factory for generation due to the budget/provider switch.
- Development grid tuning rule: evaluate ≤3 primary variants per method on development questions; choose by primary evidence coverage; tie-break by lower indexed-token count then lower runtime; for late, use fixed method's selected size even if another late size scores better; per-method independently tuned comparisons reported only as secondary.
- Evidence coverage rule: all non-whitespace characters of at least one alternative span must appear in union of selected spans from same document/version; whitespace-only gaps ignored.
- Context assembly: top 10 in rank order, subtract already-included spans, stop at 1,536-token budget; skip-and-try-next for oversized candidates; log every skip.
- Outcome denominators: unanswerable/ambiguous/diagnostic questions excluded from evidence recall/MRR; abstention on answerable question = incorrect; empty retrieval scores zero; pipeline crashes/timeouts = unsuccessful primary trials (both conservative all-question and completed-run coverage reported).
- Uncertainty: 2,000 bootstrap resamples of source/question families, seed 42, 95% percentile intervals.
- Failure-cause taxonomy (PLAN taxonomy): extraction, boundary fragmentation, missing reference context, table/list separation, overflow, retrieval miss, wrong source/version, redundant overlap, context loss, generation error, benchmark/evaluator error — multiple contributing labels and explicit "uncertain primary cause" allowed.

### 6. Limitations/caveats
- "No results have been used to choose these settings" (protocol predates results).
- "Do not claim a statistical power guarantee for 80 test questions."
- Percentile grid of 80/90/95 for semantic threshold is "an optional sensitivity study, excluded from the primary tuning budget unless all methods receive a documented comparable search budget."
- Related documents may remain retrieval distractors; grouping prevents question-label leakage, not document removal.
- `table_lookup` and `multi_passage` categories explicitly treated as "sparse/exploratory... not as a quota failure."
- Deferred checklist items at bottom: five items marked `[x]` (done) and one `[ ]` (not done): "Before final test scoring: freeze benchmark; update the plan with actual completion evidence" — deliberately not yet done because `--freeze` is irreversible and needs separate go-ahead.
- RAGAS supplements but does **not** replace human review, which remains the primary reference for answer quality.