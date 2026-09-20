# Impact of Document Chunking Strategies on RAG in Educational Systems

**Implementation roadmap:** See [PLAN.md](PLAN.md) for the phase-by-phase RAG pipeline plan, folder structure, completion checks, and progress tracker.

**Phase 1 implemented:** See [rag_pipeline/README.md](rag_pipeline/README.md) to validate the prepared corpus and inspect source references before chunking.

**Phase 3 implemented:** Fixed-size chunking uses LangChain and the pinned Jina tokenizer: 8,627 chunks from all 60 documents, 256 content tokens at most, zero overlap. See the [Phase 3 handover](rag_pipeline/docs/phase3_completion.md). Use the 20 pilot drafts while building; the reviewed benchmark is due before final evaluation. Groq free tier is selected for generation.

> **Final Year Research — Group 04**  
> Department of ICT, Faculty of Technological Studies, University of Vavuniya

---

## 👥 Team

| ID | Name | Role |
|---|---|---|
| 2020ICTS04 | J.F. Aysha | Investigator |
| 2020ICTS68 | U.L. Haleema | Investigator |
| 2020ICTS89 | V. Vithusan | Investigator |

**Supervisor:** Ms. W.A.S.C Perera, Lecturer — Department of ICT

---

## 📌 Research Overview

This study investigates how different **document chunking strategies** affect the performance and reliability of **Retrieval-Augmented Generation (RAG)** systems, with a specific focus on educational content from Sri Lankan universities.

### The Core Problem

In a RAG system, documents must be split into smaller pieces (chunks) before being stored in a vector database. How you split those documents — the chunking strategy — directly determines:
- Whether the right information gets retrieved for a given question
- Whether the generated answer is accurate and faithful to the source
- What types of failures occur when chunking is done poorly

### Research Objectives

1. Analyze how different chunking strategies affect RAG system performance
2. Compare chunking methods in terms of retrieval accuracy
3. Evaluate how chunking influences correctness and reliability of answers
4. Identify failure modes caused by inappropriate chunk segmentation
5. Provide guidelines for choosing effective chunking strategies in educational RAG systems

---

## 🔬 Chunking Strategies Compared

| # | Strategy | Description |
|---|---|---|
| 1 | **Fixed-Size** | Split every N tokens, regardless of content |
| 2 | **Sliding Window** | Fixed-size with overlapping content between chunks |
| 3 | **Structure-Aware** | Split at document headings (H1, H2, H3) |
| 4 | **Semantic** | Split at topic boundaries using embedding similarity |
| 5 | **Late Chunking** | Encode longer text, then pool representations for matched chunk spans |

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| RAG Framework | LangChain |
| LLM (Generator) | GroqCloud free tier via LangChain ChatGroq; exact model pending |
| Embeddings | Pinned Jina v2 small English via LangChain Embeddings; CPU/CUDA tested |
| Vector Database | FAISS (Facebook AI Similarity Search) |
| Evaluation | Source-evidence retrieval metrics and human rubrics; optional free/local RAGAS evaluation |
| Language | Python 3.11 |

---

## 📁 Project Structure

```
educational-rag-research/
│
├── 01_Load_pdfs/                  # Step 1: Load and explore raw PDFs
│   └── explore_pdfs.py            # Analyzes character quality in each PDF
│
├── data/
│   ├── aysha/                     # Aysha's collected PDF documents
│   ├── haleema/                   # Haleema's collected PDF documents
│   ├── vithusan/                  # Vithusan's collected PDF documents
│   └── outputs/                   # Auto-generated exploration reports (local only)
│
├── TEAM_GUIDE.md                  # Step-by-step setup guide for team members
├── requirements.txt               # All Python dependencies with versions and purposes
└── README.md                      # This file
```

> **Note:** PDF files are not committed to GitHub (they are too large).  
> Each team member stores their PDFs locally in their own `data/{name}/` folder.

---

## 🗺️ Research Pipeline (Work Plan)

| Step | Stage | Status | Folder |
|---|---|---|---|
| 1 | Literature Review | ✅ Done | — |
| 2 | Data Collection | ✅ Done | `data/` |
| 3 | Data Preparation (Load & Explore) | ✅ Done | `01_Load_pdfs/` |
| 4 | Data Cleaning | ✅ Prepared corpus validated | `02_data_cleaning/` |
| 5 | Chunking Implementation | 🔄 Fixed-size complete; four methods remaining | `rag_pipeline/chunking/` |
| 6 | RAG System Setup | ⏳ Upcoming | `04_rag_system/` |
| 7 | Experiment & Evaluation | ⏳ Upcoming | `05_evaluation/` |
| 8 | Failure Mode Analysis | ⏳ Upcoming | `06_failure_analysis/` |
| 9 | Report Writing | ⏳ Upcoming | — |

---

## 🚀 Quick Start

### First Time Setup (New Team Member)

See **[TEAM_GUIDE.md](./TEAM_GUIDE.md)** for the full step-by-step guide, including:
- Installing Python 3.11 on Windows
- Installing `uv` (package manager)
- Creating the virtual environment
- Installing all dependencies
- Running the exploration script

### Running the PDF Exploration Script (Step 3)

```bash
# 1. Activate the virtual environment
source venv/bin/activate          # Linux / Mac
venv\Scripts\activate             # Windows

# 2. Run the script (change "vithusan" to your name first)
python 01_Load_pdfs/explore_pdfs.py

# 3. View your output report
# Saved at: data/outputs/explore_pdf_output_{your_name}.txt
```

---

## 📊 Evaluation Metrics

| Metric | Type | What It Measures |
|---|---|---|
| Recall@K | Retrieval | Fraction of required source evidence units covered by the top-K spans |
| MRR (Mean Reciprocal Rank) | Retrieval | How highly is the correct chunk ranked? |
| Faithfulness | Generation | Is the answer grounded in retrieved context? |
| Answer Relevancy | Generation | Does the answer address the question? |
| Context Precision | Generation | Are the right chunks ranked highest? |
| Context Recall | Generation | Does the retrieved context contain the full answer? |

---

## 📚 Key References

1. Lewis et al. (2020) — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* — [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
2. Günther et al. (2024) — *Late Chunking: Contextual Chunk Embeddings* — [arXiv:2409.04701](https://arxiv.org/abs/2409.04701)
3. Es et al. (2023) — *RAGAS: Automated Evaluation of RAG* — [arXiv:2309.15217](https://arxiv.org/abs/2309.15217)
4. Anthropic (2024) — *Contextual Retrieval* — [anthropic.com](https://www.anthropic.com/news/contextual-retrieval)

---

## Sharing code through Git

Keep `data/` and `final_data/` on your own computer. Both directories and PDF
files are ignored by Git, including PDFs placed outside the data folders.
After cloning, create your own `data/{name}/` folder and add your source PDFs there.

Before committing and pushing code:

```bash
git status
git add .
git diff --cached --stat
git commit -m "Describe your code changes"
git push origin main
```

The staged changes should contain code and documentation. Avoid `git add -f`,
which bypasses ignore rules. GitHub's web upload also bypasses `.gitignore`;
upload code through your local Git workflow so these rules take effect.

## 📄 License

This project is licensed under the MIT License — see [LICENSE](./LICENSE) for details.
