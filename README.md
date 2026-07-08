# 🛡️ Insurance Claim RAG Agent

An intelligent **Retrieval-Augmented Generation (RAG)** chatbot for insurance policy Q&A — built with a fine-tuned embedding model, parent-child retrieval, HNSW vector indexing, and a dual-tier conversational memory system.

> Every response is **grounded strictly** in retrieved policy chunks with cited section and clause numbers. Zero hallucination on policy queries.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-🦜-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-HNSW-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Chat_UI-red?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker)
![Railway](https://img.shields.io/badge/Railway-Deployed-purple?logo=railway)

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Evaluation Results](#-evaluation-results)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Usage](#-usage)
- [How It Works](#-how-it-works)
- [Deployment](#-deployment)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Parent-Child Retrieval** | Matches queries against fine-grained child chunks (~500 words) but returns the full parent section (~2000 words) as context |
| 🧠 **Dual-Tier Memory** | Short-term buffer (last 2 turns) + long-term vector store for older conversations |
| 🔀 **Query Routing** | LLM-based classifier distinguishes self-contained (Type A) from context-dependent (Type B) queries |
| ✏️ **Silent Query Rewriting** | Ambiguous follow-up questions are silently rewritten into standalone queries before retrieval |
| 🏷️ **Metadata Filtering** | Filter retrieval by specific policy sections and clause numbers |
| ⚡ **HNSW Indexing** | ChromaDB with HNSW (cosine, ef=100, M=16) for fast approximate nearest neighbor search |
| 📎 **Citation Transparency** | Expandable citation panel shows exact retrieved chunks with similarity scores |
| 🎯 **Fine-Tuned Embeddings** | Domain-adapted `all-mpnet-base-v2` for insurance-specific semantic understanding |

---

## 🏗️ Architecture

```
                         ┌─────────────────────┐
                         │     User Query       │
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │   Query Router       │
                         │  (Type A / Type B)   │
                         └──────┬───────┬──────┘
                                │       │
                    Type A      │       │      Type B
                    (Direct)    │       │      (Rewrite)
                                │       │
                                │  ┌────▼─────────────┐
                                │  │  Short-Term       │
                                │  │  Memory Buffer    │
                                │  │  (Last 2 Turns)   │
                                │  └────┬─────────────┘
                                │       │
                                │  ┌────▼─────────────┐
                                │  │  Query Rewriter   │
                                │  │  (Silent, LLM)    │
                                │  └────┬─────────────┘
                                │       │
                         ┌──────▼───────▼──────┐
                         │  Embedding Model     │
                         │  (Fine-tuned MPNet)  │
                         └──────────┬──────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
   ┌──────────▼──────────┐  ┌──────▼───────┐  ┌─────────▼─────────┐
   │  ChromaDB HNSW      │  │  Long-Term   │  │  Metadata Filter  │
   │  (Child Chunks)     │  │  Memory      │  │  (Section/Clause) │
   └──────────┬──────────┘  └──────┬───────┘  └─────────┬─────────┘
              │                     │                     │
              └─────────────────────┼─────────────────────┘
                                    │
                         ┌──────────▼──────────┐
                         │  Parent Docstore    │
                         │  (Full Sections)    │
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │  LLM Response       │
                         │  (Llama 3.3 70B)    │
                         │  + Citations        │
                         └─────────────────────┘
```

---

## 📊 Evaluation Results

Fine-tuned `all-mpnet-base-v2` on a domain-specific insurance retrieval dataset and evaluated using **hybrid search** (75% dense + 25% BM25) on 25 query-answer pairs:

| Metric | Base Model | Fine-Tuned Model | Improvement |
|:---|:---:|:---:|:---:|
| **Recall@1** | 0.4000 | **0.5600** | **↑ 40.0%** |
| **MRR** | 0.5463 | **0.6474** | **↑ 18.5%** |
| **Recall@5** | 0.7600 | 0.7200 | ↓ 5.3% |

> The fine-tuned model significantly improves top-1 precision and mean reciprocal rank, which are the metrics that matter most for a chatbot where the first retrieved chunk drives the answer quality.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Embedding Model** | [`shaurya23102/insurance-embedding-model`](https://huggingface.co/shaurya23102/insurance-embedding-model) (fine-tuned all-mpnet-base-v2) |
| **Vector Database** | ChromaDB with HNSW indexing (cosine similarity) |
| **Text Splitting** | LangChain `RecursiveCharacterTextSplitter` |
| **LLM** | Llama 3.3 70B via Groq API |
| **Framework** | LangChain Core |
| **Frontend** | Streamlit (dark-themed chat UI) |
| **Containerization** | Docker (CPU-only PyTorch) |
| **Deployment** | Railway |

---

## 📁 Project Structure

```
├── app.py                    # Streamlit chat interface
├── rag_engine.py             # Core RAG pipeline (retrieval, memory, routing)
├── insurance_policy.md       # Source policy document
├── requirements.txt          # Python dependencies
├── Dockerfile                # Container definition for Railway
├── .streamlit/
│   └── config.toml           # Streamlit dark theme config
└── README.md
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.11+
- [Groq API Key](https://console.groq.com/) (free tier available)

### Local Setup

```bash
# Clone the repository
git clone https://github.com/Shaurya23102/Insurace_claim_rag.git
cd Insurace_claim_rag

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
.\venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variable
export GROQ_API_KEY="your_groq_api_key_here"    # Linux/Mac
set GROQ_API_KEY=your_groq_api_key_here         # Windows

# Run the app
streamlit run app.py
```

The app will be live at **http://localhost:8501**

> **First launch** takes ~30-60 seconds to download the embedding model from Hugging Face Hub and build the HNSW index (174 child chunks).

---

## 💬 Usage

### Basic Query (Type A — Self-Contained)
```
You: What does the policy state about ICU charges?

Agent: As per Section: 1. Inpatient Treatment | Clause: 1, the policy covers
Intensive Care Unit (ICU) Charges, including ICU bed, monitoring devices,
critical care nursing, and intensivist charges...
```

### Follow-Up Query (Type B — Context-Dependent)
```
You: What about Day Care Procedures? Is there a limit on them too?

  ↳ Router → Type B
  ↳ Rewrite: "What are the coverage limits and rules for Day Care
     Procedures as stated in Section 2 of the policy?"

Agent: As per Section: 2. Day Care Procedures/Treatment | Clause: 2,
the policy covers Medical Expenses up to the Annual Sum Insured...
```

### Metadata-Filtered Query
Use the sidebar dropdown to restrict retrieval to a specific policy section (e.g., only "c DEFINITIONS").

---

## ⚙️ How It Works

### 1. Document Ingestion
- The policy markdown is split on `##` headings to extract sections
- Each section is split into **parent chunks** (~2000 words) and **child chunks** (~500 words)
- Metadata is attached: `policy_name`, `section_title`, `clause_number`
- Child chunk embeddings are indexed in ChromaDB with HNSW configuration

### 2. Query Classification & Rewriting
- Every query is classified by an LLM as **Type A** (self-contained) or **Type B** (context-dependent)
- Type B queries are silently rewritten into standalone questions using the conversational buffer
- The rewritten query is used for all downstream retrieval

### 3. Retrieval
- The query embedding is matched against child chunks via HNSW cosine search
- Optional metadata filters restrict search to specific sections
- Matched child chunks are mapped back to their **full parent sections** for richer context

### 4. Memory
- **Short-term**: Last 2 conversation turns are injected directly into the prompt
- **Long-term**: Older turns are embedded and stored in a separate vector collection; top-2 relevant past turns are retrieved per query

### 5. Response Generation
- The LLM receives: policy context + short-term buffer + long-term memory + query
- Responses cite specific sections and clause numbers
- Ambiguous/edge cases are flagged for human escalation

---

## 🌐 Deployment

### Deploy on Railway

1. Push this repository to GitHub
2. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub**
3. Select this repository
4. Add environment variable: `GROQ_API_KEY` = your key
5. Railway auto-detects the `Dockerfile` and deploys
6. Go to **Settings → Networking → Generate Domain** for your public URL

### Docker (Manual)

```bash
docker build -t insurance-rag-agent .
docker run -p 8501:8501 -e GROQ_API_KEY="your_key" insurance-rag-agent
```

---


