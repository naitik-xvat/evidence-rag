# 📊 Contact Center Causal Analytics & Dialogue Agent (Hybrid RAG)

An enterprise-grade, evidence-backed, local-first **Hybrid Retrieval-Augmented Generation (RAG)** system designed to extract root-cause analytics and power multi-turn conversational intelligence from contact center call transcripts. 

The platform combines **Dense Vector Search (`all-MiniLM-L6-v2`)**, **Sparse Keyword Matching (`rank_bm25`)**, **Reciprocal Rank Fusion (RRF)**, **Cross-Encoder Re-ranking (`BAAI/bge-reranker-base`)**, and **Local Open-Weight LLMs (Llama 3.1 via Ollama)** to deliver grounded, quote-backed insights without data leakage or hallucinations.

---

## 🌟 Key Features & Business Capabilities

### 🔍 Task 1: Query-Driven Causal Root-Cause Engine
* **Corpus-Wide Pattern Analysis:** Identifies operational friction, customer churn triggers, billing disputes, and agent policy refusals across thousands of call transcripts.
* **Verbatim Evidence Citations:** Every root-cause claim is backed by direct transcript quotes, Transcript IDs, and exact turn ranges.
* **Score-Based Safety Abstention Guard:** Implements a calibrated Cross-Encoder confidence threshold (`relevance_threshold = -2.0`). For out-of-bounds queries (e.g., internal salaries or unsupported domains), the engine safely abstains rather than generating hallucinated answers.
* **Corpus Multi-Transcript Deduplication:** Limits retrieval density per transcript (`max_chunks_per_transcript = 2`) to ensure evidence spans multiple callers rather than repeating adjacent chunks from a single conversation.

### 💬 Task 2: Interactive Multi-Turn Dialogue Agent
* **Contextual Query Reformulation:** Automatically converts ambiguous or pronoun-heavy follow-up questions into standalone search queries using past dialogue context.
* **Stateless Memory Architecture:** Decouples session chat history from the global cached retriever, preventing state leakage across user sessions or browser refreshes.
* **Evidence Carry-Over & Prior Citation Boosting:** Tracks previously cited `transcript_ids` across turns to boost relevant transcripts during follow-up searches.
* **Single-Pass LLM Pipeline:** Integrates query rewriting and context retrieval into a unified execution pass to minimize latency on local hardware.
  
---

## 📁 Repository Structure

```text
evidence-rag/
├── data/                             # Raw & Processed Transcripts (Git Ignored)
│   ├── raw/
│   │   └── conversations.json        # Raw call dataset
│   ├── processed/
│   │   └── final_transcripts.json    # Cleaned & chunked transcripts
│   └── index/
│       └── chroma_db/                # ChromaDB HNSW vector database
├── src/                              # Core Source Engine
│   ├── data_processing/
│   │   └── parse_transcripts.py      # Turn parsing & sliding-window chunker
│   ├── retrieval/
│   │   ├── indexer.py                # Dense vector embeddings & indexing
│   │   └── hybrid_search.py          # Dense + BM25 + RRF + Reranker pipeline
│   ├── causal_engine/
│   │   └── causal_reasoner.py        # Task 1 root-cause analysis & abstention
│   └── agent/
│       └── conversational_agent.py   # Task 2 multi-turn conversational agent
├── evaluation/                       # Evaluation & Benchmark Suite
│   ├── generate_dataset.py           # Runs 50+ query benchmark suite
│   └── evaluate_metrics.py           # Quantitative Recall@K, MRR & Grounding metrics
├── results/                          # Output Evaluation Deliverables
│   ├── query_dataset_outputs.json    # Final 50+ query outputs
│   └── evaluation_metrics.json       # Quantitative performance report
├── app.py                            # Streamlit Web Application
├── run.sh                            # Automated Master Shell Launcher
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Security & data privacy filters
└── README.md                         # Documentation

---
## Installation & Setup

* Install and verify Ollama
    ollama serve
    ollama pull llama3.1

* Clone Repository & Setup Virtual Environment
    
    git clone <repository_url>
    cd evidence-rag

    # Create Python virtual environment
    python -m venv .venv

    # Activate environment
    # On Linux/macOS:
    source .venv/bin/activate
    # On Windows PowerShell:
    # .\.venv\Scripts\Activate.ps1

    # Upgrade pip and install dependencies
    pip install --upgrade pip
    pip install -r requirements.txt

---
## Execution
chmod +x run.sh
./run.sh