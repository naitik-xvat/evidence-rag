# 📊 Contact Center Causal Analytics & Dialogue Agent

A local-first **Hybrid Retrieval-Augmented Generation (RAG)** system designed to perform root-cause analysis and multi-turn conversational analysis over contact center call transcripts.

The platform combines:

- **Dense Vector Search** using `all-MiniLM-L6-v2`
- **Sparse Keyword Retrieval** using `rank_bm25`
- **Reciprocal Rank Fusion (RRF)**
- **Cross-Encoder Re-ranking** using `BAAI/bge-reranker-base`
- **Local Open-Weight LLMs** using Llama 3.1 via Ollama

The system focuses on evidence-backed retrieval, quote-supported analysis, contextual follow-up questions, and local inference.

---

## 🌟 Key Features

### 🔍 Task 1: Query-Driven Causal Root-Cause Engine

- **Corpus-Wide Pattern Analysis:** Identifies operational friction, customer churn triggers, billing disputes, and agent policy refusals across call transcripts.
- **Verbatim Evidence Citations:** Root-cause claims are supported by transcript quotes, Transcript IDs, and turn ranges.
- **Score-Based Safety Abstention:** Uses a Cross-Encoder relevance threshold (`relevance_threshold = -2.0`) to abstain from unsupported or low-relevance queries.
- **Corpus Multi-Transcript Deduplication:** Limits retrieval density per transcript (`max_chunks_per_transcript = 2`) so that retrieved evidence can span multiple conversations.

### 💬 Task 2: Interactive Multi-Turn Dialogue Agent

- **Contextual Query Reformulation:** Converts ambiguous or pronoun-heavy follow-up questions into standalone search queries using previous dialogue context.
- **Session-Level Conversation State:** Maintains conversation history for follow-up questions while keeping the shared retriever independent of individual sessions.
- **Evidence Carry-Over & Prior Citation Boosting:** Tracks previously cited `transcript_ids` and uses them during subsequent searches.
- **Local LLM Pipeline:** Uses a locally hosted LLM through Ollama for response generation.

---