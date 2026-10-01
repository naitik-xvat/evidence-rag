import streamlit as st
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.causal_engine.causal_reasoner import run_causal_analysis
from src.agent.conversational_agent import ConversationalAgent
from src.retrieval.hybrid_search import HybridRetriever

# ---------------------------------------------------------
# Streamlit High-Contrast CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Contact Center Causal Analytics",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    .stApp { background-color: #0b1220; color: #f8fafc; }
    h1, h2, h3, h4 { color: #ffffff !important; }

    section[data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #334155;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label { color: #f8fafc !important; }

    div[data-baseweb="input"],
    div[data-baseweb="select"],
    div[data-baseweb="base-input"],
    .stSlider {
        background-color: #1e293b !important;
        color: #ffffff !important;
        border-radius: 8px;
    }
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        background-color: #1e293b !important;
        border-color: #64748b !important;
    }
    input, textarea {
        color: #ffffff !important;
        background-color: #1e293b !important;
        caret-color: #ffffff !important;
    }
    input::placeholder, textarea::placeholder {
        color: #a8b3c2 !important; opacity: 1 !important;
    }
    div[data-baseweb="select"] * { color: #ffffff !important; }

    .stButton > button {
        background-color: #4f46e5 !important;
        color: #ffffff !important;
        font-weight: 700;
        border-radius: 8px;
        border: 1px solid #818cf8 !important;
        min-height: 42px;
    }
    .stButton > button:hover {
        background-color: #6366f1 !important;
        border-color: #a5b4fc !important;
    }
    .stButton > button p, .stButton > button span { color: #ffffff !important; }
    button[kind="primary"] {
        background-color: #7c3aed !important;
        border-color: #a78bfa !important;
        color: #ffffff !important;
    }
    button[kind="primary"]:hover { background-color: #8b5cf6 !important; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px; background-color: #0f172a; padding: 6px; border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px; background-color: #1e293b;
        border: 1px solid #475569; border-radius: 8px;
        color: #cbd5e1 !important; font-weight: 700;
    }
    .stTabs [data-baseweb="tab"] * { color: #cbd5e1 !important; }
    .stTabs [aria-selected="true"] {
        background-color: #4f46e5 !important;
        border-color: #818cf8 !important; color: #ffffff !important;
    }
    .stTabs [aria-selected="true"] * { color: #ffffff !important; }

    .query-badge {
        background-color: #075985; color: #ffffff !important;
        border: 1px solid #38bdf8; padding: 0.4rem 0.75rem;
        border-radius: 7px; font-family: monospace; font-size: 0.85rem;
        display: inline-block; margin-top: 0.5rem;
    }

    [data-testid="stChatMessage"] {
        background-color: #172033; border: 1px solid #475569;
        border-radius: 10px; padding: 1rem; margin-bottom: 0.75rem;
    }
    [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] span { color: #f8fafc !important; }

    [data-testid="stExpander"] {
        background-color: #111827; border: 1px solid #475569; border-radius: 8px;
    }
    pre, code {
        background-color: #020617 !important; color: #f8fafc !important;
        border: 1px solid #334155;
    }
    hr { border-color: #475569 !important; }
    *:focus-visible { outline: 2px solid #38bdf8 !important; outline-offset: 2px; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Cached Retriever & Agent Initialization
# ---------------------------------------------------------
@st.cache_resource
def get_shared_retriever(transcripts_path: str, db_dir: str):
    """Loads and caches the HybridRetriever instance once across all user sessions."""
    return HybridRetriever(transcripts_path=transcripts_path, db_dir=db_dir)


if "user_query_input" not in st.session_state:
    st.session_state["user_query_input"] = ""

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_history" not in st.session_state:
    st.session_state.agent_history = []


def set_sample_query(query_text: str):
    st.session_state["user_query_input"] = query_text


# Health Checks before UI rendering
db_exists = os.path.exists("data/index/chroma_db")
transcripts_exist = os.path.exists("data/processed/final_transcripts.json")

# Sidebar Controls
with st.sidebar:
    st.title("⚙️ RAG Configuration")
    st.caption("Hybrid Search (Dense + BM25 + RRF + Cross-Encoder)")
    
    model_name = st.selectbox("LLM Model (Ollama)", ["llama3.1", "qwen2.5:3b", "llama3.2:3b"], index=0)
    top_k_chunks = st.slider("Top-K Evidence Chunks", min_value=3, max_value=15, value=4)
    
    st.divider()
    st.subheader("System Status")
    
    if db_exists and transcripts_exist:
        st.success("ChromaDB & Transcripts Online")
    else:
        st.error("Index or Transcripts missing. Run parser and indexer first.")
        
    st.divider()
    if st.button("Clear Chat History (Task 2)", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent_history = []
        st.rerun()


# Initialize RAG components safely if index exists
if db_exists and transcripts_exist:
    retriever = get_shared_retriever("data/processed/final_transcripts.json", "data/index/chroma_db")
    chat_agent = ConversationalAgent(
        transcripts_path="data/processed/final_transcripts.json",
        db_dir="data/index/chroma_db",
        llm_model=model_name,
        retriever=retriever
    )
else:
    retriever = None
    chat_agent = None


# ---------------------------------------------------------
# Main Interface
# ---------------------------------------------------------
st.title("📊 Contact Center Causal Analytics & Dialogue Agent")
st.markdown("Ground-truth evidence retrieval and root-cause analysis powered by Local Open-Weight LLMs.")

tab1, tab2 = st.tabs(["🔍 Task 1: Causal Root-Cause Engine", "💬 Task 2: Interactive Follow-Up Agent"])


# =========================================================
# TAB 1: Causal Analysis
# =========================================================
with tab1:
    st.header("Query-Driven Causal Evidence Analysis")
    st.caption("Extract structured, quote-backed root causes across call transcripts.")
    
    st.markdown("**Sample Analytical Queries:**")
    col_q1, col_q2, col_q3 = st.columns(3)
    
    col_q1.button(
        "🏷️ Why request promo codes?", 
        on_click=set_sample_query, 
        args=("Why do customers request promotional code discounts?",)
    )
    col_q2.button(
        "⚠️ What triggers escalations?", 
        on_click=set_sample_query, 
        args=("Why are callers asking to speak with a manager or supervisor?",)
    )
    col_q3.button(
        "💳 Why dispute billing fees?", 
        on_click=set_sample_query, 
        args=("What triggers customer confusion over prorated initial billing charges?",)
    )

    user_query = st.text_input(
        "Enter your business query:", 
        key="user_query_input",
        placeholder="e.g., Why are customers cancelling their subscriptions?"
    )
    
    if st.button("Run Causal Analysis", type="primary"):
        if not user_query.strip():
            st.warning("Please enter a query.")
        elif not retriever:
            st.error("Index not available. Please run indexer first.")
        else:
            with st.spinner("Executing Hybrid Search, RRF Fusion, Cross-Encoder Reranking, and LLM Causal Reasoning..."):
                try:
                    result = run_causal_analysis(
                        query=user_query,
                        db_dir="data/index/chroma_db",
                        top_k=top_k_chunks,
                        llm_model=model_name,
                        retriever=retriever
                    )
                    
                    with st.container(border=True):
                        st.subheader("📌 Causal Root-Cause Analysis")
                        st.markdown(result.get("analysis", "No analysis generated."))
                        st.caption(f"Retrieved Context Sources: {result.get('evidence_sources_count', 0)} evidence chunks | Max Rerank Score: {result.get('max_rerank_score', 0.0)}")
                    
                    # Display retrieved evidence cards
                    with st.expander("📚 View Retrieved Evidence Chunks & Rerank Scores"):
                        for idx, chunk in enumerate(result.get("retrieved_chunks", []), start=1):
                            st.markdown(f"**Chunk {idx}** | Transcript ID: `{chunk['transcript_id']}` | Turns {chunk['start_turn']}-{chunk['end_turn']} | Score: `{chunk['rerank_score']}`")
                            st.code(chunk["text"], language="markdown")
                        
                except Exception as e:
                    st.error(f"Execution Error: {e}")


# =========================================================
# TAB 2: Multi-Turn Conversational Agent
# =========================================================
with tab2:
    st.header("Interactive Multi-Turn Conversational Agent")
    st.caption("Ask questions about customer call transcripts. Context is tracked automatically across turns.")
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "rewritten" in msg and msg["rewritten"]:
                st.markdown(f"<div class='query-badge'>🔍 Standalone Search Query: {msg['rewritten']}</div>", unsafe_allow_html=True)

    input_placeholder = (
        "Type your question here... (e.g., Why are insurance domain customers threatening escalations?)" 
        if not st.session_state.messages 
        else "Ask a follow-up question... (e.g., Did any agents offer fee waivers?)"
    )

    if prompt := st.chat_input(placeholder=input_placeholder):
        if not chat_agent:
            st.error("Index not available. Please run indexer first.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving grounded dialogue context & generating response..."):
                    answer = chat_agent.chat(
                        prompt, 
                        chat_history=st.session_state.agent_history, 
                        top_k=top_k_chunks
                    )
                    standalone_q = chat_agent.last_standalone_query
                    
                    st.markdown(answer)
                    if standalone_q != prompt:
                        st.markdown(f"<div class='query-badge'>🔍 Standalone Search Query: {standalone_q}</div>", unsafe_allow_html=True)

            # Update session states
            st.session_state.agent_history.append({"role": "user", "content": prompt})
            st.session_state.agent_history.append({"role": "assistant", "content": answer})
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer, 
                "rewritten": standalone_q if standalone_q != prompt else ""
            })