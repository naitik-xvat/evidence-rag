import json
import os
import sys
from typing import Dict, Any, Optional
from openai import OpenAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.retrieval.hybrid_search import HybridRetriever


def run_causal_analysis(
    query: str, 
    db_dir: str = "data/index/chroma_db",
    transcripts_path: str = "data/processed/final_transcripts.json",
    top_k: int = 4, 
    llm_model: str = "llama3.1",
    ollama_host: str = "http://localhost:11434/v1",
    retriever: Optional[HybridRetriever] = None,
    relevance_threshold: float = -2.0
) -> Dict[str, Any]:
    """Performs evidence-backed root cause analysis with safety abstention guardrails."""
    
    if retriever is None:
        retriever = HybridRetriever(transcripts_path=transcripts_path, db_dir=db_dir)
        
    retrieved_chunks = retriever.search(query=query, final_top_k=top_k)
    max_score = max([c.get("rerank_score", -999.0) for c in retrieved_chunks]) if retrieved_chunks else -999.0
    
    # Calibrated Abstention Guard for out-of-bounds queries
    if not retrieved_chunks or max_score < relevance_threshold:
        return {
            "analysis": "**Status:** Insufficient Transcript Evidence / Out-of-Bounds Query\n\n"
                        "**Notice:** The system could not identify relevant call transcript evidence in the corpus to answer this query safely.",
            "evidence_sources_count": 0,
            "abstained": True,
            "max_rerank_score": round(max_score, 4),
            "retrieved_chunks": []
        }
    
    formatted_evidence = ""
    saved_chunks_data = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        meta = chunk["metadata"]
        saved_chunks_data.append({
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "transcript_id": meta.get("transcript_id", ""),
            "start_turn": meta.get("start_turn", 0),
            "end_turn": meta.get("end_turn", 0),
            "rerank_score": round(chunk.get("rerank_score", 0.0), 4)
        })
        formatted_evidence += (
            f"\n[Evidence Source {idx}] (Transcript ID: {meta['transcript_id']}, Turns {meta['start_turn']}-{meta['end_turn']})\n"
            f"Domain: {meta.get('domain', 'N/A')} | Intent: {meta.get('intent', 'N/A')}\n"
            f"Dialogue Content: {chunk['text']}\n"
        )

    system_prompt = (
        "You are an operational Causal Analytics Engine for Contact Center Intelligence.\n"
        "Analyze the provided call transcript evidence and provide a structured root-cause response.\n\n"
        "REQUIRED FORMAT:\n"
        "**Root Cause:** Concise 1-2 sentence core mechanism.\n"
        "**Contributing Triggers:**\n"
        "* Trigger 1 (with Transcript ID citation)\n"
        "* Trigger 2 (with Transcript ID citation)\n"
        "**Transcript Evidence:** Quoted verbatim phrase from context inside double quotes with Transcript ID and Turn range.\n"
        "**Recommended Fix:** 1 actionable operational remedy."
    )

    user_prompt = f"Query: {query}\n\nEvidence Context:\n{formatted_evidence}"

    try:
        llm_client = OpenAI(base_url=ollama_host, api_key="ollama")
        response = llm_client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        cleaned_analysis = response.choices[0].message.content.strip()
    except Exception as e:
        cleaned_analysis = f"LLM Generation Error: {str(e)}"

    return {
        "analysis": cleaned_analysis,
        "evidence_sources_count": len(retrieved_chunks),
        "abstained": False,
        "max_rerank_score": round(max_score, 4),
        "retrieved_chunks": saved_chunks_data
    }