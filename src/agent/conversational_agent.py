import argparse
import json
import os
import sys
from typing import List, Dict, Any, Optional
from openai import OpenAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.retrieval.hybrid_search import HybridRetriever


class ConversationalAgent:
    def __init__(
        self, 
        transcripts_path: str = "data/processed/final_transcripts.json",
        db_dir: str = "data/index/chroma_db", 
        llm_model: str = "llama3.1", 
        ollama_host: str = "http://localhost:11434/v1",
        retriever: Optional[HybridRetriever] = None
    ):
        print("Initializing Task 2 Conversational Agent...")
        self.llm_model = llm_model
        self.ollama_host = ollama_host
        
        if retriever is not None:
            self.retriever = retriever
        else:
            self.retriever = HybridRetriever(transcripts_path=transcripts_path, db_dir=db_dir)
        
        self.llm_client = OpenAI(base_url=ollama_host, api_key="ollama")
        self.last_standalone_query: str = ""
        self.last_retrieved_chunks: List[Dict[str, Any]] = []

    def rewrite_query(self, user_query: str, chat_history: List[Dict[str, str]]) -> str:
        """Converts contextual follow-ups into standalone queries with safety guards."""
        if not chat_history:
            self.last_standalone_query = user_query
            return user_query

        history_str = ""
        for turn in chat_history[-4:]:
            history_str += f"{turn['role'].upper()}: {turn['content']}\n"

        prompt = (
            "Given the conversation history and follow-up query, rewrite it into a single, standalone retrieval search query.\n"
            "Do NOT answer the question. Output ONLY the rewritten standalone query string.\n\n"
            f"History:\n{history_str}\n"
            f"Follow-up Query: {user_query}\n\n"
            "Standalone Search Query:"
        )

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            rewritten = response.choices[0].message.content.strip()
            
            if len(rewritten.split()) > 30 or not rewritten:
                self.last_standalone_query = user_query
            else:
                self.last_standalone_query = rewritten
        except Exception:
            self.last_standalone_query = user_query

        return self.last_standalone_query

    def chat(
        self, 
        user_query: str, 
        chat_history: Optional[List[Dict[str, str]]] = None, 
        top_k: int = 4,
        relevance_threshold: float = -2.0
    ) -> str:
        """Executes stateless chat generation using passed conversation history."""
        history = chat_history if chat_history is not None else []
        standalone_query = self.rewrite_query(user_query, history)
        
        # Extract prior cited transcript IDs for evidence carry-over
        prior_t_ids = []
        for msg in history:
            for chunk in self.last_retrieved_chunks:
                t_id = chunk["metadata"].get("transcript_id")
                if t_id and t_id not in prior_t_ids:
                    prior_t_ids.append(t_id)

        retrieved_chunks = self.retriever.search(
            query=standalone_query, 
            final_top_k=top_k, 
            boosted_transcript_ids=prior_t_ids
        )
        self.last_retrieved_chunks = retrieved_chunks

        max_score = max([c.get("rerank_score", -999.0) for c in retrieved_chunks]) if retrieved_chunks else -999.0
        
        # Dialogue Abstention Guard
        if not retrieved_chunks or max_score < relevance_threshold:
            return "I could not find sufficient evidence in the call transcripts to answer this follow-up question accurately."

        formatted_context = ""
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            meta = chunk["metadata"]
            formatted_context += (
                f"\n--- [Evidence Source {idx}] ---\n"
                f"Transcript ID: {meta['transcript_id']} | Turns: {meta['start_turn']}-{meta['end_turn']}\n"
                f"Content: {chunk['text']}\n"
            )

        system_prompt = (
            "You are a concise AI support analyst assistant.\n\n"
            "RULES:\n"
            "1. Answer in 2-3 short bullet points OR 3 sentences max.\n"
            "2. Always cite evidence using [Transcript ID: ..., Turns: ...].\n"
            "3. Ground answers strictly in the context evidence.\n"
            "4. Eliminate all preamble conversational fluff."
        )

        messages = [{"role": "system", "content": system_prompt}]
        for turn in history:
            messages.append(turn)

        user_turn_content = f"User Question: {user_query}\n\nRetrieved Context Evidence:\n{formatted_context}"
        messages.append({"role": "user", "content": user_turn_content})

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.2
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating agent response: {str(e)}"