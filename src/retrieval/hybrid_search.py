import json
import os
import re
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict, Any, Optional

class HybridRetriever:
    def __init__(
        self, 
        transcripts_path: str = "data/processed/final_transcripts.json", 
        db_dir: str = "data/index/chroma_db",
        reranker_model: str = "BAAI/bge-reranker-base"
    ):
        print("Initializing Hybrid Retriever (Dense + BM25 + RRF + Cross-Encoder)...")
        
        # 1. Load ChromaDB Vector Store
        abs_db_dir = os.path.abspath(db_dir)
        self.chroma_client = chromadb.PersistentClient(path=abs_db_dir)
        self.collection = self.chroma_client.get_collection(name="conversations")
        self.dense_encoder = SentenceTransformer("all-MiniLM-L6-v2")

        # 2. Build BM25 Index with Clean Tokenization
        print(f"Loading chunks from {transcripts_path} for BM25 indexing...")
        with open(transcripts_path, 'r', encoding='utf-8') as f:
            transcripts = json.load(f)

        self.chunks_corpus = []
        tokenized_corpus = []

        for item in transcripts:
            t_id = item["transcript_id"]
            domain = item.get("domain", "")
            intent = item.get("intent", "")
            reason = item.get("reason_for_call", "")
            
            for chunk in item.get("chunks", []):
                chunk_record = {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "metadata": {
                        "transcript_id": t_id,
                        "start_turn": chunk["start_turn"],
                        "end_turn": chunk["end_turn"],
                        "domain": domain,
                        "intent": intent,
                        "reason_for_call": reason
                    }
                }
                self.chunks_corpus.append(chunk_record)
                
                # Strip punctuation and tokenize cleanly
                clean_text = re.sub(r'[^\w\s]', '', chunk["text"].lower())
                tokenized_corpus.append(clean_text.split())

        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"BM25 initialized over {len(self.chunks_corpus)} chunks.")

        # 3. Load Cross-Encoder Re-ranker
        print(f"Loading Cross-Encoder Reranker ({reranker_model})...")
        self.reranker = CrossEncoder(reranker_model)

    def dense_search(self, query: str, top_k: int = 25, domain_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        query_vec = self.dense_encoder.encode(query).tolist()
        where_clause = {"domain": domain_filter} if domain_filter else None
        
        results = self.collection.query(
            query_embeddings=[query_vec], 
            n_results=top_k,
            where=where_clause
        )
        
        if not results["documents"] or not results["documents"][0]:
            return []

        candidates = []
        for c_id, doc, meta in zip(results["ids"][0], results["documents"][0], results["metadatas"][0]):
            candidates.append({
                "chunk_id": c_id,
                "text": doc,
                "metadata": meta
            })
        return candidates

    def bm25_search(self, query: str, top_k: int = 25, domain_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        clean_query = re.sub(r'[^\w\s]', '', query.lower())
        tokenized_query = clean_query.split()
        top_indices = self.bm25.get_top_n(tokenized_query, range(len(self.chunks_corpus)), n=top_k * 2)
        
        candidates = []
        for idx in top_indices:
            chunk = self.chunks_corpus[idx]
            if domain_filter and chunk["metadata"].get("domain", "").lower() != domain_filter.lower():
                continue
            candidates.append(dict(chunk))
            if len(candidates) >= top_k:
                break
        return candidates

    def reciprocal_rank_fusion(
        self, 
        dense_results: List[Dict[str, Any]], 
        bm25_results: List[Dict[str, Any]], 
        k: int = 60,
        top_n: int = 30
    ) -> List[Dict[str, Any]]:
        scores = {}
        candidate_map = {}

        for rank, doc in enumerate(dense_results, start=1):
            c_id = doc["chunk_id"]
            scores[c_id] = scores.get(c_id, 0.0) + (1.0 / (k + rank))
            candidate_map[c_id] = dict(doc)

        for rank, doc in enumerate(bm25_results, start=1):
            c_id = doc["chunk_id"]
            scores[c_id] = scores.get(c_id, 0.0) + (1.0 / (k + rank))
            candidate_map[c_id] = dict(doc)

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [candidate_map[c_id] for c_id in sorted_ids[:top_n]]

    def deduplicate_by_transcript(self, candidates: List[Dict[str, Any]], max_per_transcript: int = 2) -> List[Dict[str, Any]]:
        """Ensures evidence spans multiple distinct calls across the corpus."""
        counts = {}
        deduped = []
        for doc in candidates:
            t_id = doc["metadata"].get("transcript_id", doc["chunk_id"])
            counts[t_id] = counts.get(t_id, 0)
            if counts[t_id] < max_per_transcript:
                deduped.append(doc)
                counts[t_id] += 1
        return deduped

    def rerank(self, query: str, candidate_chunks: List[Dict[str, Any]], top_k: int = 8) -> List[Dict[str, Any]]:
        if not candidate_chunks:
            return []
        
        pairs = [[query, chunk["text"]] for chunk in candidate_chunks]
        scores = self.reranker.predict(pairs)

        output_chunks = []
        for chunk, score in zip(candidate_chunks, scores):
            c_copy = dict(chunk)
            c_copy["rerank_score"] = float(score)
            output_chunks.append(c_copy)

        reranked = sorted(output_chunks, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    def search(
        self, 
        query: str, 
        final_top_k: int = 8, 
        top_k: Optional[int] = None,
        domain_filter: Optional[str] = None,
        boosted_transcript_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        limit = top_k if top_k is not None else final_top_k
        dense_candidates = self.dense_search(query, top_k=25, domain_filter=domain_filter)
        bm25_candidates = self.bm25_search(query, top_k=25, domain_filter=domain_filter)
        
        fused_candidates = self.reciprocal_rank_fusion(dense_candidates, bm25_candidates, top_n=30)
        deduped_candidates = self.deduplicate_by_transcript(fused_candidates, max_per_transcript=2)
        
        # Boost prior turn transcripts for Task 2 carry-over
        if boosted_transcript_ids:
            for cand in deduped_candidates:
                if cand["metadata"].get("transcript_id") in boosted_transcript_ids:
                    cand["text"] = "[Prior Cited Call] " + cand["text"]

        return self.rerank(query, deduped_candidates, top_k=limit)