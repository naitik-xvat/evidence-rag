import json
import os
import re
import sys
import random
from typing import List, Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.retrieval.hybrid_search import HybridRetriever


def evaluate_retrieval_ablation(retriever: HybridRetriever, sample_size: int = 30) -> Dict[str, Any]:
    """Computes Recall@4 and MRR across Dense-only, BM25-only, and Hybrid+Reranker."""
    print("Evaluating Retrieval Ablation (Dense vs BM25 vs Hybrid+Rerank)...")
    
    random.seed(42)
    sample_indices = random.sample(range(len(retriever.chunks_corpus)), min(sample_size, len(retriever.chunks_corpus)))
    
    test_cases = []
    for idx in sample_indices:
        chunk = retriever.chunks_corpus[idx]
        target_t_id = chunk["metadata"]["transcript_id"]
        # Formulate query from metadata intent or reason for call
        query = chunk["metadata"].get("intent") or chunk["metadata"].get("reason_for_call") or " ".join(chunk["text"].split()[:8])
        test_cases.append({"query": query, "target_transcript_id": target_t_id})

    methods = ["dense", "bm25", "hybrid_rerank"]
    results = {m: {"recall_at_4": 0.0, "mrr": 0.0} for m in methods}

    for case in test_cases:
        q = case["query"]
        target = case["target_transcript_id"]

        # 1. Dense Only
        dense_res = [c["metadata"].get("transcript_id") for c in retriever.dense_search(q, top_k=4)]
        if target in dense_res:
            results["dense"]["recall_at_4"] += 1.0
            results["dense"]["mrr"] += 1.0 / (dense_res.index(target) + 1)

        # 2. BM25 Only
        bm25_res = [c["metadata"].get("transcript_id") for c in retriever.bm25_search(q, top_k=4)]
        if target in bm25_res:
            results["bm25"]["recall_at_4"] += 1.0
            results["bm25"]["mrr"] += 1.0 / (bm25_res.index(target) + 1)

        # 3. Hybrid + Rerank
        hybrid_res = [c["metadata"].get("transcript_id") for c in retriever.search(q, final_top_k=4)]
        if target in hybrid_res:
            results["hybrid_rerank"]["recall_at_4"] += 1.0
            results["hybrid_rerank"]["mrr"] += 1.0 / (hybrid_res.index(target) + 1)

    N = max(len(test_cases), 1)
    summary = {}
    for m in methods:
        summary[m] = {
            "Recall@4": round(results[m]["recall_at_4"] / N, 4),
            "MRR": round(results[m]["mrr"] / N, 4)
        }
    return summary


def normalize_text(text: str) -> str:
    return re.sub(r'[^\w\s]', '', text.lower()).strip()


def evaluate_quote_grounding(dataset_path: str) -> Dict[str, Any]:
    """Verifies generated quote text against retrieved chunk texts."""
    if not os.path.exists(dataset_path):
        return {"error": "Dataset path not found"}

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_extracted_quotes = 0
    grounded_quotes = 0

    for eval_item in data.get("evaluations", []):
        # Task 1
        if "system_output" in eval_item:
            output_text = eval_item["system_output"]
            chunks = eval_item.get("retrieved_chunks", [])
            retrieved_combined_text = " ".join([c.get("text", "") for c in chunks])
            norm_retrieved = normalize_text(retrieved_combined_text)

            quotes = re.findall(r'"([^"]{6,})"', output_text)
            for q in quotes:
                total_extracted_quotes += 1
                norm_q = normalize_text(q)
                if norm_q in norm_retrieved or any(norm_q in normalize_text(c.get("text", "")) for c in chunks):
                    grounded_quotes += 1

        # Task 2 Turns
        elif "dialogue_sequence" in eval_item:
            for turn in eval_item["dialogue_sequence"]:
                output_text = turn.get("agent_response", "")
                chunks = turn.get("retrieved_chunks", [])
                norm_retrieved = normalize_text(" ".join([c.get("text", "") for c in chunks]))

                quotes = re.findall(r'"([^"]{6,})"', output_text)
                for q in quotes:
                    total_extracted_quotes += 1
                    norm_q = normalize_text(q)
                    if norm_q in norm_retrieved:
                        grounded_quotes += 1

    grounding_rate = (grounded_quotes / total_extracted_quotes) if total_extracted_quotes > 0 else 1.0
    return {
        "total_extracted_quotes": total_extracted_quotes,
        "grounded_quotes": grounded_quotes,
        "quote_grounding_rate": round(grounding_rate, 4)
    }


def main():
    db_dir = os.path.abspath("data/index/chroma_db")
    transcripts_path = os.path.abspath("data/processed/final_transcripts.json")
    dataset_path = os.path.abspath("results/query_dataset_outputs.json")

    retriever = HybridRetriever(transcripts_path=transcripts_path, db_dir=db_dir)
    ablation_metrics = evaluate_retrieval_ablation(retriever, sample_size=30)
    grounding_metrics = evaluate_quote_grounding(dataset_path)

    final_metrics = {
        "retrieval_ablation": ablation_metrics,
        "quote_grounding": grounding_metrics
    }

    os.makedirs("results", exist_ok=True)
    out_file = "results/evaluation_metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    print("\n" + "="*50)
    print(" QUANTITATIVE EVALUATION SUMMARY")
    print("="*50)
    print(json.dumps(final_metrics, indent=2))
    print(f"\n✅ Metrics successfully calculated and saved to: {out_file}")

if __name__ == "__main__":
    main()