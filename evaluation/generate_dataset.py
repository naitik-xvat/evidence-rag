import json
import os
import sys
import time
import gc
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.hybrid_search import HybridRetriever
from src.causal_engine.causal_reasoner import run_causal_analysis
from src.agent.conversational_agent import ConversationalAgent

# ---------------------------------------------------------
# Comprehensive Benchmark Suite Across All Business Domains
# ---------------------------------------------------------
BENCHMARK_SUITE = [
    # --- TASK 1: ESCALATIONS (8 Queries) ---
    {"id": "Q01", "task": "Task 1", "category": "Escalations", "difficulty": "Simple Retrieval", "query": "Why are callers asking to speak with a manager or supervisor?"},
    {"id": "Q02", "task": "Task 1", "category": "Escalations", "difficulty": "Simple Retrieval", "query": "What agent behaviors trigger immediate requests for supervisor intervention?"},
    {"id": "Q03", "task": "Task 1", "category": "Escalations", "difficulty": "Multi-Hop Reasoning", "query": "How do unhandled billing fee disputes escalate into demands for supervisor intervention and legal threats?"},
    {"id": "Q04", "task": "Task 1", "category": "Escalations", "difficulty": "Multi-Hop Reasoning", "query": "What sequence of policy refusals causes callers to demand manager call-backs?"},
    {"id": "Q05", "task": "Task 1", "category": "Escalations", "difficulty": "Simple Retrieval", "query": "Why are insurance policyholders threatening supervisor escalations during claim filing?"},
    {"id": "Q06", "task": "Task 1", "category": "Escalations", "difficulty": "Ambiguous Query", "query": "Why are callers getting so angry on the line?"},
    {"id": "Q07", "task": "Task 1", "category": "Escalations", "difficulty": "Simple Retrieval", "query": "What friction points during hotel reservation changes cause manager requests?"},
    {"id": "Q08", "task": "Task 1", "category": "Escalations", "difficulty": "Multi-Hop Reasoning", "query": "How do long hold times combined with unresolved tech support issues cause escalations?"},

    # --- TASK 1: PROMOTIONS & DISCOUNTS (8 Queries) ---
    {"id": "Q09", "task": "Task 1", "category": "Promotions", "difficulty": "Simple Retrieval", "query": "Why are customers calling about promo code 'summer20' failing at checkout?"},
    {"id": "Q10", "task": "Task 1", "category": "Promotions", "difficulty": "Simple Retrieval", "query": "Why do customers feel promotional codes are marketing gimmicks?"},
    {"id": "Q11", "task": "Task 1", "category": "Promotions", "difficulty": "Multi-Hop Reasoning", "query": "How does expiration terms ambiguity on promotional discounts trigger claims of misleading advertising?"},
    {"id": "Q12", "task": "Task 1", "category": "Promotions", "difficulty": "Simple Retrieval", "query": "Why are callers complaining about expired loyalty reward vouchers?"},
    {"id": "Q13", "task": "Task 1", "category": "Promotions", "difficulty": "Multi-Hop Reasoning", "query": "What causes customer dissatisfaction when combining promotional discounts with room upgrades?"},
    {"id": "Q14", "task": "Task 1", "category": "Promotions", "difficulty": "Ambiguous Query", "query": "Why didn't my discount apply?"},
    {"id": "Q15", "task": "Task 1", "category": "Promotions", "difficulty": "Simple Retrieval", "query": "What triggers calls regarding missing referral credit rewards?"},
    {"id": "Q16", "task": "Task 1", "category": "Promotions", "difficulty": "Multi-Hop Reasoning", "query": "How do discrepancies between online promo banners and phone representative quotes erode trust?"},

    # --- TASK 1: BILLING & CHARGES (8 Queries) ---
    {"id": "Q17", "task": "Task 1", "category": "Billing", "difficulty": "Simple Retrieval", "query": "What triggers customer confusion over prorated initial billing charges?"},
    {"id": "Q18", "task": "Task 1", "category": "Billing", "difficulty": "Simple Retrieval", "query": "Why are customers disputing unexpected cancellation processing fees?"},
    {"id": "Q19", "task": "Task 1", "category": "Billing", "difficulty": "Multi-Hop Reasoning", "query": "How do unnotified automatic subscription renewals lead to chargeback threats?"},
    {"id": "Q20", "task": "Task 1", "category": "Billing", "difficulty": "Simple Retrieval", "query": "Why do initial deposit charges create billing disputes during hotel check-in?"},
    {"id": "Q21", "task": "Task 1", "category": "Billing", "difficulty": "Ambiguous Query", "query": "What is wrong with my charge this month?"},
    {"id": "Q22", "task": "Task 1", "category": "Billing", "difficulty": "Simple Retrieval", "query": "Why are policyholders complaining about premium rate increases without prior notice?"},
    {"id": "Q23", "task": "Task 1", "category": "Billing", "difficulty": "Multi-Hop Reasoning", "query": "What sequence of billing system errors leads to double charges on credit cards?"},
    {"id": "Q24", "task": "Task 1", "category": "Billing", "difficulty": "Simple Retrieval", "query": "Why are callers confused about international transaction fees?"},

    # --- TASK 1: CHURN & CANCELLATIONS (8 Queries) ---
    {"id": "Q25", "task": "Task 1", "category": "Churn", "difficulty": "Simple Retrieval", "query": "Why are long-term subscribers stating an explicit intent to cancel?"},
    {"id": "Q26", "task": "Task 1", "category": "Churn", "difficulty": "Multi-Hop Reasoning", "query": "What sequence of agent policy refusals and repeated call attempts drives customers to cancel their account?"},
    {"id": "Q27", "task": "Task 1", "category": "Churn", "difficulty": "Simple Retrieval", "query": "Why do customers cancel their insurance policies following claim denials?"},
    {"id": "Q28", "task": "Task 1", "category": "Churn", "difficulty": "Multi-Hop Reasoning", "query": "How do hidden cancellation fees push frustrated callers to switch to competitors?"},
    {"id": "Q29", "task": "Task 1", "category": "Churn", "difficulty": "Ambiguous Query", "query": "Why are people leaving us?"},
    {"id": "Q30", "task": "Task 1", "category": "Churn", "difficulty": "Simple Retrieval", "query": "What service outages cause sudden spikes in account termination requests?"},
    {"id": "Q31", "task": "Task 1", "category": "Churn", "difficulty": "Multi-Hop Reasoning", "query": "How does poor agent empathy during bereavement calls trigger account cancellations?"},
    {"id": "Q32", "task": "Task 1", "category": "Churn", "difficulty": "Simple Retrieval", "query": "Why are customers canceling after mobile app UI redesigns?"},

    # --- TASK 1: REFUNDS & PRODUCT IMPROVEMENTS (8 Queries) ---
    {"id": "Q33", "task": "Task 1", "category": "Refunds", "difficulty": "Simple Retrieval", "query": "Why are customers demanding immediate cash refunds for delayed orders?"},
    {"id": "Q34", "task": "Task 1", "category": "Refunds", "difficulty": "Multi-Hop Reasoning", "query": "Why do delays in refund processing times trigger credit card dispute filings?"},
    {"id": "Q35", "task": "Task 1", "category": "Product Improvements", "difficulty": "Simple Retrieval", "query": "What complaints do callers mention regarding mobile application crashing during payment?"},
    {"id": "Q36", "task": "Task 1", "category": "Product Improvements", "difficulty": "Simple Retrieval", "query": "Why do customers report failure during hardware device pairing steps?"},
    {"id": "Q37", "task": "Task 1", "category": "Refunds", "difficulty": "Simple Retrieval", "query": "What refund policy restrictions generate caller hostility?"},
    {"id": "Q38", "task": "Task 1", "category": "Product Improvements", "difficulty": "Multi-Hop Reasoning", "query": "How do confusing navigation menus in customer web portals lead to support call volume?"},
    {"id": "Q39", "task": "Task 1", "category": "Refunds", "difficulty": "Ambiguous Query", "query": "Where is my money?"},
    {"id": "Q40", "task": "Task 1", "category": "Product Improvements", "difficulty": "Simple Retrieval", "query": "What feature requests are callers repeatedly asking for in the mobile app?"},

    # --- TASK 1: KNOWN FAILURE & OUT-OF-BOUNDS (2 Queries) ---
    {"id": "Q41", "task": "Task 1", "category": "Out-of-Bounds", "difficulty": "Known Failure Case", "query": "What are the exact internal employee salaries for tier 2 phone representatives?"},
    {"id": "Q42", "task": "Task 1", "category": "Unsupported Domain", "difficulty": "Known Failure Case", "query": "Why are users reporting issues with cryptocurrency wallet transfers?"},

    # --- TASK 2: MULTI-TURN CONVERSATIONAL SEQUENCES (3 Sequences / 10 Turns) ---
    {
        "id": "SEQ_01",
        "task": "Task 2",
        "category": "Escalation & Resolution Tracking",
        "difficulty": "Multi-Turn Sequence",
        "turns": [
            "Why are insurance domain customers threatening supervisor escalations?",
            "Did any agents offer fee waivers or policy exceptions to calm those callers?",
            "What specific policy rules prevented agents from granting exceptions?"
        ]
    },
    {
        "id": "SEQ_02",
        "task": "Task 2",
        "category": "Promotions & Churn Linkage",
        "difficulty": "Multi-Turn Sequence",
        "turns": [
            "What issues are callers reporting with promo code 'summer20'?",
            "Did customers threaten to cancel their service when the promo failed?",
            "How did reps handle those cancellation threats during the call?",
            "Were any retention discounts successfully applied?"
        ]
    },
    {
        "id": "SEQ_03",
        "task": "Task 2",
        "category": "Billing Fee Disputes",
        "difficulty": "Multi-Turn Sequence",
        "turns": [
            "Why are callers confused about prorated initial charges?",
            "How did agents explain the charge breakdown?",
            "Did the customers accept the explanation or ask for a manager?"
        ]
    }
]


def run_benchmark():
    print("Initializing Benchmark Suite Execution...")
    db_dir = os.path.abspath("data/index/chroma_db")
    transcripts_path = os.path.abspath("data/processed/final_transcripts.json")
    
    shared_retriever = HybridRetriever(transcripts_path=transcripts_path, db_dir=db_dir)
    agent = ConversationalAgent(transcripts_path=transcripts_path, db_dir=db_dir, llm_model="llama3.1", retriever=shared_retriever)
    
    output_records = []
    total_turns_counter = 0
    
    for item in BENCHMARK_SUITE:
        print(f"Executing [{item['id']}] - Task: {item['task']} | Difficulty: {item['difficulty']}")
        
        if item["task"] == "Task 1":
            start_t = time.time()
            res = run_causal_analysis(
                query=item["query"], 
                db_dir=db_dir, 
                top_k=4, 
                llm_model="llama3.1", 
                retriever=shared_retriever
            )
            elapsed = round(time.time() - start_t, 2)
            total_turns_counter += 1
            
            output_records.append({
                "query_id": item["id"],
                "task": item["task"],
                "category": item["category"],
                "difficulty": item["difficulty"],
                "input_query": item["query"],
                "latency_seconds": elapsed,
                "retrieved_chunks": res.get("retrieved_chunks", []),
                "abstained": res.get("abstained", False),
                "max_rerank_score": res.get("max_rerank_score", 0.0),
                "system_output": res["analysis"]
            })
            
        elif item["task"] == "Task 2":
            seq_outputs = []
            current_history = []
            
            for turn_idx, turn_query in enumerate(item["turns"], start=1):
                start_t = time.time()
                answer = agent.chat(turn_query, chat_history=current_history, top_k=4)
                standalone_q = agent.last_standalone_query
                elapsed = round(time.time() - start_t, 2)
                total_turns_counter += 1
                
                # Retrieve chunks saved during agent chat
                saved_agent_chunks = [
                    {
                        "chunk_id": c["chunk_id"],
                        "text": c["text"],
                        "transcript_id": c["metadata"].get("transcript_id", ""),
                        "start_turn": c["metadata"].get("start_turn", 0),
                        "end_turn": c["metadata"].get("end_turn", 0)
                    } for c in agent.last_retrieved_chunks
                ]
                
                seq_outputs.append({
                    "turn": turn_idx,
                    "user_query": turn_query,
                    "rewritten_standalone_query": standalone_q,
                    "latency_seconds": elapsed,
                    "retrieved_chunks": saved_agent_chunks,
                    "agent_response": answer
                })
                
                current_history.append({"role": "user", "content": turn_query})
                current_history.append({"role": "assistant", "content": answer})
                
            output_records.append({
                "sequence_id": item["id"],
                "task": item["task"],
                "category": item["category"],
                "difficulty": item["difficulty"],
                "dialogue_sequence": seq_outputs
            })

        gc.collect()

    os.makedirs("results", exist_ok=True)
    out_path = "results/query_dataset_outputs.json"
    
    generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "system_version": "Hybrid RAG v1.0",
                "total_evaluations": len(output_records),
                "total_queries_and_turns": total_turns_counter,
                "generation_timestamp": generation_time
            },
            "evaluations": output_records
        }, f, indent=2)
        
    print(f"Fast Benchmark Generation Complete! Output saved to: {out_path}")

if __name__ == "__main__":
    run_benchmark()