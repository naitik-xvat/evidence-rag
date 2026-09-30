import argparse
import json
import os
import torch
import chromadb
from sentence_transformers import SentenceTransformer

def build_vector_index(input_file: str, db_dir: str, collection_name: str = "conversations"):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Processed transcripts file not found at: {input_file}")

    print(f"Loading processed transcripts from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        transcripts = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device for embedding generation: {device.upper()}")

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    os.makedirs(db_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=db_dir)

    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    documents = []
    metadatas = []
    ids = []
    seen_ids = set()

    for item in transcripts:
        t_id = item["transcript_id"]
        domain = item.get("domain", "")
        intent = item.get("intent", "")
        reason = item.get("reason_for_call", "")

        for chunk in item.get("chunks", []):
            c_id = chunk["chunk_id"]
            if c_id in seen_ids:
                continue
            seen_ids.add(c_id)
            
            ids.append(c_id)
            documents.append(chunk["text"])
            metadatas.append({
                "transcript_id": t_id,
                "domain": domain,
                "intent": intent,
                "reason_for_call": reason,
                "start_turn": chunk["start_turn"],
                "end_turn": chunk["end_turn"]
            })

    total_chunks = len(documents)
    print(f"Indexing {total_chunks} unique chunks into ChromaDB...")

    batch_size = 1024
    for i in range(0, total_chunks, batch_size):
        end_i = min(i + batch_size, total_chunks)
        
        batch_docs = documents[i:end_i]
        batch_metas = metadatas[i:end_i]
        batch_ids = ids[i:end_i]

        batch_embeddings = embedding_model.encode(
            batch_docs, 
            batch_size=256, 
            show_progress_bar=False, 
            convert_to_numpy=True
        ).tolist()

        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_docs,
            metadatas=batch_metas
        )

        print(f"Indexed {end_i}/{total_chunks} chunks ({(end_i/total_chunks)*100:.1f}%)")

    print(f"\nSuccessfully indexed {collection.count()} chunks into '{db_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index transcript chunks into ChromaDB.")
    parser.add_argument("--input", type=str, required=True, help="Path to final_transcripts.json")
    parser.add_argument("--db_dir", type=str, default="data/index/chroma_db")
    parser.add_argument("--collection", type=str, default="conversations")

    args = parser.parse_args()
    build_vector_index(args.input, args.db_dir, args.collection)