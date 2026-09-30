import argparse
import json
import os

def parse_transcripts(input_path: str, output_path: str, window_size: int = 4, overlap: int = 1):
    """Parses conversations.json, adds turn IDs, retains metadata, and generates clean sliding chunks."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at: {input_path}")

    print(f"Loading raw dataset from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    processed_transcripts = []

    for item in raw_data:
        transcript_id = item.get("transcript_id", "unknown_id")
        time_of_interaction = item.get("time_of_interaction", "")
        domain = item.get("domain", "")
        intent = item.get("intent", "")
        reason_for_call = item.get("reason_for_call", "")

        raw_turns = item.get("conversation", [])
        cleaned_turns = []

        for idx, turn in enumerate(raw_turns, start=1):
            cleaned_turns.append({
                "turn_id": idx,
                "speaker": turn.get("speaker", "Unknown"),
                "text": turn.get("text", "").strip()
            })

        chunks = []
        step = max(1, window_size - overlap)
        
        for i in range(0, len(cleaned_turns), step):
            chunk_turns = cleaned_turns[i:i + window_size]
            if not chunk_turns:
                continue
            
            # Avoid trailing single-turn duplicate windows if already covered
            if len(chunk_turns) == 1 and len(cleaned_turns) > window_size and i > 0:
                continue

            chunk_text = "\n".join([f"{t['speaker']}: {t['text']}" for t in chunk_turns])
            start_turn = chunk_turns[0]["turn_id"]
            end_turn = chunk_turns[-1]["turn_id"]

            chunks.append({
                "chunk_id": f"{transcript_id}_turn_{start_turn}_to_{end_turn}",
                "start_turn": start_turn,
                "end_turn": end_turn,
                "text": chunk_text
            })

        processed_transcripts.append({
            "transcript_id": transcript_id,
            "time_of_interaction": time_of_interaction,
            "domain": domain,
            "intent": intent,
            "reason_for_call": reason_for_call,
            "total_turns": len(cleaned_turns),
            "turns": cleaned_turns,
            "chunks": chunks
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_transcripts, f, indent=2)

    print(f"Successfully processed {len(processed_transcripts)} transcripts to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse raw call transcripts into clean JSON.")
    parser.add_argument("--input", type=str, required=True, help="Path to raw conversations.json")
    parser.add_argument("--output", type=str, required=True, help="Path to output final_transcripts.json")
    parser.add_argument("--window_size", type=int, default=4, help="Turns per chunk")
    parser.add_argument("--overlap", type=int, default=1, help="Turn overlap")

    args = parser.parse_args()
    parse_transcripts(args.input, args.output, args.window_size, args.overlap)