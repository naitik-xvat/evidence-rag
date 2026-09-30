#!/bin/bash
set -e


echo " CONTACT CENTER CAUSAL RAG SYSTEM - RUNTIME INITIALIZER"


# 1. Health Checks for Ollama Host & Model
echo "1. Performing system health checks..."
if command -v curl >/dev/null 2>&1; then
    if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo " Error: Ollama is not running on http://localhost:11434."
        echo "Please start Ollama using 'ollama serve' and pull the model: 'ollama pull llama3.1'."
        exit 1
    else
        echo "Ollama service detected online."
    fi
fi

# Parse Flags
FORCE_REINDEX=false
EVAL_ONLY=false
UI_ONLY=false

for arg in "$@"; do
    case $arg in
        --force-reindex) FORCE_REINDEX=true ;;
        --eval-only) EVAL_ONLY=true ;;
        --ui-only) UI_ONLY=true ;;
    esac
done

if [ "$UI_ONLY" = true ]; then
    echo "Launching Streamlit UI directly..."
    streamlit run app.py
    exit 0
fi

# 2. Parse Transcripts
if [ -f "data/raw/conversations.json" ] && [ ! -f "data/processed/final_transcripts.json" ]; then
    echo "2. Parsing raw conversations dataset..."
    python src/data_processing/parse_transcripts.py --input data/raw/conversations.json --output data/processed/final_transcripts.json
elif [ -f "data/processed/final_transcripts.json" ]; then
    echo "2. Processed transcripts found. Skipping parsing step."
fi

# 3. Build Vector Database
if [ "$FORCE_REINDEX" = true ] || [ ! -d "data/index/chroma_db" ]; then
    echo "3. Indexing transcripts into ChromaDB vector store..."
    python src/retrieval/indexer.py --input data/processed/final_transcripts.json --db_dir data/index/chroma_db
else
    echo "3. Existing ChromaDB index found. Skipping vector indexing."
fi

# 4. Run Benchmark & Evaluation Metrics
if [ "$EVAL_ONLY" = true ]; then
    echo "4. Executing Benchmark Suite & Metrics Evaluation..."
    python -m evaluation.generate_dataset
    python -m evaluation.evaluate_metrics
    exit 0
fi


echo "✅ Setup complete! Launching Streamlit UI..."


streamlit run app.py