#!/bin/bash
# 1.1. Set up environment variables
# Python path
export PYTHONPATH="/home/sathishbabu.ki/src:$PYTHONPATH"

# Huggingface cache (all the vllm models are stored here)
export HF_HOME="/scratch/sathishbabu.ki/vllm_models/vllm/.cache/huggingface"

# Marker model path (for PDF processing)
export MARKER_MODEL_CACHE="/scratch/sathishbabu.ki/marker-models"

# Data folder paths
export BASE_DIR="/scratch/sathishbabu.ki/data_files"
export INPUT_DIR="$BASE_DIR/input_pdf"
export ROTATED_DIR="$BASE_DIR/rotated_pdf"
export EXTRACTED_MD_DIR="$BASE_DIR/extracted_md"
export SUMMARY_MD_DIR="$BASE_DIR/summary_md"
export CLEANED_MD_DIR="$BASE_DIR/cleaned_md"
export BM25_ARTIFACT_DIR="$BASE_DIR/bm25_artifacts"
export EMBEDDING_DIR="$BASE_DIR/embeddings"
export RAG_LOG_DIR="$BASE_DIR/logs"

# Work directory
export WORK_DIR="/home/sathishbabu.ki/src"

# CUDA optimization
export PYTORCH_ALLOC_CONF='expandable_segments:True'

monitor_gpu_usage() {
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local log_file="${RAG_LOG_DIR}/gpu_usage_${timestamp}.log"

    # Create log directory if it doesn't exist
    mkdir -p "$RAG_LOG_DIR"

    # CSV Header
    echo "timestamp,memory_used_mb,memory_total_mb,memory_util_pct,gpu_util_pct" > "$log_file"

    echo "Monitoring GPU usage (CSV). Logging to: $log_file"

    while true; do
        local now=$(date +"%Y-%m-%d %H:%M:%S")

        # Query nvidia-smi for machine-readable output
        local line=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.memory,utilization.gpu \
                                --format=csv,noheader,nounits)

        echo "$now,$line" >> "$log_file"
        sleep 10
    done &
    GPU_MONITOR_PID=$!
}

# Start GPU monitoring
monitor_gpu_usage

# Ensure GPU monitoring stops when the script exits
trap "kill $GPU_MONITOR_PID 2>/dev/null" EXIT

# Activate marker environment
# source activate /scratch/sathishbabu.ki/capstone_marker

# # 2.1 Rotate PDFs
# echo "[$(date +"%Y-%m-%d %H:%M:%S")] Rotating PDFs..."
# python "$WORK_DIR/ingestion/page_rotation.py" \
#     --input_dir "$INPUT_DIR" \
#     --output_dir "$ROTATED_DIR"

# # 2.2 Extract markdown from PDFs
# echo "[$(date +"%Y-%m-%d %H:%M:%S")] Extracting markdown..."
# python "$WORK_DIR/ingestion/marker_extraction.py" \
#     --input_dir "$ROTATED_DIR" \
#     --output_dir "$EXTRACTED_MD_DIR"

# # Deactivate marker environment
# conda deactivate

# echo "[$(date +"%Y-%m-%d %H:%M:%S")] PDF processing completed."

# Activate UV environment
source /scratch/sathishbabu.ki/capstone_llm/bin/activate

# Verify environment
echo "[$(date +"%Y-%m-%d %H:%M:%S")] Environment check:"
python -c "import sys; print(f'Python: {sys.executable}')"
python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')" 2>/dev/null || echo "PyTorch check failed"
python -c "import chromadb; print(f'ChromaDB: {chromadb.__version__}')" 2>/dev/null || echo "ChromaDB check failed"
python -c "import vllm; print(f'vLLM: {vllm.__version__}')" 2>/dev/null || echo "vLLM check failed"

# # 3.1 Clean markdown
# echo "[$(date +"%Y-%m-%d %H:%M:%S")] Cleaning markdown..."
# python "$WORK_DIR/ingestion/noise_classifer.py" \
#     --input_dir "$EXTRACTED_MD_DIR" \
#     --output_dir "$CLEANED_MD_DIR"

# # 3.2 Generate summaries
# echo "[$(date +"%Y-%m-%d %H:%M:%S")] Generating summaries..."
# python "$WORK_DIR/ingestion/summary_generator.py" \
#     --input_dir "$CLEANED_MD_DIR" \
#     --output_dir "$SUMMARY_MD_DIR"

# Alternative with custom parameters:
# python "$WORK_DIR/ingestion/summary_generator.py" \
#     --input_dir "$EXTRACTED_MD_DIR" \
#     --output_dir "$SUMMARY_MD_DIR" \
#     --max-context 32000 \
#     --output-budget 3000 \
#     --no-skip-existing

# 3.3 Generate BM25 index
echo "[$(date +"%Y-%m-%d %H:%M:%S")] Generating BM25 index..."
python "$WORK_DIR/ingestion/bm25_generator.py" \
    --input_dir "$CLEANED_MD_DIR" \
    --output_dir "$BM25_ARTIFACT_DIR"

echo "[$(date +"%Y-%m-%d %H:%M:%S")] Generating embeddings..."
python "$WORK_DIR/ingestion/embedding_generator.py" \
    --stage1_input_dir "$CLEANED_MD_DIR" \
    --stage2_input_dir "$CLEANED_MD_DIR" \
    --output_dir "$EMBEDDING_DIR"

# Deactivate UV environment
deactivate

echo "[$(date +"%Y-%m-%d %H:%M:%S")] All stages completed successfully."
echo ""
echo "Output locations:"
echo "  - Rotated PDFs:    $ROTATED_DIR"
echo "  - Extracted MD:    $EXTRACTED_MD_DIR"
echo "  - Cleaned MD:      $CLEANED_MD_DIR"
echo "  - Summaries:       $SUMMARY_MD_DIR"
echo "  - BM25 Index:      $BM25_ARTIFACT_DIR"
echo "  - Embeddings (ChromaDB): $EMBEDDING_DIR"
echo "  - Logs:            $RAG_LOG_DIR"
echo ""
echo "To query the embeddings, use embedding_search.py with ChromaDB:"
echo "  python ingestion/embedding_search.py --db_path $EMBEDDING_DIR"