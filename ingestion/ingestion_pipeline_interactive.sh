#!/bin/bash
# 1.1. Set up environment variables
# Python path
export PYTHONPATH="/home/sathishbabu.ki/src:$PYTHONPATH"
# Huggingface cache (all the vllm models are stored here)
export HF_HOME="/scratch/sathishbabu.ki/vllm_models/vllm/.cache/huggingface"
# Marker model path
# this is also the default path for marker in the code you might need to change it in marker_prod.py later
export MARKER_MODEL_CACHE="/scratch/sathishbabu.ki/marker-models"
# Data folder paths
export BASE_DIR="/scratch/sathishbabu.ki/data_files2"
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

# 1.2. Function to monitor GPU usage
monitor_gpu_usage() {
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local log_file="${RAG_LOG_DIR}/gpu_usage_${timestamp}.log"

    # CSV Header
    echo "timestamp,memory_used_mb,memory_total_mb,memory_util_pct,gpu_util_pct" > "$log_file"

    echo "Monitoring GPU usage (CSV). Logging to: $log_file"

    while true; do
        local now=$(date +"%Y-%m-%d %H:%M:%S")

        # Query nvidia-smi for a machine-readable output
        local line=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.memory,utilization.gpu \
                                --format=csv,noheader,nounits)

        # line format example: "1024, 11178, 10, 25"
        echo "$now,$line" >> "$log_file"

        sleep 10
    done &
    GPU_MONITOR_PID=$!
}

# Start GPU monitoring
monitor_gpu_usage

# Ensure GPU monitoring stops when the script exits
trap "kill $GPU_MONITOR_PID" EXIT

# # 2. Activate marker environment
source activate /scratch/sathishbabu.ki/ingestion_marker

# # 3. Run ingestion pipeline
# # 3.1 Rotate pdf
python $WORK_DIR/ingestion/page_rotation.py --input_dir $INPUT_DIR --output_dir $ROTATED_DIR
# # 3.2 Extract md
python $WORK_DIR/ingestion/marker_extraction.py --input_dir $ROTATED_DIR --output_dir $EXTRACTED_MD_DIR

# # 4. Deactivate marker environment and activate vllm environment
conda deactivate
source /scratch/sathishbabu.ki/processing_vllm/bin/activate

# # 5. Run ingestion pipeline
# # 5.1 Clean md
python $WORK_DIR/ingestion/noise_classifer.py --input_dir $EXTRACTED_MD_DIR --output_dir $CLEANED_MD_DIR
# # 5.2 Summary generation
python $WORK_DIR/ingestion/summary_generator.py --input_dir $CLEANED_MD_DIR --output_dir $SUMMARY_MD_DIR
# # python $WORK_DIR/ingestion/summary_generator.py \
# #     --input_dir $EXTRACTED_MD_DIR \
# #     --output_dir $SUMMARY_MD_DIR \
# #     --max-context 32000 \
# #     --output-budget 3000 \
# #     --no-skip-existing
# # 5.3 BM25
python $WORK_DIR/ingestion/bm25_generator.py --input_dir $CLEANED_MD_DIR --output_dir $BM25_ARTIFACT_DIR
# # 5.4 Embedding
python $WORK_DIR/ingestion/embedding_generator.py --stage1_input_dir $CLEANED_MD_DIR --stage2_input_dir $CLEANED_MD_DIR --output_dir $EMBEDDING_DIR

# 6. Deactivate vllm environment
deactivate
