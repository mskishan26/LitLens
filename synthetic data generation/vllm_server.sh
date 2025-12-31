#!/bin/bash
# ============================================================================
# vLLM Server Starter Script
# ============================================================================
# This script starts vLLM as an OpenAI-compatible API server.
# Run this BEFORE running the RAG evaluation generator.
# ============================================================================

# Configuration - modify these as needed
# Use local path instead of Hugging Face name
MODEL_PATH="${MODEL_PATH:-/scratch/sathishbabu.ki/vllm_models/vllm/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"

echo "=============================================="
echo "Starting vLLM OpenAI-Compatible Server"
echo "=============================================="
echo "Model Path: $MODEL_PATH"
echo "Port: $PORT"
echo "GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "Tensor Parallel Size: $TENSOR_PARALLEL"
echo "=============================================="

# Set Hugging Face cache directory explicitly
export HF_HOME="/scratch/sathishbabu.ki/vllm_models/vllm/.cache/huggingface"
export HF_HUB_OFFLINE=1  # Work offline - don't try to connect to HF

source activate /scratch/sathishbabu.ki/eval_gen

# Check if vLLM is installed
if ! python -c "import vllm" 2>/dev/null; then
    echo "ERROR: vLLM is not installed. Install with:"
    echo "  pip install vllm"
    exit 1
fi

# Check if model directory exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model directory not found: $MODEL_PATH"
    echo "Looking for model in cache..."
    find "$HF_HOME" -name "*.json" -path "*Qwen*" 2>/dev/null | head -5
    exit 1
fi

# Start the server
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --trust-remote-code \
    --disable-log-requests