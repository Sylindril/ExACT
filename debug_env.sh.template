#!/bin/bash
# debug_env.sh - Configure environment for debugging Wikipedia tasks

# Python path
export PYTHONPATH=$(pwd)

# Dataset
export DATASET=visualwebarena

# ===== EDIT THESE FOR YOUR SETUP =====

# For debugging with REAL Wikipedia (not VWA Wikipedia)
export WIKIPEDIA="https://en.wikipedia.org"

# Your local VLM endpoint (change port if different)
export VLM_ENDPOINT="http://localhost:8000/v1"
export VLM_MODEL_NAME="your-model-name"  # e.g., "llava-v1.6-34b", "qwen2-vl-7b"

# ===== OPTIONAL: Other VWA environments (can be placeholders for Wikipedia-only tasks) =====
export REDDIT="http://localhost:9999"      # or your actual Reddit instance
export CLASSIFIEDS="http://localhost:9980" # or your actual Classifieds instance
export SHOPPING="http://localhost:7770"    # or your actual Shopping instance

# ===== VLM API Configuration =====
export PROVIDER="local_vlm"  # This should match a provider in configs/llms/providers.yaml
export AGENT_LLM_API_BASE="$VLM_ENDPOINT"
export AGENT_LLM_API_KEY="not-needed"  # Many local VLMs don't require a key

# ===== OPTIONAL: For embeddings (if your agent uses them) =====
# export OPENAI_API_KEY="sk-your-key-here"  # Uncomment if using OpenAI for embeddings
export EMBEDDING_MODEL_PROVIDER="openai"

# ===== Debug Settings =====
export DEBUG=True

# ===== GPU Selection =====
export CUDA_VISIBLE_DEVICES=0  # Change if you want to use a different GPU

echo "Environment configured:"
echo "  DATASET: $DATASET"
echo "  WIKIPEDIA: $WIKIPEDIA"
echo "  VLM_ENDPOINT: $VLM_ENDPOINT"
echo "  VLM_MODEL: $VLM_MODEL_NAME"
echo "  PROVIDER: $PROVIDER"
echo ""
echo "Run './debug_run.sh' to test a simple Wikipedia task"
