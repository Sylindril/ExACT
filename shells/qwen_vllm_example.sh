#!/bin/bash
#
# Example script for running Visual Web Arena tasks with QWEN-vLLM
# This script demonstrates how to use a self-hosted QWEN model via vLLM
#
export PYTHONPATH=$(pwd)
export DATASET=visualwebarena

#==============================================================================
# Model Configuration
#==============================================================================
export PROVIDER="sglang"  # vLLM uses OpenAI-compatible API, same as sglang
export AGENT_LLM_API_BASE="http://localhost:9001/v1"
export AGENT_LLM_API_KEY="qwen"
export VALUE_FUNC_PROVIDER="sglang"
export VALUE_FUNC_API_BASE="http://localhost:9001/v1"
export RLM_PROVIDER="sglang"
export EMBEDDING_MODEL_PROVIDER="openai"  # For embeddings, still use OpenAI
export AZURE_TOKEN_PROVIDER_BASE=""
export AZURE_OPENAI_API_VERSION=""
EVAL_GPU_IDX=0

#==============================================================================
# vLLM Server Configuration
#==============================================================================
VLLM_MODEL_PATH="${QWEN_MODEL_PATH:-Qwen/Qwen2.5-VL-72B-Instruct}"
VLLM_NUM_GPUS="${QWEN_NUM_GPUS:-8}"
VLLM_PORT="${VLLM_PORT:-9001}"

#==============================================================================
# Model and Agent Configuration
#==============================================================================
model="qwen_vllm"          # Model name as served by vLLM
model_id="qwen_vllm"
rlm_model="qwen_vllm"
embedding_model="text-embedding-3-small"  # OpenAI embeddings for memory/retrieval
instruction_path="src/prompts/vwa/jsons/p_som_cot_id_actree_3s_final.json"
test_config_dir="configs/visualwebarena/test_classifieds_v2"

agent="rmcts_mad"  # Options: "prompt" (ReACT), "rmcts_mad" (R-MCTS+Debate), "sagent" (SearchAgent)

#==============================================================================
# Search Configuration (for MCTS-based agents)
#==============================================================================
max_depth=4                      # Max depth of search tree (4 = 5 step lookahead)
max_steps=5                      # Maximum steps per task
branching_factor=5               # Number of actions to explore per state
vf_budget=20                     # Value function evaluation budget
time_budget=2.5                  # Time budget per step in minutes (soft limit)

# Policy configuration
prompt_constructor_type=ReinforcedPolicyPConstructor
max_reflections_per_task=3      # Number of reflections for learning
reflection_threshold=0.5        # Threshold for triggering reflection
puct=1.0                        # PUCT exploration constant

# Value function configuration
v_func_method=ReinforcedDebateValueFunction  # Multi-agent debate for value estimation
value_max_reflections_per_task=1
value_reflection_threshold=0.5

#==============================================================================
# Task Selection
#==============================================================================
test_idx="10,11,70,71"  # Example task IDs from Classifieds

#==============================================================================
# Output Configuration
#==============================================================================
RUN_FILE=runners/eval/eval_vwa_ragent.py
SAVE_ROOT_DIR=data/${DATASET}/eval_results/qwen_vllm/example
echo "SAVEDIR=${SAVE_ROOT_DIR}"
mkdir -p $SAVE_ROOT_DIR
cp "$0" "${SAVE_ROOT_DIR}/run.sh"

export DEBUG=True

#==============================================================================
# Start vLLM Server (if not already running)
#==============================================================================
echo "================================================================"
echo "Checking vLLM server status..."
echo "================================================================"

# Check if vLLM server is already running
if lsof -Pi :${VLLM_PORT} -sTCP:LISTEN -t >/dev/null 2>&1 || pgrep -f "vllm serve" >/dev/null 2>&1; then
    echo "[INFO] vLLM server is already running on port ${VLLM_PORT}"
else
    echo "[INFO] Starting vLLM server..."
    echo "[INFO] Model: ${VLLM_MODEL_PATH}"
    echo "[INFO] GPUs: ${VLLM_NUM_GPUS}"
    echo "[INFO] Port: ${VLLM_PORT}"
    echo "[INFO] Logs will be saved to vllm_logs/"

    # Start vLLM server in background
    bash scripts/vllm/serve_qwen.sh "${VLLM_MODEL_PATH}" "${VLLM_NUM_GPUS}" "${VLLM_PORT}" &

    # Wait for server to be ready
    echo "[INFO] Waiting for vLLM server to be ready..."
    for i in {1..60}; do
        if curl --max-time 2 -s -H "Authorization: Bearer ${AGENT_LLM_API_KEY}" \
                -o /dev/null http://localhost:${VLLM_PORT}/v1/models; then
            echo "[INFO] vLLM server is ready!"
            break
        else
            echo "[WARN] Waiting for vLLM server... attempt $i/60"
            sleep 10
        fi

        if [ "$i" -eq 60 ]; then
            echo "[ERROR] vLLM server did not respond after 10 minutes, exiting."
            exit 1
        fi
    done
fi

echo "================================================================"
echo "Starting Visual Web Arena evaluation with QWEN-vLLM"
echo "================================================================"

#==============================================================================
# Run Evaluation
#==============================================================================
CUDA_VISIBLE_DEVICES=${EVAL_GPU_IDX} \
python $RUN_FILE \
--instruction_path $instruction_path \
--test_idx $test_idx \
--model $model \
--provider $PROVIDER \
--agent_type $agent \
--prompt_constructor_type $prompt_constructor_type \
--puct $puct \
--branching_factor $branching_factor \
--vf_budget $vf_budget \
--time_budget $time_budget \
--max_reflections_per_task $max_reflections_per_task \
--reflection_threshold $reflection_threshold \
--value_function $model \
--value_function_method $v_func_method \
--value_max_reflections_per_task $value_max_reflections_per_task \
--value_reflection_threshold $value_reflection_threshold \
--rlm_model $rlm_model \
--rlm_provider $PROVIDER \
--embedding_provider $EMBEDDING_MODEL_PROVIDER \
--embedding_model $embedding_model \
--db_path $SAVE_ROOT_DIR/db \
--result_dir $SAVE_ROOT_DIR \
--test_config_base_dir $test_config_dir \
--repeating_action_failure_th 5 \
--viewport_height 2048 \
--max_obs_length 3840 \
--action_set_tag som \
--observation_type image_som \
--top_p 0.95 \
--temperature 1.0 \
--max_steps $max_steps

#==============================================================================
# Cleanup and Post-processing
#==============================================================================
echo "================================================================"
echo "Repartitioning log files..."
echo "================================================================"
python runners/utils/repartition_log_files.py $SAVE_ROOT_DIR/log_files

echo "================================================================"
echo "Evaluation complete!"
echo "Results saved to: ${SAVE_ROOT_DIR}"
echo "================================================================"

#==============================================================================
# Optional: Kill vLLM server after evaluation
#==============================================================================
# Uncomment the following lines if you want to automatically stop the vLLM server
# echo "[INFO] Stopping vLLM server..."
# pkill -f "vllm serve"
