#!/bin/bash
# debug_run.sh - Run a simple Wikipedia task for debugging

# Load environment
if [ ! -f debug_env.sh ]; then
    echo "Error: debug_env.sh not found. Please create it first."
    exit 1
fi

source debug_env.sh

# ===== Model Configuration =====
model="$VLM_MODEL_NAME"  # From debug_env.sh
instruction_path="src/prompts/vwa/jsons/p_som_cot_id_actree_3s_final.json"
test_config_dir="configs/visualwebarena/test_wikipedia_debug"
test_idx="999"  # Our simple test task

# ===== Agent Configuration =====
agent="prompt"  # Simple ReACT-style agent (not MCTS for faster debugging)
max_steps=3     # Just a few steps for quick test
prompt_constructor_type="MCoTPolicyPConstructor"

# ===== Save Directory =====
SAVE_ROOT_DIR="data/debug_wikipedia"
mkdir -p $SAVE_ROOT_DIR

echo "========================================="
echo "Running Wikipedia Debug Task"
echo "========================================="
echo "Model: $model"
echo "Provider: $PROVIDER"
echo "Test task: $test_idx"
echo "Max steps: $max_steps"
echo "Results will be saved to: $SAVE_ROOT_DIR"
echo "========================================="
echo ""

# ===== Run Evaluation =====
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
python runners/eval/eval_vwa_agent.py \
--instruction_path $instruction_path \
--test_idx $test_idx \
--model $model \
--provider $PROVIDER \
--agent_type $agent \
--prompt_constructor_type $prompt_constructor_type \
--result_dir $SAVE_ROOT_DIR \
--test_config_base_dir $test_config_dir \
--repeating_action_failure_th 5 \
--viewport_height 2048 \
--max_obs_length 3840 \
--action_set_tag som \
--observation_type image_som \
--temperature 0.7 \
--top_p 0.9 \
--max_steps $max_steps

echo ""
echo "========================================="
echo "Done!"
echo "========================================="
echo "Results saved to: $SAVE_ROOT_DIR"
echo ""
echo "To view results:"
echo "  - HTML visualization: firefox $SAVE_ROOT_DIR/render_999.html"
echo "  - Logs: cat $SAVE_ROOT_DIR/log_files/*.log"
echo "  - Trajectory: ls $SAVE_ROOT_DIR/trajectories/"
echo ""
