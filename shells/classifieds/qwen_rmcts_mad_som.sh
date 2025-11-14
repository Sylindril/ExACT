#!/bin/bash
export PYTHONPATH=$(pwd)
export DATASET=visualwebarena

## Define the model, result directory, and instruction path variables
export PROVIDER="sglang"  # vLLM uses OpenAI-compatible API
export AGENT_LLM_API_BASE="http://localhost:9001/v1"
export AGENT_LLM_API_KEY="qwen"
export VALUE_FUNC_PROVIDER="sglang"
export VALUE_FUNC_API_BASE="http://localhost:9001/v1"
export RLM_PROVIDER="sglang"
export EMBEDDING_MODEL_PROVIDER="openai"  # Use OpenAI for embeddings
export AZURE_TOKEN_PROVIDER_BASE=""
export AZURE_OPENAI_API_VERSION=""
EVAL_GPU_IDX=0

model="qwen_vllm"
model_id="qwen_vllm"
rlm_model="qwen_vllm"
embedding_model="text-embedding-3-small"
instruction_path="src/prompts/vwa/jsons/p_som_cot_id_actree_3s_final.json"
test_config_dir="configs/visualwebarena/test_classifieds_v2"

agent="rmcts_mad"

##### start of search config
max_depth=4  # max_depth=4 means 5 step lookahead
max_steps=5
branching_factor=5  # default 5
vf_budget=20        # default 20
time_budget=5.0     # 5.0 min per step (soft maximum), will override vf_budget if > 0.0

# policy config
prompt_constructor_type=ReinforcedPolicyPConstructor
max_reflections_per_task=3
reflection_threshold=0.5
puct=1.0

# vfunc config
v_func_method=ReinforcedDebateValueFunction
value_max_reflections_per_task=1
value_reflection_threshold=0.5

##### end of search config

test_idx="[[[test_idx]]]"  # replaced by runners/eval/eval_vwa_parallel.py

RUN_FILE=runners/eval/eval_vwa_ragent.py
SAVE_ROOT_DIR="[[[SAVE_ROOT_DIR]]]"
echo "SAVEDIR=${SAVE_ROOT_DIR}"
mkdir -p $SAVE_ROOT_DIR
cp "$0" "${SAVE_ROOT_DIR}/run.sh"

export DEBUG=True
####### start eval
CUDA_VISIBLE_DEVICES=${EVAL_GPU_IDX} \
python $RUN_FILE \
--instruction_path $instruction_path \
--test_idx $test_idx \
--model $model \
--provider $PROVIDER \
--agent_type $agent \
--prompt_constructor_type $prompt_constructor_type \
--puct $puct \
--branching_factor $branching_factor --vf_budget $vf_budget --time_budget $time_budget \
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
--repeating_action_failure_th 5 --viewport_height 2048 --max_obs_length 3840 \
--action_set_tag som  --observation_type image_som \
--top_p 0.95 --temperature 1.0 --max_steps $max_steps

##### cleanups
python runners/utils/repartition_log_files.py $SAVE_ROOT_DIR/log_files
