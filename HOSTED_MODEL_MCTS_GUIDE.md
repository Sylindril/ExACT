# Using Your Hosted Model for MCTS in the VWA Branch

This guide explains how to configure the ExACT framework to use **only your hosted model** for the entire MCTS (Monte Carlo Tree Search) process in the VisualWebArena (VWA) branch.

## Table of Contents
1. [Overview](#overview)
2. [Understanding Model Usage in MCTS](#understanding-model-usage-in-mcts)
3. [Step-by-Step Configuration](#step-by-step-configuration)
4. [Running MCTS with Your Hosted Model](#running-mcts-with-your-hosted-model)
5. [Troubleshooting](#troubleshooting)
6. [Advanced Configuration](#advanced-configuration)

---

## Overview

The MCTS agent in ExACT uses models for two main purposes:
1. **Policy Model**: Generates possible actions at each state (action selection)
2. **Value Function Model**: Evaluates how promising a state is (state evaluation)

By default, these use OpenAI's GPT-4o, but you can configure them to use your own hosted model instead.

---

## Understanding Model Usage in MCTS

### Where Models Are Called

The MCTS process makes model calls in these key locations:

1. **Action Generation** (`src/agent/mcts_agent.py:273-416`)
   - Method: `_gen_next_actions()`
   - Purpose: Generate multiple candidate actions from the current state
   - Model parameter: `--model` and `--provider`
   - Default: Uses GPT-4o via OpenAI API

2. **State Evaluation** (`src/agent/mcts_agent.py:477-522`)
   - Method: `_simulation()`
   - Purpose: Evaluate how good/promising a state is (returns value 0.0 to 1.0)
   - Model parameter: `--value_function` and `--provider`
   - Default: Uses GPT-4o-mini via OpenAI API

### API Calls Flow

```
MCTS Iteration
├── Selection: Navigate tree using UCT formula
├── Expansion: Generate new actions
│   └── _gen_next_actions() → calls POLICY MODEL (--model)
├── Simulation: Evaluate new state
│   └── _simulation() → calls VALUE FUNCTION MODEL (--value_function)
└── Backpropagation: Update Q-values
```

---

## Step-by-Step Configuration

### Step 1: Set Up Your Hosted Model

First, ensure your model is hosted and accessible via an OpenAI-compatible API endpoint. You need:
- **API Base URL**: e.g., `http://localhost:8000/v1` or `https://your-server.com/v1`
- **API Key**: Your authentication key (can be "EMPTY" for local servers)
- **Model Name**: The identifier for your model (e.g., `Qwen/Qwen2-VL-72B-Instruct`)

**Example hosting options:**
- SGLang: `python -m sglang.launch_server --model-path your-model --port 8000`
- vLLM: `python -m vllm.entrypoints.openai.api_server --model your-model --port 8000`
- LMDeploy: `lmdeploy serve api_server your-model --server-port 8000`

### Step 2: Configure Provider Settings

Edit `configs/llms/providers.yaml` to add or update your hosted model configuration:

```yaml
# configs/llms/providers.yaml

openai:
    provider: openai
    llm_api_base: https://api.openai.com/v1
    llm_api_version: ''

sglang:
    provider: sglang
    llm_api_base: http://localhost:8000/v1  # ← Change this to your endpoint
    llm_api_version: ''

azure:
    provider: azure
    llm_api_base: http://xxx
    llm_api_version: 'xxx'

# Add your custom provider (optional)
my_hosted_model:
    provider: sglang  # Use 'sglang' for OpenAI-compatible APIs
    llm_api_base: http://your-server:port/v1
    llm_api_version: ''
```

**Key Points:**
- Use `provider: sglang` for any OpenAI-compatible API (not just SGLang)
- The `llm_api_base` should be the full URL including `/v1`
- Leave `llm_api_version` as empty string for most providers

### Step 3: Set Environment Variables

Create or update your API key file (e.g., `.keys`):

```bash
# .keys file
export OPENAI_API_KEY="your-fallback-key"  # Only needed for embeddings
export OPENAI_ORGANIZATION="your-org"       # Optional

# For your hosted model - these override the defaults
export AGENT_LLM_API_BASE="http://localhost:8000/v1"
export AGENT_LLM_API_KEY="EMPTY"  # or your actual API key
export VALUE_FUNC_PROVIDER="sglang"
export VALUE_FUNC_API_BASE="http://localhost:8000/v1"
```

**Important Environment Variables:**
- `AGENT_LLM_API_BASE`: API endpoint for the policy model
- `AGENT_LLM_API_KEY`: Authentication key for policy model
- `VALUE_FUNC_PROVIDER`: Provider for value function (same as `PROVIDER`)
- `VALUE_FUNC_API_BASE`: API endpoint for value function
- `PROVIDER`: Main provider name (e.g., "sglang", "openai")

### Step 4: Update Your Shell Script

Modify your evaluation script (e.g., `shells/classifieds/rmcts_mad_som.sh`):

```bash
#!/bin/bash
export PYTHONPATH=$(pwd)
source /path/to/your/.keys  # ← Update this path

# ... (environment setup for VWA websites) ...

## Model Configuration
# Instead of: model="gpt-4o"
# Use your hosted model name:
model="Qwen/Qwen2-VL-72B-Instruct"  # ← Your model name
model_id="my-model"
rlm_model="Qwen/Qwen2-VL-72B-Instruct"  # Same as model

# For embeddings, you'll likely still need OpenAI
# (unless you also host an embedding model)
embedding_model="text-embedding-3-small"

# ... (rest of configuration) ...

# CRITICAL: Update these lines in the python command
python $RUN_FILE \
--model $model \
--provider sglang \  # ← Change from "openai" to "sglang"
--value_function $model \  # ← Use same model for value function
--rlm_provider sglang \  # ← Same provider
--embedding_provider openai \  # ← Keep OpenAI for embeddings or use your embedding provider
# ... (rest of parameters)
```

---

## Running MCTS with Your Hosted Model

### Quick Start Example

```bash
# 1. Start your model server
python -m sglang.launch_server \
    --model-path Qwen/Qwen2-VL-72B-Instruct \
    --port 8000 \
    --tp-size 4  # Adjust based on your GPUs

# 2. Update providers.yaml
# Edit configs/llms/providers.yaml as shown above

# 3. Source your environment
source .keys
export DATASET=visualwebarena

# 4. Run a test task
cd /path/to/ExACT
bash shells/example.sh
```

### Running Parallel Evaluation

For large-scale evaluation on Classifieds (234 tasks):

```bash
python runners/eval/eval_vwa_parallel.py \
--env_name classifieds \
--save_dir data/results/my_model_classifieds \
--eval_script shells/classifieds/rmcts_mad_som.sh \
--run_mode greedy \
--start_idx 0 \
--end_idx 234 \
--num_parallel 2 \
--main_api_providers sglang,sglang \  # ← Use your provider
--num_task_per_script 2 \
--num_task_per_reset 8
```

**Key Parameters:**
- `--main_api_providers`: Comma-separated list of providers (one per parallel worker)
- `--num_parallel`: Number of parallel evaluation processes
- `--eval_script`: Path to your updated shell script

---

## Troubleshooting

### Issue: "Model not found" Error

**Problem:** Your hosted model name doesn't match what the server expects.

**Solution:**
```bash
# Test your API endpoint
curl http://localhost:8000/v1/models

# Use the exact model name from the response
# Update the 'model' variable in your shell script
```

### Issue: "Connection refused" Error

**Problem:** The API endpoint is not accessible.

**Solution:**
1. Check if your model server is running: `ps aux | grep sglang`
2. Verify the port: `netstat -tuln | grep 8000`
3. Test connectivity: `curl http://localhost:8000/v1/models`
4. Check firewall settings if using remote server

### Issue: Vision Model Not Receiving Images

**Problem:** Your hosted VLM isn't processing images correctly.

**Solution:**
1. Ensure your model supports vision (check `is_vlm()` in `src/llms/utils.py:27-34`)
2. Add your model to the VLM check:
   ```python
   # src/llms/utils.py
   def is_vlm(lm_config: lm_config.LMConfig):
       if ("gemini" in lm_config.model
           or ("gpt-4" in lm_config.model and "vision" in lm_config.model)
           or "gpt-4o" in lm_config.model
           or "Qwen2-VL" in lm_config.model):  # ← Add your model pattern
           return True
       # ...
   ```

### Issue: Value Function Returns 0.0

**Problem:** State evaluation always returns 0.0.

**Solution:**
1. Check logs for errors in `_simulation()` method
2. Verify your model supports the value function prompt format
3. Test value function independently:
   ```bash
   # Add debug logging in src/agent/mcts_agent.py:515
   logger.error(f"Value function call failed: {e}", exc_info=True)
   ```

### Issue: "Unsupported value function model" Error

**Problem:** Your model name triggers the `NotImplementedError` at line 514.

**Solution:**
Update `src/agent/mcts_agent.py`:
```python
# Around line 502
elif self.value_function_model in ["gpt-4o", "gpt-4o-mini", "OpenGVLab/InternVL2-Llama3-76B", "Qwen/Qwen2-VL-72B-Instruct"]:
    # Add your model name here ↑
    v = self.value_function.evaluate_success(
        screenshots=last_screenshots,
        actions=copy.deepcopy(state.action_trajectory_str),
        current_url=state.env.page.url,
        last_reasoning="",
        intent=intent,
        models=[v_model],
        init_screenshot=init_screenshot,
        intent_images=images if len(images) > 0 else None
    )
```

---

## Advanced Configuration

### Using Different Models for Policy and Value Function

You may want to use a larger model for the value function and a smaller one for policy:

```bash
# In your shell script
policy_model="Qwen/Qwen2-VL-7B-Instruct"
value_model="Qwen/Qwen2-VL-72B-Instruct"

python $RUN_FILE \
--model $policy_model \
--value_function $value_model \
--provider sglang \
# ...
```

Then set up two different endpoints:
```bash
# .keys
export AGENT_LLM_API_BASE="http://localhost:8000/v1"  # Smaller model
export VALUE_FUNC_API_BASE="http://localhost:8001/v1"  # Larger model
```

### Tuning MCTS Parameters for Your Model

Different models may require different MCTS hyperparameters:

```bash
# In your shell script
branching_factor=3  # Reduce if your model is slower (default: 5)
vf_budget=10        # Reduce for faster models (default: 20)
time_budget=3.0     # Minutes per step (default: 5.0)
max_steps=10        # Increase if your model is more capable
puct=1.5            # Increase for more exploration (default: 1.0)
```

**Recommendations:**
- **Smaller models**: Reduce `branching_factor` to 3, increase `vf_budget` to 30
- **Larger models**: Keep defaults or increase `branching_factor` to 7
- **Faster models**: Increase `vf_budget` to 50 for more thorough search

### Monitoring API Calls

Track model usage and costs:

```bash
# Enable debug logging
export DEBUG=True

# Logs will be in: $SAVE_ROOT_DIR/log_files/
# Search for API call patterns:
grep "LLM Top actions" $SAVE_ROOT_DIR/log_files/*.log
grep "Simulating state value" $SAVE_ROOT_DIR/log_files/*.log
```

### Custom Prompt Engineering

If your model requires different prompting:

1. Create custom prompt constructor in `src/prompts/vwa/`
2. Update `instruction_path` in your shell script:
   ```bash
   instruction_path="src/prompts/vwa/jsons/my_custom_prompt.json"
   ```

---

## Summary Checklist

Before running MCTS with your hosted model:

- [ ] Your model server is running and accessible
- [ ] `configs/llms/providers.yaml` has correct endpoint URL
- [ ] Environment variables are set (via `.keys` file)
- [ ] Shell script updated with:
  - [ ] Correct model name in `model=` variable
  - [ ] Correct provider in `--provider` parameter
  - [ ] Same model/provider for `--value_function`
  - [ ] Same provider for `--rlm_provider`
- [ ] Your model is added to `is_vlm()` check (if it's a vision model)
- [ ] Your model is added to value function model check in `mcts_agent.py`
- [ ] Test run on single task works: `test_idx="10"`

---

## Example: Complete Configuration for Qwen2-VL

Here's a complete example using Qwen2-VL-72B-Instruct:

### 1. Start Server
```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2-VL-72B-Instruct \
    --port 8000 \
    --tp-size 4 \
    --chat-template qwen2-vl
```

### 2. Update `configs/llms/providers.yaml`
```yaml
sglang:
    provider: sglang
    llm_api_base: http://localhost:8000/v1
    llm_api_version: ''
```

### 3. Create `.keys`
```bash
# .keys
export OPENAI_API_KEY="sk-xxx"  # For embeddings
export AGENT_LLM_API_BASE="http://localhost:8000/v1"
export AGENT_LLM_API_KEY="EMPTY"
export VALUE_FUNC_PROVIDER="sglang"
export VALUE_FUNC_API_BASE="http://localhost:8000/v1"
export PROVIDER="sglang"
```

### 4. Update Shell Script
```bash
# shells/classifieds/my_qwen_rmcts.sh
#!/bin/bash
export PYTHONPATH=$(pwd)
source .keys
export DATASET=visualwebarena

# ... VWA environment variables ...

model="Qwen/Qwen2-VL-72B-Instruct"
model_id="qwen2-vl-72b"
rlm_model="Qwen/Qwen2-VL-72B-Instruct"
embedding_model="text-embedding-3-small"

instruction_path="src/prompts/vwa/jsons/p_som_cot_id_actree_3s_final.json"
test_config_dir="configs/visualwebarena/test_classifieds_v2"

agent="rmcts_mad"
max_steps=5
branching_factor=5
vf_budget=20
time_budget=5.0

test_idx="10,11,70,71"
SAVE_ROOT_DIR="data/results/qwen2vl_classifieds_test"

python runners/eval/eval_vwa_ragent.py \
--instruction_path $instruction_path \
--test_idx $test_idx \
--model $model \
--provider sglang \
--agent_type $agent \
--value_function $model \
--rlm_model $rlm_model \
--rlm_provider sglang \
--embedding_provider openai \
--embedding_model $embedding_model \
--result_dir $SAVE_ROOT_DIR \
--test_config_base_dir $test_config_dir \
--branching_factor $branching_factor \
--vf_budget $vf_budget \
--time_budget $time_budget \
--action_set_tag som \
--observation_type image_som \
--max_steps $max_steps
```

### 5. Update Code (if needed)
```python
# src/llms/utils.py - Add to is_vlm()
def is_vlm(lm_config: lm_config.LMConfig):
    if ("gemini" in lm_config.model
        or ("gpt-4" in lm_config.model and "vision" in lm_config.model)
        or "gpt-4o" in lm_config.model
        or "Qwen2-VL" in lm_config.model):  # ← Add this
        return True
    # ...

# src/agent/mcts_agent.py - Add to value function check
elif self.value_function_model in ["gpt-4o", "gpt-4o-mini", "OpenGVLab/InternVL2-Llama3-76B", "Qwen/Qwen2-VL-72B-Instruct"]:
    # ↑ Add your model here
```

### 6. Run
```bash
bash shells/classifieds/my_qwen_rmcts.sh
```

---

## Additional Resources

- **Main README**: See root README.md for VWA setup instructions
- **MCTS Agent Source**: `src/agent/mcts_agent.py`
- **LLM Utils**: `src/llms/utils.py` - for API call implementation
- **Provider Config**: `configs/llms/providers.yaml` and `configs/llms/providers.py`
- **Example Scripts**: `shells/*/rmcts_mad_som.sh`

---

## Questions?

Common scenarios:

**Q: Can I use a text-only model?**
A: Yes, but you need to use text-based observation mode. Change:
- `--observation_type image_som` → `--observation_type accessibility_tree`
- `--action_set_tag som` → `--action_set_tag id_accessibility_tree`
- Use shell scripts ending in `_text.sh` instead of `_som.sh`

**Q: Do I need to keep OpenAI API for anything?**
A: Typically yes, for the embedding model used in reflective learning. If you host your own embedding model, you can eliminate OpenAI dependency completely.

**Q: Can I use multiple hosted models?**
A: Yes! Just set different endpoints for `AGENT_LLM_API_BASE` and `VALUE_FUNC_API_BASE`, and use different model names for `--model` and `--value_function`.

**Q: What's the minimum model size needed?**
A: For decent performance, we recommend at least 7B parameters for vision-language models. Smaller models may struggle with the complex reasoning required for web navigation.

---

**Last Updated**: 2025-11-16
**ExACT Version**: Compatible with vwa branch
**Maintained by**: ExACT Project Contributors
