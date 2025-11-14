# QWEN-vLLM Integration Guide for Visual Web Arena

This guide explains how to run ExACT (R-MCTS agents) on the Visual Web Arena dataset using a self-hosted QWEN Vision Language Model (VLM) served via vLLM on your HPC cluster.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Running the vLLM Server](#running-the-vllm-server)
6. [Running VWA Tasks](#running-vwa-tasks)
7. [Advanced Configuration](#advanced-configuration)
8. [Troubleshooting](#troubleshooting)

---

## Overview

This integration allows you to:
- Host a QWEN Vision Language Model using vLLM on your HPC cluster
- Use the QWEN VLM as the backbone for ExACT agents on Visual Web Arena tasks
- Leverage multi-GPU parallelism with tensor parallelism
- Optimize memory usage for large models (e.g., Qwen2.5-VL-72B-Instruct)

The setup consists of:
1. **vLLM Server**: Serves the QWEN model with OpenAI-compatible API
2. **ExACT Framework**: Runs R-MCTS agents for Visual Web Arena
3. **Model Wrapper**: Bridges ExACT's agent code with the vLLM server

---

## Prerequisites

### System Requirements
- HPC cluster with NVIDIA GPUs (CUDA-enabled)
- For 72B models: At least 4-8 GPUs (A100/H100 recommended)
- For smaller models (7B/14B): 1-2 GPUs
- Sufficient shared memory (or configured NCCL to avoid shared memory issues)
- Network connectivity between compute nodes (if distributed)

### Software Requirements
- Python 3.9+
- CUDA 11.8+ or 12.1+
- vLLM (compatible with your CUDA version)
- PyTorch 2.0+
- OpenAI Python SDK
- Visual Web Arena environment (Docker or AWS setup)

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/microsoft/ExACT.git
cd ExACT
git checkout claude/qwen-vllm-vwa-integration-01T1AbRgTQdkRHpYJSJi37Zr
```

### Step 2: Install Dependencies

```bash
# Create a conda/virtual environment (recommended)
conda create -n exact-vwa python=3.10
conda activate exact-vwa

# Install ExACT dependencies
pip install -r requirements.txt  # Create this if needed

# Install vLLM (adjust for your CUDA version)
pip install vllm

# Install OpenAI SDK
pip install openai

# Install other required packages
pip install Pillow wandb  # wandb is optional but recommended for logging
```

### Step 3: Download QWEN Model

Download your desired QWEN model from Hugging Face:

```bash
# For example, Qwen2.5-VL-72B-Instruct
export HF_TOKEN="your_huggingface_token"
huggingface-cli login --token $HF_TOKEN

# The model will be auto-downloaded when you start vLLM
# Or pre-download with:
huggingface-cli download Qwen/Qwen2.5-VL-72B-Instruct
```

### Step 4: Set Up Visual Web Arena

Follow the [Visual Web Arena setup instructions](https://github.com/web-arena-x/visualwebarena) to:
1. Deploy the VWA websites (Classifieds, Shopping, Reddit, etc.) using Docker or AWS
2. Note down the URLs and ports for each website

---

## Configuration

### Step 1: Environment Variables

Create a file `~/.exact_env` with your configuration:

```bash
# VLM Model Configuration
export QWEN_MODEL_PATH="/path/to/Qwen/Qwen2.5-VL-72B-Instruct"
export VLLM_PORT=9001
export VLLM_API_KEY="qwen"  # Can be any string, used for authentication

# Visual Web Arena URLs
export DATASET="visualwebarena"
export CLASSIFIEDS="<your_classifieds_domain>:9980"
export CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"
export SHOPPING="<your_shopping_site_domain>:7770"
export REDDIT="<your_reddit_domain>:9999"
export WIKIPEDIA="<your_wikipedia_domain>:8888"
export SHOPPING_ADMIN="<your_e_commerce_cms_domain>:7780/admin"
export GITLAB="<your_gitlab_domain>:8023"
export MAP="<your_map_domain>:3000"
export HOMEPAGE="<your_homepage_domain>:4399"

# Optional: Hugging Face Token
export HF_TOKEN="your_hf_token"
```

Load the environment:
```bash
source ~/.exact_env
```

### Step 2: Update Provider Configuration

The vLLM provider is already configured in `configs/llms/providers.yaml`:

```yaml
vllm:
    provider: sglang  # Uses same OpenAI-compatible interface
    llm_api_base: http://localhost:9001/v1
    llm_api_version: ''
```

**Note**: We use `provider: sglang` because the codebase already supports sglang which uses OpenAI-compatible APIs (same as vLLM).

### Step 3: Generate Task Configs

Generate the task configurations for Visual Web Arena:

```bash
export DATASET=visualwebarena
python runners/utils/generate_test_configs.py
```

This creates task configs in `configs/visualwebarena/`.

---

## Running the vLLM Server

### Using the Provided Script

We provide `scripts/vllm/serve_qwen.sh` which handles:
- GPU configuration
- NCCL environment variables (for multi-GPU)
- Shared memory optimization
- vLLM server startup with proper parameters

#### Script Usage

```bash
bash scripts/vllm/serve_qwen.sh <MODEL_PATH> <NUM_GPUS> <PORT>
```

**Example for 72B model on 8 GPUs:**
```bash
bash scripts/vllm/serve_qwen.sh \
    "/path/to/Qwen/Qwen2.5-VL-72B-Instruct" \
    8 \
    9001
```

**Example for 7B model on 1 GPU:**
```bash
bash scripts/vllm/serve_qwen.sh \
    "/path/to/Qwen/Qwen2-VL-7B-Instruct" \
    1 \
    9001
```

#### What the Script Does

1. **Validates** model path exists
2. **Configures NCCL** to avoid shared memory issues on HPC clusters
3. **Sets GPU visibility** for tensor parallelism
4. **Starts vLLM** with optimized parameters:
   - Image handling: Up to 30 images per prompt
   - Image resolution: 12,960,000 max pixels (high-res screenshots)
   - GPU memory utilization: 75%
   - Tensor parallel size: Specified number of GPUs
5. **Logs output** to `vllm_logs/` directory

#### Verify Server is Running

```bash
# Check if server is responding
curl http://localhost:9001/v1/models \
    -H "Authorization: Bearer qwen"

# Expected output:
# {"object":"list","data":[{"id":"qwen_vllm","object":"model",...}]}
```

### Manual vLLM Startup (Alternative)

If you prefer to start vLLM manually:

```bash
vllm serve /path/to/Qwen/Qwen2.5-VL-72B-Instruct \
    --port 9001 \
    --served-model-name "qwen_vllm" \
    --gpu-memory-utilization 0.75 \
    --tensor-parallel-size 8 \
    --limit-mm-per-prompt "image=30" \
    --mm-processor-kwargs '{"max_pixels":12960000,"min_pixels":4096}' \
    --api-key "qwen"
```

---

## Running VWA Tasks

### Quick Start: Example Script

Run a few test tasks to verify the setup:

```bash
# Load environment
source ~/.exact_env

# Run example tasks (tasks 10, 11, 70, 71 from Classifieds)
bash shells/qwen_vllm_example.sh
```

This will:
- Start the vLLM server (if not already running)
- Run 4 VWA tasks using your QWEN model
- Save results to `data/visualwebarena/eval_results/qwen_vllm/example/`

### Running Individual Agents

We provide shell scripts for different agent types using QWEN-vLLM:

#### 1. ReACT Agent (Baseline)
```bash
bash shells/classifieds/qwen_react_som.sh
```

#### 2. R-MCTS Agent (Reflective MCTS)
```bash
bash shells/classifieds/qwen_rmcts_som.sh
```

#### 3. R-MCTS with Multi-Agent Debate (Best Performance)
```bash
bash shells/classifieds/qwen_rmcts_mad_som.sh
```

### Parallel Evaluation

For large-scale evaluation across multiple tasks, use the parallel runner:

```bash
python runners/eval/eval_vwa_parallel.py \
    --env_name classifieds \
    --save_dir data/visualwebarena/eval_results/qwen_rmcts/classifieds \
    --eval_script shells/classifieds/qwen_rmcts_mad_som.sh \
    --run_mode greedy \
    --start_idx 0 \
    --end_idx 234 \
    --num_parallel 2 \
    --main_api_providers vllm,vllm \
    --num_task_per_script 2 \
    --num_task_per_reset 8
```

**Key Parameters:**
- `--num_parallel`: Number of parallel processes (careful with GPU memory!)
- `--start_idx`, `--end_idx`: Range of task IDs to evaluate
- `--num_task_per_reset`: How many tasks before resetting the environment
- `--main_api_providers`: Use `vllm` for your self-hosted model

### Other Environments

Run on different VWA environments:

```bash
# Reddit (210 tasks)
python runners/eval/eval_vwa_parallel.py \
    --env_name reddit \
    --eval_script shells/reddit/qwen_rmcts_mad_som.sh \
    ...

# Shopping (466 tasks)
python runners/eval/eval_vwa_parallel.py \
    --env_name shopping \
    --eval_script shells/shopping/qwen_rmcts_mad_som.sh \
    ...
```

---

## Advanced Configuration

### Customizing the Model Wrapper

The QWEN-vLLM wrapper is located at `src/llms/wrappers/qwen_vllm_wrapper.py`. You can modify:

#### Generation Parameters
```python
# In the wrapper __init__
self.max_new_tokens = 512  # Adjust max output length
self.temperature = 0.7     # Sampling temperature
self.top_p = 0.95          # Nucleus sampling
self.frequency_penalty = 0.0  # Repetition penalty
```

#### Image Processing
```python
# For high-resolution screenshots (recommended for VWA)
# In serve_qwen.sh, adjust mm-processor-kwargs:
--mm-processor-kwargs '{"max_pixels":12960000,"min_pixels":4096}'

# For lower GPU memory usage, reduce max_pixels:
--mm-processor-kwargs '{"max_pixels":6480000,"min_pixels":4096}'
```

#### Multi-Image Support
```python
# The wrapper already supports multi-crop/multi-image inputs
# Adjust in serve_qwen.sh:
--limit-mm-per-prompt "image=30"  # Max 30 images per request
```

### R-MCTS Search Parameters

Modify search behavior in your shell script:

```bash
# In shells/classifieds/qwen_rmcts_mad_som.sh

# Search depth and budget
max_depth=4          # Lookahead depth (4 = 5 steps)
max_steps=5          # Maximum steps per task
branching_factor=5   # Number of actions to explore per state
vf_budget=20         # Value function evaluation budget
time_budget=5.0      # Time budget per step (minutes)

# MCTS exploration
puct=1.0            # Exploration constant (higher = more exploration)

# Reflection settings
max_reflections_per_task=3    # Number of reflections
reflection_threshold=0.5      # When to trigger reflection
```

### Prompt Customization

Modify the prompt templates in `src/prompts/vwa/raw/`:

```bash
# Main prompt file for SoM (Set-of-Marks) representation
src/prompts/vwa/raw/p_som_cot_id_actree_3s_final.py

# Generate JSON from Python prompts
python src/prompts/vwa/to_json.py
```

---

## Troubleshooting

### Issue 1: vLLM Server Won't Start

**Symptom:** Server crashes on startup or shows NCCL errors

**Solutions:**
1. **Shared memory issues:**
   ```bash
   # The script already handles this, but if you run manually:
   export NCCL_P2P_DISABLE=1
   export NCCL_SHM_USE_CUDA_MEMCPY=1
   ```

2. **GPU memory:**
   ```bash
   # Reduce memory utilization in serve_qwen.sh:
   --gpu-memory-utilization 0.6  # Instead of 0.75
   ```

3. **Check GPU availability:**
   ```bash
   nvidia-smi  # Verify GPUs are visible and have free memory
   ```

### Issue 2: Connection Refused

**Symptom:** `requests.exceptions.ConnectionError: ('Connection aborted.', ConnectionRefusedError(111, 'Connection refused'))`

**Solutions:**
1. **Verify server is running:**
   ```bash
   curl http://localhost:9001/v1/models -H "Authorization: Bearer qwen"
   ```

2. **Check port is listening:**
   ```bash
   lsof -i :9001  # Should show vllm process
   ```

3. **Firewall rules** (if running on different nodes):
   ```bash
   # Allow port 9001 through firewall
   # Adjust llm_api_base in configs/llms/providers.yaml to correct hostname
   ```

### Issue 3: Out of Memory (OOM)

**Symptom:** CUDA out of memory errors

**Solutions:**
1. **Reduce image resolution:**
   ```bash
   # In serve_qwen.sh, lower max_pixels:
   --mm-processor-kwargs '{"max_pixels":6480000,"min_pixels":4096}'
   ```

2. **Use fewer GPUs with tensor parallelism** (counterintuitive but helps):
   ```bash
   # For 72B model, try 4 GPUs instead of 8
   bash scripts/vllm/serve_qwen.sh <MODEL_PATH> 4 9001
   ```

3. **Enable CPU offloading** (slower but uses less GPU memory):
   ```bash
   # Add to vllm serve command:
   --enable-cpu-offload
   ```

### Issue 4: Slow Inference

**Symptom:** Each request takes very long

**Solutions:**
1. **Check GPU utilization:**
   ```bash
   nvidia-smi -l 1  # Monitor GPU usage
   ```

2. **Increase GPU memory utilization:**
   ```bash
   --gpu-memory-utilization 0.85  # If you have headroom
   ```

3. **Enable prefix caching** (for repeated prompts):
   ```bash
   # Add to vllm serve:
   --enable-prefix-caching
   ```

### Issue 5: Model Not Downloading

**Symptom:** `OSError: <model_path> does not appear to be a folder`

**Solutions:**
1. **Verify HF token:**
   ```bash
   huggingface-cli whoami
   ```

2. **Pre-download model:**
   ```bash
   huggingface-cli download Qwen/Qwen2.5-VL-72B-Instruct \
       --local-dir /path/to/save/model
   ```

3. **Use model ID instead of path:**
   ```bash
   # In serve_qwen.sh, use HF model ID:
   vllm serve Qwen/Qwen2.5-VL-72B-Instruct ...
   ```

### Issue 6: Tasks Failing to Execute

**Symptom:** Agent cannot interact with websites

**Solutions:**
1. **Verify VWA websites are running:**
   ```bash
   curl $CLASSIFIEDS  # Should return HTML
   ```

2. **Check environment variables:**
   ```bash
   echo $CLASSIFIEDS
   echo $SHOPPING
   # Should print URLs, not empty
   ```

3. **Reset environments:**
   ```bash
   # Use the VWA management server
   cd runners/utils/vwa_mgm_server
   python gradio_server.py --port 55405
   # Then reset via GUI at localhost:55405
   ```

### Getting Help

- **ExACT Issues:** https://github.com/microsoft/ExACT/issues
- **vLLM Issues:** https://github.com/vllm-project/vllm/issues
- **VWA Issues:** https://github.com/web-arena-x/visualwebarena/issues

---

## Performance Tips

### 1. Optimize for Your Cluster

```bash
# In serve_qwen.sh, tune NCCL for your network:
export NCCL_IB_DISABLE=0     # Enable InfiniBand if available
export NCCL_NET_GDR_LEVEL=0  # Disable GPU Direct if issues
```

### 2. Batch Multiple Requests

For parallel evaluation, the system automatically batches requests to the vLLM server.

### 3. Monitor Resource Usage

```bash
# GPU memory and utilization
watch -n 1 nvidia-smi

# vLLM logs
tail -f vllm_logs/vllm_logfile_*.txt
```

### 4. Use Wandb for Tracking

The wrapper automatically logs to Weights & Biases if available:

```bash
pip install wandb
wandb login

# Metrics logged:
# - Generation time
# - Token counts
# - Temperature/sampling params
# - Number of images processed
```

---

## Directory Structure

```
ExACT/
├── configs/
│   ├── llms/
│   │   └── providers.yaml          # API provider configs (vLLM here)
│   └── visualwebarena/
│       └── test_classifieds_v2/    # Task configs
├── scripts/
│   └── vllm/
│       └── serve_qwen.sh           # vLLM server startup script
├── src/
│   ├── llms/
│   │   ├── wrappers/
│   │   │   └── qwen_vllm_wrapper.py  # Model wrapper
│   │   └── utils.py                # LLM calling utilities
│   ├── agent/                      # Agent implementations
│   └── prompts/                    # Prompt templates
├── shells/
│   ├── example.sh                  # Quick start example
│   └── classifieds/
│       ├── qwen_react_som.sh       # ReACT agent
│       ├── qwen_rmcts_som.sh       # R-MCTS agent
│       └── qwen_rmcts_mad_som.sh   # R-MCTS + Debate
├── runners/
│   └── eval/
│       ├── eval_vwa_ragent.py      # Single task runner
│       └── eval_vwa_parallel.py    # Parallel task runner
└── data/                           # Results saved here
```

---

## Next Steps

1. **Start simple:** Run the example script with a few tasks
2. **Iterate on prompts:** Customize prompts for your use case
3. **Scale up:** Use parallel evaluation for full benchmarks
4. **Fine-tune:** Optionally fine-tune QWEN on VWA trajectories (see Exploratory Learning in main README)

## Citation

If you use this integration or the ExACT framework, please cite:

```bibtex
@misc{yu2024exactteachingaiagents,
    title={ExACT: Teaching AI Agents to Explore with Reflective-MCTS and Exploratory Learning},
    author={Xiao Yu and Baolin Peng and Vineeth Vajipey and Hao Cheng and Michel Galley and Jianfeng Gao and Zhou Yu},
    year={2024},
    eprint={2410.02052},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2410.02052},
}
```

---

**Happy evaluating with QWEN on Visual Web Arena!**
