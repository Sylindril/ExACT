# Quick Start: Running VWA with QWEN-vLLM

This is a condensed quick-start guide. For the full guide, see [QWEN_VLLM_INTEGRATION_GUIDE.md](QWEN_VLLM_INTEGRATION_GUIDE.md).

## Prerequisites

- HPC cluster with NVIDIA GPUs
- Python 3.9+
- vLLM installed (`pip install vllm`)
- Visual Web Arena environment set up

## 1. Set Environment Variables

```bash
# Create ~/.exact_env
export QWEN_MODEL_PATH="/path/to/Qwen/Qwen2.5-VL-72B-Instruct"
export QWEN_NUM_GPUS=8
export VLLM_PORT=9001

# VWA URLs
export DATASET="visualwebarena"
export CLASSIFIEDS="<your_domain>:9980"
export SHOPPING="<your_domain>:7770"
export REDDIT="<your_domain>:9999"
# ... (see full guide for all URLs)

# OpenAI API key (for embeddings only)
export OPENAI_API_KEY="sk-..."

source ~/.exact_env
```

## 2. Start vLLM Server

```bash
cd /path/to/ExACT

# Start vLLM server
bash scripts/vllm/serve_qwen.sh \
    "${QWEN_MODEL_PATH}" \
    ${QWEN_NUM_GPUS} \
    ${VLLM_PORT}
```

Wait for server to be ready (check logs in `vllm_logs/`).

## 3. Generate Task Configs

```bash
export DATASET=visualwebarena
python runners/utils/generate_test_configs.py
```

## 4. Run Example Tasks

```bash
# Run the quick example (4 tasks)
bash shells/qwen_vllm_example.sh
```

Results will be in `data/visualwebarena/eval_results/qwen_vllm/example/`.

## 5. Run Full Evaluation

### Single Environment (Classifieds)

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

### All Environments

```bash
# Reddit (210 tasks)
python runners/eval/eval_vwa_parallel.py \
    --env_name reddit \
    --save_dir data/visualwebarena/eval_results/qwen_rmcts/reddit \
    --eval_script shells/reddit/qwen_rmcts_mad_som.sh \
    --start_idx 0 --end_idx 210 \
    --num_parallel 2 --main_api_providers vllm,vllm \
    --num_task_per_script 2 --num_task_per_reset 8

# Shopping (466 tasks)
python runners/eval/eval_vwa_parallel.py \
    --env_name shopping \
    --save_dir data/visualwebarena/eval_results/qwen_rmcts/shopping \
    --eval_script shells/shopping/qwen_rmcts_mad_som.sh \
    --start_idx 0 --end_idx 466 \
    --num_parallel 2 --main_api_providers vllm,vllm \
    --num_task_per_script 2 --num_task_per_reset 8
```

## Agent Types

Different agent scripts are available:

- `qwen_react_som.sh` - ReACT baseline (no search)
- `qwen_rmcts_mad_som.sh` - R-MCTS with Multi-Agent Debate (best performance)
- Copy and modify for other environments (reddit, shopping)

## Troubleshooting

### Server won't start
```bash
# Check GPU availability
nvidia-smi

# Reduce memory usage
# Edit scripts/vllm/serve_qwen.sh:
--gpu-memory-utilization 0.6
```

### Connection refused
```bash
# Verify server is running
curl http://localhost:9001/v1/models -H "Authorization: Bearer qwen"

# Check port
lsof -i :9001
```

### Out of memory
```bash
# Reduce image resolution in serve_qwen.sh:
--mm-processor-kwargs '{"max_pixels":6480000,"min_pixels":4096}'
```

## File Structure

```
ExACT/
├── QWEN_VLLM_INTEGRATION_GUIDE.md    # Full guide
├── QUICKSTART_QWEN.md                # This file
├── scripts/vllm/
│   └── serve_qwen.sh                 # vLLM server startup
├── src/llms/wrappers/
│   └── qwen_vllm_wrapper.py          # Model wrapper
├── shells/
│   ├── qwen_vllm_example.sh          # Quick example
│   └── classifieds/
│       ├── qwen_react_som.sh         # ReACT agent
│       └── qwen_rmcts_mad_som.sh     # R-MCTS agent
└── configs/llms/
    └── providers.yaml                # vLLM config
```

## Next Steps

- **Customize prompts**: Edit `src/prompts/vwa/raw/*.py`
- **Tune search**: Adjust parameters in shell scripts
- **Scale up**: Use parallel evaluation
- **Full guide**: Read [QWEN_VLLM_INTEGRATION_GUIDE.md](QWEN_VLLM_INTEGRATION_GUIDE.md)

## Getting Help

- **GitHub Issues**: https://github.com/microsoft/ExACT/issues
- **Full Documentation**: See QWEN_VLLM_INTEGRATION_GUIDE.md
