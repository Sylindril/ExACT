#!/bin/bash

# check if argument is provided, otherwise use default
CHECKPOINT_PATH=$1
NUM_GPUS=$2
PORT=$3

# Check if checkpoint path exists
if [ ! -e "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint path '$CHECKPOINT_PATH' does not exist." >&2
    exit 1
fi

echo "CHECKPOINT_PATH: $CHECKPOINT_PATH"
echo "NUM_GPUS: $NUM_GPUS"
echo "PORT: $PORT"

mkdir -p vllm_logs

#------------------------- In Case of Shared Memory Issues --------------------------------

# Clean up any leftover shared memory segments from previous runs
echo "Cleaning up any leftover shared memory segments..."
ls /dev/shm/nccl-* 2>/dev/null | xargs rm -f 2>/dev/null || true
ipcs -m | grep $USER | awk '{print $2}' | xargs -r ipcrm -m 2>/dev/null || true

# NCCL environment variables - focus on shared memory size limits
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1           # Disable P2P to avoid shared memory issues
export NCCL_CUMEM_ENABLE=0          # Already set in logs
export NCCL_IB_DISABLE=0            # Keep InfiniBand enabled for HPC
export NCCL_NET_GDR_LEVEL=0         # Disable GPU Direct RDMA

# Key fix: Limit shared memory usage and force smaller segments
export NCCL_SHM_USE_CUDA_MEMCPY=1   # Use CUDA memcpy instead of shared memory
export NCCL_BUFFSIZE=1048576        # Reduce buffer size to 1MB (from default ~9MB)
export NCCL_NTHREADS=8              # Reduce threads to limit memory usage
export NCCL_MAX_NCHANNELS=4         # Limit number of channels

# Force NCCL to use network transport for large transfers
export NCCL_MIN_NCHANNELS=1
export NCCL_CROSS_NIC=1

# Additional memory and performance settings
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Explicitly set visible GPUs
export OMP_NUM_THREADS=1                     # Prevent CPU thread contention

echo "Starting vLLM server with NCCL shared memory optimizations..."
#---------------------------------------------------------------------------------------------

vllm serve $CHECKPOINT_PATH \
    --port $PORT \
    --served-model-name "qwen_vllm" \
    --gpu-memory-utilization 0.75 \
    --tensor-parallel-size $NUM_GPUS \
    --uvicorn-log-level info \
    --limit-mm-per-prompt "image=30" \
    --mm-processor-kwargs '{"max_pixels":12960000,"min_pixels":4096}' \
    --api-key "qwen" \
    > "vllm_logs/vllm_logfile_$(date '+%Y-%m-%d_%H-%M-%S').txt" 2>&1

# Check if the command succeeded, and log a failure message if not
if [ $? -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: vLLM serve command failed" | tee -a "vllm_logs/vllm_logfile_$(date '+%Y-%m-%d_%H-%M-%S').txt"
    #------------------------- In Case of Shared Memory Issues --------------------------------
    # Clean up on failure
    # echo "Cleaning up shared memory segments after failure..."
    # ls /dev/shm/nccl-* 2>/dev/null | xargs rm -f 2>/dev/null || true
    # ipcs -m | grep $USER | awk '{print $2}' | xargs -r ipcrm -m 2>/dev/null || true
    #---------------------------------------------------------------------------------------------
fi
