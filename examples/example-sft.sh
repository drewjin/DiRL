cd llama_factory_sdar

# 分布式环境变量（兼容 torchrun / accelerate / Slurm 等）
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-12345}
export NODE_RANK=${RANK:-0}  # 注意：有些系统用 RANK 表示 node rank

# 自动推导 nnodes（节点数）和 nproc_per_node（每节点GPU数）
WORLD_SIZE=${WORLD_SIZE:-1}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
NUM_MACHINES=${WORLD_SIZE}
TOTAL_GPUS=$((NUM_MACHINES * NUM_GPUS_PER_NODE))

# 设置 CUDA 相关环境变量
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export PYTHONDONTWRITEBYTECODE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# 添加模型目录到 PYTHONPATH，解决自定义模型配置导入问题
MODEL_DIR=${MODEL_DIR:-"xxx/SDAR-8B-Chat"}
export PYTHONPATH="${MODEL_DIR}:${PYTHONPATH}"

# 清理冲突变量
unset NCCL_BLOCKING_WAIT
unset NCCL_ASYNC_ERROR_HANDLING

# Wandb 配置
SCRIPT_NAME="${SCRIPT_NAME:-sft_sdar}"
RUN_NAME=WANDB_${SCRIPT_NAME}
export WANDB_PROJECT=${RUN_NAME}
export WANDB_DIR=${WANDB_PROJECT}
mkdir -p "$WANDB_DIR"
export WANDB_MODE=offline

echo "[NUM NODES: $NUM_MACHINES, GPUs per node: $NUM_GPUS_PER_NODE, Total GPUs: $TOTAL_GPUS] SFT training start ..."

torchrun \
    --nnodes ${NUM_MACHINES} \
    --node_rank ${NODE_RANK} \
    --nproc_per_node ${NUM_GPUS_PER_NODE} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    src/llamafactory/launcher.py \
    examples/train_full_sdar/sdar_8b/sft_sdar_8b.yaml

echo "SFT training finished ..."
