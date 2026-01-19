#!/bin/bash

echo $(which python)

# 分布式环境变量（兼容 torchrun / accelerate / Slurm 等）
export MASTER_ADDR=${MASTER_ADDR:-"localhost"}
export MASTER_PORT=${MASTER_PORT:-29500}
export NODE_RANK=${RANK:-0}  # 注意：有些系统用 RANK 表示 node rank

# 自动推导 num_machines（节点数）
WORLD_SIZE=${WORLD_SIZE:-1}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
NUM_MACHINES=${WORLD_SIZE}
TOTAL_GPUS=$((NUM_MACHINES * NUM_GPUS_PER_NODE))

# 设置 CUDA 相关环境变量
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# 设置每个进程独立的编译缓存目录，避免多进程写冲突
export TRITON_CACHE_DIR=/tmp/triton_cache_rank${RANK:-0}
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache_rank${RANK:-0}

export PYTHONDONTWRITEBYTECODE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=./cache/${SCRIPT_NAME}

# 清理冲突变量
unset NCCL_BLOCKING_WAIT
unset NCCL_ASYNC_ERROR_HANDLING

# RL 参数（可通过环境变量传入）
SCRIPT_NAME="${SCRIPT_NAME:-rl_sdar}"
RUN_NAME=WANDB_${SCRIPT_NAME}

# ========= 统一 checkpoint 保存前缀（基于工作区路径推导） =========
# 默认把所有实验的 ckpt 都收敛到 <repo_root>/temp_ckpt/ 下。
# 你之后只需要把 temp_ckpt 做软链接到大盘/共享盘即可，脚本无需再改。
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/temp_ckpt}"
PROJECT_DIR="${CKPT_ROOT}/${SCRIPT_NAME}_lmdeploy"
mkdir -p "${PROJECT_DIR}"
echo "[PROJECT_DIR: ${PROJECT_DIR}]"

export WANDB_PROJECT=${RUN_NAME}
export WANDB_DIR="${PROJECT_DIR}/wandb"
mkdir -p "$WANDB_DIR"
export WANDB_MODE=offline

# ========= Ray / /dev/shm 兼容性（lmdeploy 的推理引擎会用 Ray） =========
# 你的环境里 /dev/shm 很小（例如 Docker 默认 64MB），Ray 默认会尝试分配很大的 object store
# 导致报错：object store size exceeds /dev/shm size。
# 这里默认允许 Ray 使用磁盘作为 object store（会慢一点，但能跑起来）。
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=${RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE:-1}
# 把 Ray 的临时目录放到实验目录下，避免写到系统盘（可按需改到大盘路径）
export RAY_TMPDIR=${RAY_TMPDIR:-${PROJECT_DIR}/ray_tmp}
mkdir -p "${RAY_TMPDIR}"
# 可选：限制 Ray object store 最大值，避免默认 80GB（单位：bytes）
export RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=${RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES:-5000000000}

PRETRAINED_MODEL=${PRETRAINED_MODEL:-xxx/SDAR-8B-Chat}
NUM_TASK_PER_STEP=${NUM_TASK_PER_STEP:-128}
NUM_RESPONSE_PER_TASK=${NUM_RESPONSE_PER_TASK:-8}
REWARD_FUNCS=${REWARD_FUNCS:-accuracy}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
BLOCK_SIZE=${BLOCK_SIZE:-4}
MAX_TOKEN=${MAX_TOKEN:-8192}
TOP_K=${TOP_K:-50}
TOP_P=${TOP_P:-1.0}
TEMPERATURE=${TEMPERATURE:-1.0}
DENOISING_STEPS_PER_BLOCK=${DENOISING_STEPS_PER_BLOCK:-4}
REMASKING_STRATEGY=${REMASKING_STRATEGY:-low_confidence_dynamic}
DYNAMIC_THRESHOLD=${DYNAMIC_THRESHOLD:-0.90}
SAVE_EVERY=${SAVE_EVERY:-1}
TRAIN_DATASET=${TRAIN_DATASET:-MATH_train}
EVAL_DATASET=${EVAL_DATASET:-MATH500}
EVAL_BLOCK_SIZE=${EVAL_BLOCK_SIZE:-4}
EVAL_DENOISING_STEPS=${EVAL_DENOISING_STEPS:-4}
EVAL_MAX_TOKEN=${EVAL_MAX_TOKEN:-8192}
EVAL_TOP_K=${EVAL_TOP_K:-50}
EVAL_TOP_P=${EVAL_TOP_P:-1.0}
EVAL_REMASKING_STRATEGY=${EVAL_REMASKING_STRATEGY:-low_confidence_dynamic}
EVAL_DYNAMIC_THRESHOLD=${EVAL_DYNAMIC_THRESHOLD:-0.90}
EVAL_TEMPERATURE=${EVAL_TEMPERATURE:-1.0}
EVAL_EVERY=${EVAL_EVERY:-0}
KL=${KL:-0}
SHRINK=${SHRINK:-1}
THINK=${THINK:-False}
LR=${LR:-1e-6}
BS=${BS:-1}
MINI_BS=${MINI_BS:-0}
CURSOR=${CURSOR:-0}
ITERATIONS=${ITERATIONS:-1}
COLLATE=${COLLATE:-false}
NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-10}
CURRENT_STEP=${CURRENT_STEP:-1}
echo "[NUM NODES: $NUM_MACHINES] RL training start ..."

# # 显存优化：减少碎片化
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  # --config_file accelerate_configs/${WORLD_SIZE}_node_8_gpus_deepspeed_zero1.yaml \
# # 启动训练
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --main_process_port ${MASTER_PORT} \
  --machine_rank ${NODE_RANK} \
  --main_process_ip ${MASTER_ADDR} \
  --config_file accelerate_configs/deepspeed_zero1.yaml \
  --num_machines ${NUM_MACHINES} \
  --num_processes ${TOTAL_GPUS} \
  train/grpo.py \
  config=configs/rl.yaml \
  model.pretrained_model=${PRETRAINED_MODEL} \
  experiment.project=${SCRIPT_NAME}_lmdeploy \
  experiment.output_dir=${PROJECT_DIR} \
  experiment.cursor=${CURSOR} \
  experiment.num_nodes=${NUM_MACHINES} \
  experiment.save_every=${SAVE_EVERY} \
  dataset.train_dataset=${TRAIN_DATASET} \
  dataset.eval_dataset=${EVAL_DATASET} \
  models.pretrained_model=${PRETRAINED_MODEL} \
  experiment.current_step=${CURRENT_STEP} \
  training.num_train_steps=${NUM_TRAIN_STEPS} \
  training.batch_size_lm=${BS} \
  training.mini_batch_size=${MINI_BS} \
  training.num_iterations=${ITERATIONS} \
  training.collate=${COLLATE} \
  rollout.start_with_think=${THINK} \
  rollout.num_task_per_step=${NUM_TASK_PER_STEP} \
  rollout.num_response_per_task=${NUM_RESPONSE_PER_TASK} \
  training.beta=${KL} \
  training.shrink=${SHRINK} \
  training.reward_funcs=${REWARD_FUNCS} \
  optimizer.params.learning_rate=${LR} \
  training.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
  rollout.block_size=${BLOCK_SIZE} \
  rollout.max_token=${MAX_TOKEN} \
  rollout.top_k=${TOP_K} \
  rollout.top_p=${TOP_P} \
  rollout.temperature=${TEMPERATURE} \
  rollout.denoising_steps_per_block=${DENOISING_STEPS_PER_BLOCK} \
  rollout.remasking_strategy=${REMASKING_STRATEGY} \
  rollout.dynamic_threshold=${DYNAMIC_THRESHOLD} \
  rollout.do_sample=True \
  evaluation.do_sample=True \
  evaluation.block_size=${EVAL_BLOCK_SIZE} \
  evaluation.denoising_steps_per_block=${EVAL_DENOISING_STEPS} \
  evaluation.max_active=${EVAL_MAX_ACTIVE} \
  evaluation.max_token=${EVAL_MAX_TOKEN} \
  evaluation.top_k=${EVAL_TOP_K} \
  evaluation.remasking_strategy=${EVAL_REMASKING_STRATEGY} \
  evaluation.dynamic_threshold=${EVAL_DYNAMIC_THRESHOLD} \
  evaluation.temperature=${EVAL_TEMPERATURE} \
  experiment.eval_every=${EVAL_EVERY}

echo "RL training finished ..."

