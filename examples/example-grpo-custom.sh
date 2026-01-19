#!/bin/bash
# 运行前请确保在项目根目录 DiRL 下运行

CKPT_PATH=/root/workspace/jyj/SDAR/training/llama_factory_sdar/temp/sdar_ckpt
CKPT_NAME=sdar_8b_b32_ga1_lr2e-5_math_glm_openr1math
export CKPT_ROOT=/root/workspace/jyj/DiRL/temp_ckpts

# ===== 基础配置参数 =====
export MODEL=sdar                           # 模型架构类型，默认为 sdar (Diffusion LM)
export SCRIPT_NAME=dirl_grpo_experiment_custom_sft_model     # 实验名称，用于 WandB 记录和模型保存子目录名
export PRETRAINED_MODEL=${CKPT_PATH}/${CKPT_NAME}/full/sft # 预训练/SFT 后的模型路径 (建议使用绝对路径)

# ===== 数据集配置 =====
export DATASET=BigMath_train                # 原始数据集标识
export TRAIN_DATASET=MATH_train             # RL 训练使用的题目数据集文件名 (位于 data/ 目录下)
export EVAL_DATASET=MATH500                 # 评估模型性能的测试数据集文件名
export CURSOR=0                              # 数据集遍历游标：从第几条题目开始采样 (断点续训时很有用)
export CURRENT_EPOCH=1                      # 当前训练的 Epoch 轮数计数

# ===== RL (GRPO) 核心算法参数 =====
export NUM_TASK_PER_STEP=128                # 每个训练步采样的题目数量 (Batch Size of tasks)
export NUM_RESPONSE_PER_TASK=8              # GRPO 中的 K 值：每道题生成的回答数量。用于计算相对 Reward
export KL=0.0                               # KL 散度惩罚系数 (Beta)，设为 0 表示不强制约束模型漂移
export REWARD_FUNCS=accuracy                # 奖励函数类型，可选: accuracy, cosine, speed, format 等 (逗号分隔)
export SPEED=false                          # 是否启用速度/效率相关的特殊奖励逻辑

# ===== 训练超参数 =====
export LR=1e-6                              # 学习率 (Learning Rate)，RL 阶段通常设为很小的值
export BS=1                                 # 每个进程的训练批大小 (Batch Size)
export GRADIENT_ACCUMULATION_STEPS=1        # 梯度累积步数
export NUM_EPOCHS=80                        # 总训练 Epoch 数
export SAVE_EVERY=1                         # 每隔多少个 Step 保存一次模型 Checkpoint
export SHRINK=1                             # 扩散模型特有：去噪步骤的收缩/压缩倍率，1 为标准配置

# ===== 采样 (Rollout) 生成参数 (扩散模型特有) =====
export BLOCK_SIZE=32                        # 每个扩散 Block 包含的 token 数量
export MAX_TOKEN=8192                       # 最大生成长度限制
export TOP_K=50                             # 采样时的 Top-K 过滤
export TOP_P=1.0                            # 采样时的 Top-P (Nucleus) 过滤
export TEMPERATURE=1.0                      # 采样温度，越高随机性越大
export DENOISING_STEPS_PER_BLOCK=4          # 每个 Block 的去噪迭代步数
export REMASKING_STRATEGY=low_confidence_dynamic # 重掩码策略：基于低置信度动态重掩码
export DYNAMIC_THRESHOLD=0.90               # 动态重掩码的置信度阈值

# ===== 评估 (Evaluation) 参数 (通常与采样参数保持一致或更严格) =====
export EVAL_EVERY=1                         # 每隔多少个 Step 进行一次测试集评估
export EVAL_BLOCK_SIZE=4
export EVAL_DENOISING_STEPS=4
export EVAL_MAX_TOKEN=8192
export EVAL_TOP_K=50
export EVAL_TOP_P=1.0
export EVAL_REMASKING_STRATEGY=low_confidence_dynamic
export EVAL_DYNAMIC_THRESHOLD=0.90
export EVAL_TEMPERATURE=1.0

# 启动训练脚本
scripts/grpo.sh
