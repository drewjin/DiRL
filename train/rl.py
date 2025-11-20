import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import wandb
import torch
from torch.optim import AdamW

from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed



from models import SDARForCausalLM
from train.prompting_utils import UniversalPrompting
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error

from torch.utils.data import Dataset, DataLoader
from train.utils import get_config, flatten_omega_conf, AverageMeter

try:
    import apex

    is_apex_available = True
except ImportError:
    is_apex_available = False

from rollout_dapo import rollout_sampling, rollout_sampling_eval
# from rollout import rollout_sampling

logger = get_logger(__name__, log_level="INFO")

import torch.distributed as dist
from transformers import AutoTokenizer

import os

import os

def inject_debug_print_all(model_path: str, target_files=None):
    """
    在模型目录下的多个文件顶部插入调试打印语句。
    默认包含：configuration_*.py, tokenization_*.py, modeling_*.py
    """
    if target_files is None:
        target_files = [
            "configuration_sdar.py",
            "tokenization_qwen2.py",
            "modeling_sdar.py"
        ]

    for filename in target_files:
        file_path = os.path.join(model_path, filename)
        if not os.path.exists(file_path):
            print(f"[跳过] {file_path} 不存在")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            original = f.read()

        injection = f'print(f"--- I am EXECUTING {filename} from location: {{__file__}} ---")\n'

        if injection.strip() not in original:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(injection + original)
            print(f"[完成] 已在 {file_path} 顶部插入调试打印。")
        else:
            print(f"[跳过] {file_path} 已经有调试打印了。")




class TrainDataset(Dataset):
    def __init__(self, extended_input_ids, p_mask, tok_idx_ext, labels, reward, raw_reward):
        self.extended_input_ids = extended_input_ids
        self.p_mask = p_mask
        self.tok_idx_ext = tok_idx_ext
        self.labels = labels
        self.reward   = reward
        self.raw_reward = raw_reward
        self.logp_old_tok = torch.full(
            (len(extended_input_ids), p_mask.shape[1]), 
            float('-inf')
        )

    def __len__(self):
        return len(self.extended_input_ids)

    def __getitem__(self, idx):
        return (
            idx,
            self.extended_input_ids[idx],
            self.p_mask[idx],
            self.tok_idx_ext[idx],
            self.labels[idx],
            self.reward[idx],
            self.raw_reward[idx],
        )


def train():
    #########################
    # SETUP Accelerator     #
    #########################
    config = get_config()

    project_name = config.experiment.project
    if config.experiment.current_epoch == 1:
        pretrained_model = config.model.pretrained_model
    else:
        pretrained_model = config.experiment.prefix_dir + project_name + f"/ckpt/epoch-{config.experiment.current_epoch-1}"

    # Enable TF32 on Ampere GPUs
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.project) / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
        split_batches=True,
    )

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    # Initialize wandb tracking (on all processes)
    resume_wandb_run = config.wandb.resume
    run_id = config.wandb.get("run_id", None)
    if run_id is None:
        resume_wandb_run = False
        run_id = wandb.util.generate_id()
        config.wandb.run_id = run_id

    wandb_init_kwargs = dict(
        name=config.experiment.project,
        id=run_id,
        resume=resume_wandb_run,
        entity=config.wandb.get("entity", None),
        config_exclude_keys=[],
    )
    wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
    wandb_config.pop("experiment.resume_from_checkpoint", None)

    accelerator.init_trackers(
        config.experiment.project,
        config=wandb_config,
        init_kwargs={"wandb": wandb_init_kwargs},
    )

    if accelerator.is_main_process:
        os.makedirs(config.experiment.project, exist_ok=True)
        config_path = Path(config.experiment.project) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")

    # Initialize global step counter
    global_step = 0  # counts every batch
    optimizer_step = 0  # counts every optimizer update (after gradient accumulation)
    model = None
    ref_model = None  # 参考模型，只加载一次
    optimizer = None
    lr_scheduler = None
    # 数据集遍历位置，用于顺序遍历（支持续训）
    data_cursor = getattr(config.experiment, 'cursor', 0)
    if data_cursor > 0:
        logger.info(f"Resuming from data cursor: {data_cursor}")

    import torch.nn.functional as F

    for epoch in range(config.experiment.current_epoch, config.training.num_train_epochs+1):

        logger.info(f"{'='*80}")
        logger.info(f"Starting Epoch {epoch}/{config.training.num_train_epochs}")
        logger.info(f"{'='*80}")

        #########################
        # 每轮的rollout采样  #
        #########################

        if epoch != config.experiment.current_epoch or config.experiment.current_epoch != 1:
            model_path = config.experiment.prefix_dir + project_name + f"/ckpt/epoch-{epoch-1}"
        else:
            model_path = pretrained_model
        logger.info(f"Loading model from {model_path}")
        # 防止加载导入时冲突
        sys.path.insert(0, model_path)
        from tokenization_qwen2 import Qwen2Tokenizer
        accelerator.wait_for_everyone()

        logger.info(f"Loading model from {model_path}")
        sys.path.insert(0, model_path)
        if accelerator.is_main_process:
            tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            tokenizer = Qwen2Tokenizer.from_pretrained(model_path)

        logger.info(f"Starting rollout sampling for epoch {epoch}...")

        if model is not None:
            model = model.cpu()

        # 注意：现在 rollout_sampling 返回 (data, updated_cursor) 元组
        dataset_load, data_cursor = rollout_sampling(
            config.dataset.train_dataset, epoch, config,
            model_path, tokenizer, accelerator,
            normalize=True, filter=True, mode="train", split_rank=False, reward_funcs=config.training.reward_funcs,
            data_cursor=data_cursor  # 传入当前cursor，顺序遍历数据集
        )

        logger.info(f"[Rank {accelerator.process_index}] Rollout sampling completed. Collected {len(dataset_load)} samples.")
        # Log response length statistics from rollout
        try:
            response_lengths = []
            correctness_list = []
            raw_rewards_list = []
            reward_components_list = []
            extract_rewards_list = []
            for item in dataset_load:
                if isinstance(item, dict):
                    if "response_length" in item:
                        response_lengths.extend(item["response_length"])
                    if "correctness" in item:
                        correctness_list.extend(item["correctness"])
                    if "raw_rewards" in item:
                        raw_rewards_list.extend(item["raw_rewards"])
                    if "reward_components" in item:
                        reward_components_list.extend(item["reward_components"])
                    if "extract_reward" in item:
                        extract_rewards_list.extend(item["extract_reward"])
            
            # Gather 所有 rank 的数据（一次性打包）
            from accelerate.utils import gather_object
            local_data = {
                'num_tasks': len(dataset_load),
                'response_lengths': response_lengths,
                'correctness': correctness_list,
                'raw_rewards': raw_rewards_list,
                'reward_components': reward_components_list,
                'extract_rewards': extract_rewards_list,
            }
            all_data = gather_object([local_data])
            
            # 解包并合并所有 rank 的数据
            all_num_tasks = sum(d['num_tasks'] for d in all_data)
            all_response_lengths = [item for d in all_data for item in d['response_lengths']]
            all_correctness = [item for d in all_data for item in d['correctness']]
            all_raw_rewards = [item for d in all_data for item in d['raw_rewards']]
            all_reward_components = [item for d in all_data for item in d['reward_components']]
            all_extract_rewards = [item for d in all_data for item in d['extract_rewards']]
            
            metrics_rollout = {
                "rollout/num_tasks": float(all_num_tasks),
                "rollout/epoch": float(epoch),
            }
            
            if len(all_response_lengths) > 0:
                resp_arr = np.asarray(all_response_lengths, dtype=np.int64)
                max_token = int(getattr(config.rollout, "max_token", 0) or 0)
                trunc_rate = float((resp_arr >= max_token).mean()) if max_token > 0 else 0.0
                metrics_rollout.update({
                    "rollout/num_responses": float(len(all_response_lengths)),
                    "rollout/response_length_mean": float(resp_arr.mean()),
                    "rollout/response_length_p50": float(np.percentile(resp_arr, 50)),
                    "rollout/response_length_p95": float(np.percentile(resp_arr, 95)),
                    "rollout/response_length_min": float(resp_arr.min()),
                    "rollout/response_length_max": float(resp_arr.max()),
                    "rollout/response_trunc_rate": trunc_rate,
                })
            
            if len(all_correctness) > 0:
                corr_arr = np.asarray(all_correctness, dtype=np.float32)
                metrics_rollout["rollout/correctness"] = float(corr_arr.mean())
            
            if len(all_extract_rewards) > 0:
                ext_rew_arr = np.asarray(all_extract_rewards, dtype=np.float32)
                metrics_rollout["rollout/extract_reward"] = float(ext_rew_arr.mean())
            
            # 统计raw_reward（加权求和后的值）
            if len(all_raw_rewards) > 0:
                raw_rew_arr = np.array(all_raw_rewards, dtype=np.float32)
                metrics_rollout["rollout/raw_reward"] = float(raw_rew_arr.mean())
            
            # 统计每个reward function的均值
            if len(all_reward_components) > 0:
                # all_reward_components是列表的列表，每个元素是一个列表[reward1, reward2, ...]
                reward_func_names = config.training.get("reward_funcs", "accuracy").split(',')
                reward_func_names = [name.strip() for name in reward_func_names]
                
                # 获取 reward_weights
                reward_weights_str = config.training.get("reward_weights", None)
                if reward_weights_str is not None:
                    if isinstance(reward_weights_str, str):
                        reward_weights = [float(w.strip()) for w in reward_weights_str.split(',') if w.strip()]
                    else:
                        reward_weights = list(reward_weights_str)
                else:
                    # 默认权重：所有reward function权重为1
                    reward_weights = [1.0] * len(reward_func_names)
                
                # 转换为numpy数组: shape = (num_samples, num_reward_funcs)
                comp_matrix = np.array(all_reward_components, dtype=np.float32)
                
                # 如果是1D数组（兼容性处理），reshape为2D
                if comp_matrix.ndim == 1:
                    comp_matrix = comp_matrix.reshape(-1, 1)
                
                # 计算每个reward function的均值
                for idx, func_name in enumerate(reward_func_names):
                    if idx < comp_matrix.shape[1]:
                        func_rewards = comp_matrix[:, idx]
                        metrics_rollout[f"rollout/reward_{func_name}"] = float(func_rewards.mean())
            
            if accelerator.is_main_process:
                # 打印基本统计信息
                logger.info(f"\n{'='*60}")
                logger.info(f"Rollout Statistics - Epoch {epoch}")
                logger.info(f"{'='*60}")
                logger.info(f"Total tasks collected: {metrics_rollout['rollout/num_tasks']:.0f}")
                
                if "rollout/num_responses" in metrics_rollout:
                    logger.info(f"Response length stats:")
                    logger.info(f"  Mean: {metrics_rollout['rollout/response_length_mean']:.1f}")
                    logger.info(f"  Min: {metrics_rollout['rollout/response_length_min']:.1f}")
                    logger.info(f"  Median (p50): {metrics_rollout['rollout/response_length_p50']:.1f}")
                    logger.info(f"  p95: {metrics_rollout['rollout/response_length_p95']:.1f}")
                    logger.info(f"  Max: {metrics_rollout['rollout/response_length_max']:.1f}")
                    logger.info(f"  Truncation rate: {metrics_rollout['rollout/response_trunc_rate']:.3f}")
                
                if "rollout/correctness" in metrics_rollout:
                    logger.info(f"Correctness: {metrics_rollout['rollout/correctness']:.4f}")
                
                if "rollout/extract_reward" in metrics_rollout:
                    logger.info(f"Extract reward: {metrics_rollout['rollout/extract_reward']:.4f}")
                
                if "rollout/raw_reward" in metrics_rollout:
                    logger.info(f"Total raw reward (weighted sum): {metrics_rollout['rollout/raw_reward']:.4f}")
                
                # 打印每个reward function的均值
                reward_func_names = config.training.get("reward_funcs", "accuracy").split(',')
                reward_func_names = [name.strip() for name in reward_func_names]
                
                # 获取 reward_weights 用于显示
                reward_weights_str = config.training.get("reward_weights", None)
                if reward_weights_str is not None:
                    if isinstance(reward_weights_str, str):
                        reward_weights = [float(w.strip()) for w in reward_weights_str.split(',') if w.strip()]
                    else:
                        reward_weights = list(reward_weights_str)
                else:
                    reward_weights = [1.0] * len(reward_func_names)
                
                logger.info(f"Individual reward functions:")
                for idx, func_name in enumerate(reward_func_names):
                    key = f"rollout/reward_{func_name}"
                    if key in metrics_rollout:
                        weight = reward_weights[idx] if idx < len(reward_weights) else 1.0
                        logger.info(f"  {func_name} (weight={weight}): {metrics_rollout[key]:.4f}")
                
                logger.info(f"{'='*60}\n")
            
            accelerator.log(metrics_rollout, step=optimizer_step)

        except Exception as e:
            logger.warning(f"Failed to compute/log rollout stats: {e}")

        config.training.max_gen_length = config.rollout.max_token

        # 训练种子要设置在采样种子后面，防止每个epoch随机抽取相同的子集题目
        if config.training.seed is not None:
            set_seed(config.training.seed + epoch)

        uni_prompting = UniversalPrompting(
            tokenizer,
            max_prompt_len=config.training.max_prompt_len,
            max_gen_length=config.training.max_gen_length,
            ignore_id=-100
        )

        # 使用 AutoModelForCausalLM 自动根据 config.json 识别模型类型（dense 或 MoE）
        from transformers import AutoModelForCausalLM
        
        # 先不加载训练模型，等ref_logprob计算完再加载

        mask_id = tokenizer.mask_token_id
        pad_id = tokenizer.pad_token_id

        ##################################
        #         内部函数（全部保留）    #
        #################################

        def collapse_k_unique(lst, k: int):
            if k <= 0:
                raise ValueError("k must be > 0")
            uniq = sorted(set(lst))

            mapping = {}
            n = len(uniq)
            for idx, val in enumerate(uniq):
                group = idx // k
                end_idx = min((group + 1) * k - 1, n - 1)
                rep = uniq[end_idx]
                mapping[val] = rep
            return [mapping[x] for x in lst]

        def make_basic_block_attention(
            N: int,
            start_pos: int,            # = L0
            block_size: int,           # = b
        ) -> torch.Tensor:
            B = 1
            L0_ = start_pos
            L1_ = (N - L0_) // 2          # N = L0 + 2·L1
            assert L0_ + 2 * L1_ == N, "input length must be L0 + 2*L1"

            bias = torch.full((B, 1, N, N), 0)

            rows = torch.arange(L0_ + L1_, L0_ + 2 * L1_)              # (L1,)
            rows_token = torch.arange(L0_, L0_ + L1_)                  # (L1,)

            for bi in range((L1_ + block_size - 1) // block_size):
                left_end   = L0_ + min((bi) * block_size, L1_)
                right_start= L0_ + L1_ + (left_end - L0_)

                i_start = bi * block_size
                i_end   = min((bi + 1) * block_size, L1_)

                block_rows = rows[i_start:i_end]
                bias[:, :, block_rows.unsqueeze(-1), 0:left_end] = 1
                bias[:, :, block_rows.unsqueeze(-1), right_start:(right_start + block_size)] = 1

                block_rows = rows_token[i_start:i_end]
                left_end   = L0_ + min((bi + 1) * block_size, L1_)
                bias[:, :, block_rows.unsqueeze(-1), 0:left_end] = 1

            if L0_ > 0:
                num_blocks_pre = (L0_ + block_size - 1) // block_size
                for bi in range(num_blocks_pre):
                    row_end   = max(L0_ - bi * block_size, 0)
                    row_start = max(L0_ - (bi + 1) * block_size, 0)
                    if row_end > row_start:
                        block_rows = torch.arange(row_start, row_end)
                        bias[:, :, block_rows.unsqueeze(-1), 0:row_end] = 1

            return bias        # (B,1,N,N)

        def process_pad(attn, input_ids):
            N = attn.shape[-1]
            device = input_ids.device

            cols = torch.arange(N, device=device)                  # (N,)
            key_mask = (cols < start_pos).unsqueeze(0) & (input_ids == pad_id)  # (B, N)

            attn.masked_fill_(key_mask[:, None, None, :], 0)

            A = attn[:, 0]  # (B, N, N)
            bad = (A.sum(dim=-1) == 0) & (torch.arange(A.size(1), device=A.device).unsqueeze(0) < start_pos)
            if bad.any():
                b, r = bad.nonzero(as_tuple=True)
                A[b, r, :] = 0; A[b, r, r] = 1

            return attn

        def one_round_vectorized(input_ids_b, step_map_b, L0, L1, block_size, mask_id):
            device = input_ids_b.device
            NB = (L1 + block_size - 1) // block_size

            step_pad = torch.full((NB * block_size,), -1, dtype=torch.long, device=device)
            step_pad[:L1] = step_map_b
            step_blk = step_pad.view(NB, block_size)

            valid = step_blk.ge(0)
            big = torch.iinfo(step_blk.dtype).max
            tmp = step_blk.masked_fill(~valid, big)
            min_vals, _ = tmp.min(dim=1, keepdim=True)

            pmask_blk = step_blk.eq(min_vals) & valid
            if not pmask_blk.any():
                return None, None, step_map_b, False

            ge_mask_blk = step_blk.ge(min_vals) & valid

            pmask_tail = pmask_blk.view(-1)[:L1]
            ge_mask_tail = ge_mask_blk.view(-1)[:L1]

            pmask_b = torch.zeros(L0 + L1, dtype=torch.bool, device=device)
            pmask_b[L0:] = pmask_tail

            tail = input_ids_b[L0:L0+L1].clone()
            tail[ge_mask_tail] = mask_id

            extended_input_ids_b = torch.empty(L0 + L1 + L1, dtype=input_ids_b.dtype, device=device)
            extended_input_ids_b[:L0+L1] = input_ids_b
            extended_input_ids_b[L0+L1:] = tail

            new_step_map_b = step_map_b.clone()
            new_step_map_b[pmask_tail] = -1

            return extended_input_ids_b, pmask_b, new_step_map_b, True

        def collect_training_data(input_ids, step_map_list, reward, raw_reward):
            B, L = input_ids.shape
            L0_, L1_ = start_pos, L - start_pos
            block_size = config.training.block_size
            lower, upper = config.training.lower_p, config.training.upper_p

            if config.training.method == "TraceRL":
                if accelerator.is_main_process:
                    print(f"shrink: {config.training.shrink}")
                for b in range(B):
                    step_map_i = step_map_list[b]
                    for j in range(int((L1_ - 1) / block_size) + 1):
                        start = j * block_size; end = min(L1_, (j + 1) * block_size)
                        step_map_list[b][start:end] = collapse_k_unique(step_map_i[start:end], config.training.shrink)

                step_map = torch.as_tensor(step_map_list, dtype=torch.long)
                assert step_map.shape[1] == L1_

                extended_input_ids_list, pmask_list, reward_list_, raw_reward_list_ = [], [], [], []
                for b in range(B):
                    step_b = step_map[b]
                    while True:
                        out = one_round_vectorized(
                            input_ids_b=input_ids[b],
                            step_map_b=step_b,
                            L0=L0_,
                            L1=L1_,
                            block_size=block_size,
                            mask_id=mask_id,
                        )
                        extended_b, pmask_b, step_b, has_any = out
                        if not has_any: break
                        extended_input_ids_list.append(extended_b)
                        pmask_list.append(pmask_b)
                        reward_list_.append(reward[b])
                        raw_reward_list_.append(raw_reward[b])
            else:
                raise ValueError(f"Unknown training.method: {config.training.method}")

            extended_input_ids = torch.stack(extended_input_ids_list, dim=0)
            p_mask = torch.stack(pmask_list, dim=0).to(torch.bool)

            pad_resp = (extended_input_ids[:, :L] == pad_id) & p_mask
            if config.training.post_num is not None:
                cum_pad = torch.cumsum(pad_resp.int(), dim=1)
                p_mask &= ~(pad_resp & (cum_pad > config.training.post_num))

            labels = extended_input_ids[:, :L].clone()

            idx = torch.arange(L).unsqueeze(0).expand(extended_input_ids.shape[0], -1)
            valid = (idx >= start_pos) | extended_input_ids[:, :L].ne(pad_id)
            tok_idx = valid.long().cumsum(dim=-1) - 1
            tok_idx = tok_idx.masked_fill(~valid, 1)
            tok_idx_resp = tok_idx[:, start_pos:]
            tok_idx_ext = torch.cat([tok_idx, tok_idx_resp], dim=1)

            keep = p_mask.view(p_mask.size(0), -1).any(dim=1)
            keep_idx = keep.nonzero(as_tuple=True)[0]

            extended_input_ids = extended_input_ids[keep_idx]
            p_mask            = p_mask[keep_idx]
            tok_idx_ext       = tok_idx_ext[keep_idx]
            labels            = labels[keep_idx]

            reward_kept = [reward_list_[i] for i in keep_idx.tolist()]
            raw_reward_kept = [raw_reward_list_[i] for i in keep_idx.tolist()]
            if not reward_kept:
                print(f"collect_training_data: reward_kept is empty")

            return extended_input_ids, p_mask, tok_idx_ext, labels, reward_kept, raw_reward_kept

        # 旧 logprob 计算（批量版）
        @torch.no_grad()
        def compute_logp_old_tok_batch(
            extended_input_ids: torch.Tensor,
            p_mask: torch.Tensor,
            tok_idx_ext: torch.Tensor,
            labels: torch.Tensor,
            basic_block_attention: torch.Tensor,
        ) -> torch.Tensor:
            """
            返回形状 (B, L0+L1) 的旧策略 logp_tok（只在 p_mask 位置有效，其它位置值无所谓）
            """
            model.eval()
            B, L = p_mask.shape
            L0_ = start_pos
            L1_ = L - L0_

            attn = basic_block_attention.clone().repeat_interleave(B, dim=0).to(extended_input_ids.device)
            attn = process_pad(attn, extended_input_ids)

            logits = model(
                input_ids=extended_input_ids,
                attention_mask=attn,
                position_ids=tok_idx_ext
            ).logits
            logits = torch.cat([logits[:, :L0_, :], logits[:, L0_ + L1_ :, :]], dim=1)  # (B, L0+L1, V)

            log_probs = F.log_softmax(logits, dim=-1)
            logp_tok  = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            model.train()
            return logp_tok
        
        # 参考模型 logprob 计算（用于KL散度）
        @torch.no_grad()
        def compute_ref_logprob_batch(
            ref_model_instance,
            extended_input_ids: torch.Tensor,
            p_mask: torch.Tensor,
            tok_idx_ext: torch.Tensor,
            labels: torch.Tensor,
            basic_block_attention: torch.Tensor,
        ) -> torch.Tensor:
            """
            使用参考模型计算 logp_tok，用于 KL 散度计算
            返回形状 (B, L0+L1)
            """
            B, L = p_mask.shape
            L0_ = start_pos
            L1_ = L - L0_

            attn = basic_block_attention.clone().repeat_interleave(B, dim=0).to(extended_input_ids.device)
            attn = process_pad(attn, extended_input_ids)

            logits = ref_model_instance(
                input_ids=extended_input_ids,
                attention_mask=attn,
                position_ids=tok_idx_ext
            ).logits
            logits = torch.cat([logits[:, :L0_, :], logits[:, L0_ + L1_ :, :]], dim=1)  # (B, L0+L1, V)

            log_probs = F.log_softmax(logits, dim=-1)
            logp_tok  = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            return logp_tok

        ##################################
        #        准备训练输入与注意力     #
        ##################################
        # 把 rollout 数据展开为 prompt/response 序列
        prompt_list, response_list, step_map_list, reward_list, raw_reward_list = [], [], [], [], []
        for x in dataset_load:
            prompt_list.extend(x["prompt"])
            response_list.extend(x["response"])
            reward_list.extend(x["rewards"])
            raw_reward_list.extend(x["raw_rewards"])

        input_ids_lm, _, start_pos, drop_num = uni_prompting((prompt_list, response_list))

        _, L = input_ids_lm.shape
        L0, L1 = start_pos, L - start_pos

        # 将 step_map 统一成每条样本一个 list[int]
        for x in dataset_load:
            if "step_map" not in x.keys() or x['step_map'] is None:
                step_map_list.extend([[j for j in range(L1)] * len(x["prompt"])])
            else:
                for step_map_i in x["step_map"]:
                    if len(step_map_i) > L1:
                        step_map_i = step_map_i[:L1]
                    else:
                        step_map_i = step_map_i + [max(step_map_i) + 1] * (L1 - len(step_map_i))
                    step_map_list.append(step_map_i)

        # 固定的 block 注意力（N = L0 + 2*L1）
        basic_block_attention = make_basic_block_attention(L0 + 2 * L1, start_pos, config.training.block_size).cpu()

        ##################################
        #     forward_process（保留）     #
        ##################################
        def forward_process(extended_input_ids, p_mask, tok_idx_ext, labels, adv, logp_old_tok, ref_logprob=None, is_final_update=False, micro_batch_size=None):
            # Number of iterations over the same data (like GRPO)
            num_iterations_ = config.training.get("num_iterations", 8)

            adv = torch.as_tensor(adv, device=extended_input_ids.device).detach()

            B, Lp = p_mask.shape
            L0_ = start_pos
            L1_ = Lp - L0_
            device = extended_input_ids.device

            # Process in micro-batches if specified
            if micro_batch_size is not None and micro_batch_size < B:
                all_logits = []
                
                for i in range(0, B, micro_batch_size):
                    end_idx = min(i + micro_batch_size, B)
                    micro_B = end_idx - i
                    
                    micro_extended_input_ids = extended_input_ids[i:end_idx]
                    micro_tok_idx_ext = tok_idx_ext[i:end_idx]
                    
                    micro_attention_mask = basic_block_attention.clone()
                    micro_attention_mask = micro_attention_mask.repeat_interleave(micro_B, dim=0).to(device)
                    micro_attention_mask = process_pad(micro_attention_mask, micro_extended_input_ids)
                    
                    micro_logits = model(input_ids=micro_extended_input_ids, attention_mask=micro_attention_mask, position_ids=micro_tok_idx_ext).logits
                    micro_logits = torch.cat([micro_logits[:, :L0_, :], micro_logits[:, L0_ + L1_:, :]], dim=1)
                    
                    all_logits.append(micro_logits)
                
                logits = torch.cat(all_logits, dim=0)
            else:
                attention_mask = basic_block_attention.clone()
                attention_mask = attention_mask.repeat_interleave(B, dim=0).to(device)
                attention_mask = process_pad(attention_mask, extended_input_ids)

                logits = model(input_ids=extended_input_ids, attention_mask=attention_mask, position_ids=tok_idx_ext).logits
                logits = torch.cat([logits[:, :L0_, :], logits[:, L0_ + L1_:, :]], dim=1)  # (B, L0+L1, V)

            log_probs = F.log_softmax(logits, dim=-1)
            logp_new_tok = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # (B, T)
            
            if logp_old_tok is None:
                logp_old_tok = logp_new_tok.detach()
            else:
                logp_old_tok = logp_old_tok.to(device)

            ratio = logp_new_tok - logp_old_tok
            ratio = torch.where(p_mask, ratio, torch.zeros_like(ratio)).clamp(-10.0, 10.0)
            ratio = torch.exp(ratio)  # (B, T)
            # print(ratio)
            clipped = torch.clamp(ratio, 1 - config.training.eps, 1 + config.training.eps + 0.08)  # (B, T)

            adv_tok = adv.unsqueeze(1)
            surrogate_tok = torch.min(ratio * adv_tok, clipped * adv_tok)  # (B, T)
            surrogate_tok = surrogate_tok * p_mask

            num_mask = torch.clamp(p_mask.sum(dim=1), min=1)
            # surrogate_tok = surrogate_tok.sum(dim=1) / L1_
            surrogate_tok = surrogate_tok.sum(dim=1) / num_mask
            if accelerator.is_main_process:
                print(f"num_mask: {num_mask.tolist()}, L1: {L1_}")

            policy_loss = -(surrogate_tok.sum())

            # KL penalty（只使用参考模型的logprob和当前模型的logprob）
            kl_loss = torch.tensor(0.0, device=policy_loss.device)
            if config.training.beta > 0 and ref_logprob is not None:
                # KL散度 = KL(new||ref) = E[log_new - log_ref] = new_logprob - ref_logprob
                ref_logprob = ref_logprob.to(device)
                kl_seq = logp_new_tok - ref_logprob
                
                kl_seq = torch.where(p_mask, kl_seq, torch.zeros_like(kl_seq))
                if config.training.use_kl_estimator_k3:
                    t = (-kl_seq).clamp(-10.0, 10.0)
                    kl_seq = t.exp() - 1.0 + kl_seq
                # kl_seq = (kl_seq * p_mask).sum(dim=1) / L1_
                kl_seq = (kl_seq * p_mask).sum(dim=1) / num_mask
                kl_loss = config.training.beta * kl_seq.sum()
                total_loss = policy_loss + kl_loss
            else:
                total_loss = policy_loss
            # time_start = time.time()
            # 节省显存的写法（减少 needless allocation），避免 probs 的中间展开
            
            with torch.no_grad():
                probs = torch.exp(log_probs)
                entropy = -(probs * log_probs).sum(dim=-1)
                entropy_masked = (entropy * p_mask).sum(dim=1) / num_mask

            # time_end = time.time()
            # print(f"entropy time: {time_end - time_start}")
            return total_loss, kl_loss, entropy_masked

        ##################################
        #         训练循环（动态扩展）    #
        ##################################
        from tqdm.auto import tqdm

        logger.info("***** Running training (dynamic per-batch expansion) *****")
        logger.info(f"  Num response = {len(dataset_load)}")
        logger.info(f"  Num sample dropped = {drop_num}")
        logger.info(f"  Num original samples = {input_ids_lm.shape[0]}")
        num_train_epochs = config.training.num_train_epochs
        num_iterations = config.training.get("num_iterations", 8)
        logger.info(f"  Number of iterations over same data: {num_iterations}")

        total_samples = input_ids_lm.shape[0]
        batch_size_train = config.training.batch_size_lm

        ##################################
        #   预处理：收集所有扩展数据并计算 old_logprob 和 ref_logprob   #
        ##################################
        logger.info("***** Preprocessing: collecting training data and computing old_logprob & ref_logprob *****")
        
        old_logprob_batches = []
        ref_logprob_batches = []
        extended_input_ids_batches = []
        p_mask_batches = []
        tok_idx_ext_batches = []
        labels_batches = []
        rewards_batches = []
        raw_rewards_batches = []

        indices = list(range(total_samples))
        num_batches = math.ceil(total_samples / batch_size_train)

        # 收集所有扩展后的数据
        logger.info(f"[Rank {accelerator.process_index}] Collecting training data: {num_batches} batches")
        for bstep in range(num_batches):
            start_b = bstep * batch_size_train
            end_b = min(start_b + batch_size_train, total_samples)
            batch_idx = indices[start_b:end_b]
            # 动态扩展
            input_ids_batch = input_ids_lm[batch_idx]
            step_map_batch = [step_map_list[i] for i in batch_idx]
            rewards_batch = [reward_list[i] for i in batch_idx]
            raw_rewards_batch = [raw_reward_list[i] for i in batch_idx]

            extended_input_ids, p_mask, tok_idx_ext, labels, rewards, raw_rewards = collect_training_data(
                input_ids_batch, step_map_batch, rewards_batch, raw_rewards_batch
            )
            extended_input_ids_batches.append(extended_input_ids)
            p_mask_batches.append(p_mask)
            tok_idx_ext_batches.append(tok_idx_ext)
            labels_batches.append(labels)
            rewards_batches.append(rewards)
            raw_rewards_batches.append(raw_rewards)

        # Step 1: 加载/移动参考模型并计算 ref_logprob（用于KL散度）
        if config.training.beta > 0:
            logger.info("=" * 80)
            logger.info("Step 1: Loading/moving reference model and computing ref_logprob")
            logger.info("=" * 80)
            
            # 只在第一个epoch加载参考模型
            if ref_model is None:
                # 如果是第一个epoch且不是续训，ref_model和model相同，可以先加载一个然后克隆
                if epoch == config.experiment.current_epoch and config.experiment.current_epoch == 1:
                    logger.info(f"[Rank {accelerator.process_index}] Loading model from {pretrained_model} for both training and reference")
                    
                    if dist.get_rank() == 0:
                        model = AutoModelForCausalLM.from_pretrained(pretrained_model, trust_remote_code=True, torch_dtype="auto")
                    dist.barrier()
                    if dist.get_rank() != 0:
                        model = AutoModelForCausalLM.from_pretrained(pretrained_model, trust_remote_code=True, torch_dtype="auto")
                    
                    if hasattr(model, "config"):
                        model.config.fuse_cross_entropy = False
                    
                    # 克隆模型作为参考模型
                    import copy
                    logger.info(f"[Rank {accelerator.process_index}] Cloning model as reference model")
                    ref_model = copy.deepcopy(model)
                    ref_model.eval()
                    # 冻结参考模型参数
                    for param in ref_model.parameters():
                        param.requires_grad = False
                    
                    # 训练模型设置
                    if config.training.gradient_checkpointing_enable:
                        model.gradient_checkpointing_enable()
                        if hasattr(model, "config"):
                            model.config.use_cache = False
                    
                    model_already_loaded = True
                else:
                    # 续训或后续epoch，正常加载参考模型
                    ref_model_path = config.model.pretrained_model
                    logger.info(f"[Rank {accelerator.process_index}] Loading reference model from {ref_model_path} (first time)")
                    
                    if dist.get_rank() == 0:
                        ref_model = AutoModelForCausalLM.from_pretrained(ref_model_path, trust_remote_code=True, torch_dtype="auto")
                    dist.barrier()
                    if dist.get_rank() != 0:
                        ref_model = AutoModelForCausalLM.from_pretrained(ref_model_path, trust_remote_code=True, torch_dtype="auto")
                    
                    if hasattr(ref_model, "config"):
                        ref_model.config.fuse_cross_entropy = False
                    ref_model.eval()
                    # 冻结参考模型参数
                    for param in ref_model.parameters():
                        param.requires_grad = False
                    
                    model_already_loaded = False
            else:
                logger.info(f"[Rank {accelerator.process_index}] Moving reference model from CPU to GPU")
                model_already_loaded = False
            
            # 移动到GPU
            ref_model = ref_model.to(accelerator.device)
            
            logger.info(f"[Rank {accelerator.process_index}] Computing ref_logprob for {num_batches} batches")
            for bstep in range(num_batches):
                ref_logprob = compute_ref_logprob_batch(
                    ref_model,
                    extended_input_ids_batches[bstep].to(accelerator.device),
                    p_mask_batches[bstep].to(accelerator.device),
                    tok_idx_ext_batches[bstep].to(accelerator.device),
                    labels_batches[bstep].to(accelerator.device),
                    basic_block_attention.to(accelerator.device)
                )
                ref_logprob_batches.append(ref_logprob.cpu())  # 移到CPU节省显存
            
            accelerator.wait_for_everyone()
            logger.info(f"[Rank {accelerator.process_index}] Reference logprob computation completed")
            
            # 移动参考模型到CPU释放显存
            logger.info(f"[Rank {accelerator.process_index}] Moving reference model to CPU to free GPU memory")
            ref_model = ref_model.cpu()
            torch.cuda.empty_cache()
            accelerator.wait_for_everyone()
            logger.info(f"[Rank {accelerator.process_index}] Reference model moved to CPU")
        else:
            # 如果不需要计算 ref_logprob，设为 None
            logger.info("KL penalty disabled (beta=0), skipping ref_logprob computation")
            ref_logprob_batches = [None] * num_batches
            model_already_loaded = False
        
        # Step 2: 加载训练模型并计算 old_logprob（用于PPO的ratio）
        logger.info("=" * 80)
        logger.info("Step 2: Loading training model and computing old_logprob")
        logger.info("=" * 80)
        
        # 如果model还没有加载（在Step 1中没有被克隆），现在加载
        if epoch == config.experiment.current_epoch and not model_already_loaded:
            logger.info(f"[Rank {accelerator.process_index}] Loading training model from {pretrained_model}")
            if dist.get_rank() == 0:
                model = AutoModelForCausalLM.from_pretrained(pretrained_model, trust_remote_code=True, torch_dtype="auto")
            dist.barrier()
            if dist.get_rank() != 0:
                model = AutoModelForCausalLM.from_pretrained(pretrained_model, trust_remote_code=True, torch_dtype="auto")

            # calculate loss ourselves, needs logits，so aviod fuse CE
            if hasattr(model, "config"):
                model.config.fuse_cross_entropy = False

            # Setup gradient checkpointing
            if config.training.gradient_checkpointing_enable:
                model.gradient_checkpointing_enable()
                if hasattr(model, "config"):
                    model.config.use_cache = False
            else:
                model = model.to(accelerator.device)
        elif epoch == config.experiment.current_epoch and model_already_loaded:
            logger.info(f"[Rank {accelerator.process_index}] Training model already loaded (cloned from reference model)")
            # 模型已经加载了，只需要移到GPU（如果还没在GPU上）
            if not config.training.gradient_checkpointing_enable:
                model = model.to(accelerator.device)
        else:
            # 后续epoch，从CPU移到GPU
            model = model.to(accelerator.device)


        ##################################
        #   Optimizer and LR scheduler   #
        #################################
        if epoch == config.experiment.current_epoch:
            logger.info("Initializing optimizer and LR scheduler")
            optimizer_config = config.optimizer.params

            # no decay on bias and layernorm and embedding
            no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if
                               p.requires_grad and not any(nd in n for nd in no_decay)],
                    "weight_decay": optimizer_config.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                               p.requires_grad and any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_type = config.optimizer.name
            if optimizer_type == "adamw":
                optimizer = AdamW(
                    optimizer_grouped_parameters,
                    lr=optimizer_config.learning_rate,
                    betas=(optimizer_config.beta1, optimizer_config.beta2),
                    weight_decay=optimizer_config.weight_decay,
                    eps=optimizer_config.epsilon,
                )
            else:
                raise ValueError(f"Optimizer {optimizer_type} not supported")

        ##################################
        #   LR Scheduler & Accelerator   #
        ##################################
        num_batches_train = math.ceil(input_ids_lm.shape[0] / config.training.batch_size_lm)
        max_train_steps = num_batches_train * num_iterations * num_train_epochs
        if epoch == config.experiment.current_epoch:
            logger.info("Initializing LR scheduler and preparing with accelerator")
            lr_scheduler = get_scheduler(
                config.lr_scheduler.scheduler,
                optimizer=optimizer,
                num_training_steps=max_train_steps,
                num_warmup_steps=config.lr_scheduler.params.warmup_steps,
                min_lr_scale=config.lr_scheduler.params.min_lr_scale
            )

            dummy_dataset = torch.utils.data.TensorDataset(torch.zeros(1, 1, dtype=torch.long))
            dummy_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=1)
            # 注意：lr_scheduler 不要被 accelerator.prepare() 包装，否则会导致 current_step 累加 world_size 倍
            model, optimizer, dummy_loader = accelerator.prepare(model, optimizer, dummy_loader)
        else:
            dummy_dataset = torch.utils.data.TensorDataset(torch.zeros(1, 1, dtype=torch.long))
            dummy_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=1)
            dummy_loader = accelerator.prepare(dummy_loader)

        # 计算 old_logprob（在多轮迭代的情况下，用于PPO ratio计算）
        # 放在 prepare 之后，这样 model 已经在正确的设备上了
        if num_iterations > 1:
            logger.info(f"[Rank {accelerator.process_index}] Computing old_logprob for {num_batches} batches")
            model.eval()
            for bstep in range(num_batches):
                old_logprob = compute_logp_old_tok_batch(
                    extended_input_ids_batches[bstep].to(accelerator.device),
                    p_mask_batches[bstep].to(accelerator.device),
                    tok_idx_ext_batches[bstep].to(accelerator.device),
                    labels_batches[bstep].to(accelerator.device),
                    basic_block_attention.to(accelerator.device)
                )
                old_logprob_batches.append(old_logprob.cpu())  # 移到CPU节省显存
            
            accelerator.wait_for_everyone()
            logger.info(f"[Rank {accelerator.process_index}] Old logprob computation completed")
        else:
            # 如果只有一轮迭代，old_logprob 会在 forward_process 中动态计算
            old_logprob_batches = [None] * num_batches
        
        # 确保模型处于训练模式
        model.train()

        logger.info(f"[Rank {accelerator.process_index}] Preprocessing completed. Ready for training.")

        ##################################
        #         训练循环           #
        ##################################
        for iteration in range(num_iterations):
            logger.info(f"Iteration {iteration+1}/{num_iterations}")
            
            progress_bar = tqdm(range(num_batches),
                                desc=f"Epoch {epoch}/{num_train_epochs} - Iter {iteration+1}/{num_iterations}",
                                disable=not accelerator.is_local_main_process,
                                dynamic_ncols=True,
                                leave=True)

            optimizer.zero_grad(set_to_none=True)
            
            # 用于累积所有batch的指标
            total_loss = 0.0
            total_raw_reward = 0.0
            total_entropy = 0.0
            total_kl = 0.0
            num_steps = 0

            for bstep in progress_bar:
                extended_input_ids = extended_input_ids_batches[bstep]
                p_mask = p_mask_batches[bstep]
                tok_idx_ext = tok_idx_ext_batches[bstep]
                labels = labels_batches[bstep]
                rewards = rewards_batches[bstep]
                raw_rewards = raw_rewards_batches[bstep]
                
                # 如果是单轮迭代且 old_logprob 未计算，传入 None 让 forward_process 动态计算
                old_lp = old_logprob_batches[bstep]
                if old_lp is not None:
                    old_lp = old_lp.to(accelerator.device)
                
                # 获取 ref_logprob（用于 KL 散度计算）
                ref_lp = ref_logprob_batches[bstep]
                if ref_lp is not None:
                    ref_lp = ref_lp.to(accelerator.device)
                
                micro_batch_size = getattr(config.training, 'micro_batch_size', None)
                
                loss_lm, kl_loss, entropy = forward_process(
                    extended_input_ids=extended_input_ids.to(accelerator.device),
                    p_mask=p_mask.to(accelerator.device),
                    tok_idx_ext=tok_idx_ext.to(accelerator.device),
                    labels=labels.to(accelerator.device),
                    adv=rewards,
                    logp_old_tok=old_lp,
                    ref_logprob=ref_lp,
                    is_final_update=True,
                    micro_batch_size=micro_batch_size
                )
                # print(loss_lm.item(), total_samples)
                accelerator.backward(loss_lm / total_samples) # 每个rank额外归一化

                global_step += 1
                
                # 累积指标
                total_loss += loss_lm.item()
                total_raw_reward += float(sum(raw_rewards)) / len(raw_rewards)
                total_entropy += entropy.mean().item()
                total_kl += kl_loss.item()
                num_steps += 1

                # 每个 batch 都打印
                if accelerator.is_main_process:
                    logger.info(
                        f"[global_step {global_step}] Loss={loss_lm.item():.6f} "
                        f"| Reward={float(sum(raw_rewards))/len(raw_rewards):.4f} "
                        f"| Ent={entropy.mean().item():.4f} | KL={kl_loss.item():.6f}"
                    )
                
                # 删除变量并清理GPU缓存，防止OOM
                del loss_lm, kl_loss, entropy, extended_input_ids, p_mask, tok_idx_ext, labels, old_lp, ref_lp
                torch.cuda.empty_cache()

            if config.training.max_grad_norm is not None:
                accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            optimizer_step += 1

            # === Step 级别 log（所有batches的平均值）===
            log_dict = {
                "train/loss": total_loss / total_samples,
                "train/raw_reward": total_raw_reward / num_steps,
                "train/entropy": total_entropy / total_samples,
                "train/kl": total_kl / total_samples,
                "train/learning_rate": lr_scheduler.get_last_lr()[0],
                "train/epoch": epoch,
                "train/global_step": global_step,
                "train/optimizer_step": optimizer_step,
            }
            accelerator.log(log_dict, step=optimizer_step)
            logger.info(
                f"Step {optimizer_step} | "
                f"Loss: {log_dict['train/loss']:.8f} | "
                f"Raw Reward: {log_dict['train/raw_reward']:.8f} | "
                f"Entropy: {log_dict['train/entropy']:.8f} | "
                f"KL: {log_dict['train/kl']:.8f} | "
                f"LR: {log_dict['train/learning_rate']:.2e} | "
                f"Epoch: {log_dict['train/epoch']} | "
                f"Global Step: {log_dict['train/global_step']}"
            )

            # accelerator.wait_for_everyone()

            # === Step 级别保存 ===
            save_every_steps = config.training.get("save_steps", None)
            if save_every_steps and optimizer_step % save_every_steps == 0:
                logger.info(f"Saving checkpoint at optimizer_step {optimizer_step}")
                save_checkpoint(
                    model, tokenizer, config, accelerator,
                    f"epoch-{epoch}-step-{optimizer_step}"
                )

        accelerator.wait_for_everyone()

        logger.info(f"Epoch {epoch} training completed")
        logger.info(f"Total optimizer updates: {optimizer_step}")

        # # save checkpoint at the end of each epoch
        # logger.info(f"Saving checkpoint: {config.model.optimized_name}")
        # save_checkpoint(model, tokenizer, config, accelerator, config.model.optimized_name)

        if epoch % config.experiment.save_every == 0:
            logger.info(f"Saving epoch checkpoint: epoch-{epoch}")
            save_checkpoint(model, tokenizer, config, accelerator, f"epoch-{epoch}")

        # Evaluation
        eval_every = config.experiment.get("eval_every", 1)
        if epoch % eval_every == 0:
            logger.info(f"Starting evaluation on {config.dataset.eval_dataset}...")
            eval_model_path = config.experiment.prefix_dir + project_name + f"/ckpt/epoch-{epoch}"
            eval_data = rollout_sampling_eval(
                config.dataset.eval_dataset, epoch, config,
                eval_model_path, tokenizer, accelerator,
                normalize=False, filter=False, mode="eval"
            )

            total_correct = 0
            total_samples_eval = 0
            for item in eval_data:
                total_correct += sum(item["correctness"])
                total_samples_eval += len(item["correctness"])

            accuracy = total_correct / total_samples_eval if total_samples_eval > 0 else 0.0
            logger.info(f"Evaluation accuracy: {accuracy*100:.2f}% ({total_correct}/{total_samples_eval})")

            accelerator.log({
                "eval/accuracy": accuracy,
                "eval/total_correct": total_correct,
                "eval/total_samples": total_samples_eval,
                "eval/epoch": epoch,
            }, step=optimizer_step)

        logger.info(f"Epoch {epoch} completed successfully\n")
        accelerator.wait_for_everyone()

    # End training after all epochs are complete
    logger.info(f"{'='*80}")
    logger.info(f"All training completed! Total epochs: {config.training.num_train_epochs}")
    logger.info(f"{'='*80}")
    accelerator.end_training()



def main():
    train()

def save_checkpoint(model, tokenizer, config, accelerator, name):

    from pathlib import Path
    import time, json, shutil, os, glob

    output_dir = Path(config.experiment.project)
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator.wait_for_everyone()
    save_base = output_dir / "ckpt"
    save_base.mkdir(exist_ok=True)

    model_to_save = accelerator.unwrap_model(model)

    if accelerator.is_main_process:
        save_dir = save_base / name

        # 直接从 model_to_save 抽取权重，避免 get_state_dict 丢参数
        model_to_save.save_pretrained(
            save_dir,
            save_function=accelerator.save,
            safe_serialization=True,
        )

        tokenizer.save_pretrained(save_dir)

        # 保存当前 config，而不是覆盖旧的
        model_to_save.config.save_pretrained(save_dir)

        # 复制自定义源码文件
        model_path = config.model.pretrained_model
        for pattern in ("modeling_*.py", "configuration_*.py", "tokenization_*.py", "processing_*.py"):
            for fn in glob.glob(os.path.join(model_path, pattern)):
                dst = os.path.join(save_dir, os.path.basename(fn))
                if not os.path.exists(dst):
                    shutil.copy2(fn, dst)

        # 记录元数据
        metadata = {
            "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "step": getattr(config, "global_step", None)
        }
        with (save_base / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Saved checkpoint to {save_dir}")

    accelerator.wait_for_everyone()



if __name__ == "__main__":
    main()
