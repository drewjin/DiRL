import os
import sys
from typing import Any
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import math
import random
import re
from collections import Counter
from transformers import AutoTokenizer, AutoConfig
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import load_dataset
import torch.distributed as dist
from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig
from lmdeploy.pytorch.tools.utils import Timer, visualize_pipe_out

from sample_api import sample
from orm import orms


from utils.data_chunk import get_data_chunk
from utils.math_utils import equation, extract_last_boxed
from train.utils import get_config

from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig, ChatTemplateConfig
from lmdeploy.pytorch.tools.utils import Timer, visualize_pipe_out
from accelerate import Accelerator, utils
import accelerate
from transformers import AutoTokenizer
import re

EXTRACT_FAILED="Can not extract the answer!"
def extract_final_boxed_answer(s: str):
    tag = r'boxed{'
    start = s.rfind(tag)
    if start == -1:
        return "Can not extract the answer!"

    i = start + len(tag)
    depth = 1
    buf = []
    while i < len(s) and depth:
        ch = s[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                break
        buf.append(ch)
        i += 1
    return ''.join(buf) if depth == 0 else "Can not extract the answer!"


def z_score_normalize(lst):
    mean = sum(lst) / len(lst)
    std = (sum((x - mean) ** 2 for x in lst) / len(lst)) ** 0.5
    return [(x - mean) / std if std != 0 else 0 for x in lst]

# 全局缓存字典，用于存储已加载的数据集
_DATASET_CACHE = {}
# 全局缓存字典，用于存储每个数据集的打乱索引
_SHUFFLED_INDICES_CACHE = {}

def rollout_sampling(client, dataset_name, epoch, config, model_path, model_name, tokenizer, accelerator: Accelerator, reward_function=None, normalize=True, filter=True, mode="train", split_rank=False, reward_funcs="accuracy", data_cursor=0, dynamic_threshold=None):
    # 顺序遍历模式：通过 data_cursor 参数记录数据集遍历位置
    # 每个epoch按顺序遍历 num_task_per_step * sampling_multiplier 的数据
    # 从 data_cursor 位置开始，返回时更新 cursor
    
    project_name = config.experiment.project

    reward = config.dataset.data_type

    # Parse reward_funcs (comma-separated string)
    reward_func_list = [func.strip() for func in reward_funcs.split(',') if func.strip()]
    
    # Parse reward_weights (comma-separated string or list)
    if not hasattr(config.training, 'reward_weights') or config.training.reward_weights is None:
        reward_weights = [1.0] * len(reward_func_list)
    elif isinstance(config.training.reward_weights, str):
        reward_weights = [float(w.strip()) for w in config.training.reward_weights.split(',') if w.strip()]
    else:
        reward_weights = list(config.training.reward_weights)
    
    if len(reward_weights) != len(reward_func_list):
        reward_weights = [1.0] * len(reward_func_list)

    # Initialize ORMs (keep all for backward compatibility)
    math_judge = orms['accuracy']()
    cosine_reward = orms['cosine'](cosine_max_len=config.rollout.max_token, accuracy_orm=math_judge)
    speed_reward = orms['speed'](speed_max_len=config.rollout.block_size, accuracy_orm=math_judge)
    speed_penalty = orms['speed_penalty'](speed_max_len=config.rollout.block_size, accuracy_orm=math_judge)
    repetition_reward = orms['repetition']()
    format_reward = orms['format']()
    boxed_reward = orms['boxed']()
    binary_reward = orms['binary']()
    soft_overlong_reward = orms['soft_overlong'](config.rollout.max_token, config.rollout.max_token//2)
    
    # Initialize ORM instances for reward_func_list
    orm_instances = {}
    for func_name in reward_func_list:
        if func_name == 'accuracy':
            orm_instances[func_name] = math_judge
        elif func_name == 'cosine':
            orm_instances[func_name] = cosine_reward
        elif func_name == 'speed':
            orm_instances[func_name] = speed_reward
        elif func_name == 'speed_penalty':
            orm_instances[func_name] = speed_penalty
        elif func_name == 'repetition':
            orm_instances[func_name] = repetition_reward
        elif func_name == 'format':
            orm_instances[func_name] = format_reward
        elif func_name == 'boxed':
            orm_instances[func_name] = boxed_reward
        elif func_name == 'binary':
            orm_instances[func_name] = binary_reward
        elif func_name == "soft_overlong":
            orm_instances[func_name] = soft_overlong_reward
        else:
            # Try to instantiate directly from orms
            try:
                orm_instances[func_name] = orms[func_name]()
            except Exception as e:
                pass
                    
    def compute_reward_from_funcs(text, gt, token_ids=None, step_map=None):
        """Compute reward by calling each ORM function and summing results"""
        total_reward = 0.0
        reward_components = {}
        
        for func_name in reward_func_list:
            if func_name not in orm_instances:
                continue
                
            orm = orm_instances[func_name]
            try:

                score = orm(completions=[text], solution=[gt], response_token_ids=[token_ids], max_tokens=config.rollout.max_token, step_map=[step_map] if step_map is not None else None)[0]
            
                reward_components[func_name] = score
                total_reward += score
            except Exception as e:
                reward_components[func_name] = 0.0
        
        return total_reward, reward_components


    # config.
    if mode == "train":
        gen_config = GenerationConfig(
            top_p=config.rollout.top_p,
            top_k=config.rollout.top_k,
            temperature=config.rollout.temperature,
            do_sample=config.rollout.do_sample,
            max_new_tokens=config.rollout.max_token,
            # min_new_tokens=128,
            skip_special_tokens=False,
            # random_seed=config.training.seed + epoch,
        )

    start_with_think = config.rollout.start_with_think

    # Load data (只在第一次加载，后续epoch直接使用缓存)
    global _DATASET_CACHE
    cache_key = f"{dataset_name}"
    
    if cache_key not in _DATASET_CACHE:
        # 第一次加载：rank0 构建缓存，其余等待后读取
        if accelerator.is_main_process:
            dataset = load_dataset("json", data_files=f"data/{dataset_name}.json")
            _DATASET_CACHE[cache_key] = dataset
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            dataset = load_dataset("json", data_files=f"data/{dataset_name}.json")
            _DATASET_CACHE[cache_key] = dataset
    else:
        # 后续epoch：直接使用缓存
        dataset = _DATASET_CACHE[cache_key]

    ds = dataset['train']
    prompts_all = ds['question']
    gts_all = ds['ground_truth_answer']

    reason_prompt = "<|im_start|>user\n{problem}\nPlease reason step by step, and put your final answer within $\\boxed{{}}$.<|im_end|>\n<|im_start|>assistant\n"

    if start_with_think:
        reason_prompt = "<|im_start|>system\nPlease reason step by step, and put your final answer within $\\boxed{{}}$.<|im_end|>\n<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n<think>"

    # Build pool of all (idx, (prompt, gt))
    all_data = list(enumerate(zip(prompts_all, gts_all)))
    
    # 初始化或获取打乱的索引
    global _SHUFFLED_INDICES_CACHE
    indices_cache_key = f"{dataset_name}_indices"
    
    if indices_cache_key not in _SHUFFLED_INDICES_CACHE:
        # 首次初始化：创建打乱的索引
        if accelerator.is_main_process:
            shuffled_indices = list(range(len(all_data)))
            random.shuffle(shuffled_indices)
            _SHUFFLED_INDICES_CACHE[indices_cache_key] = shuffled_indices
        accelerator.wait_for_everyone()
        # 其他进程也需要相同的打乱顺序，通过broadcast同步
        if not accelerator.is_main_process:
            _SHUFFLED_INDICES_CACHE[indices_cache_key] = None
        from accelerate.utils import broadcast_object_list
        shuffled_indices = broadcast_object_list([_SHUFFLED_INDICES_CACHE[indices_cache_key]], from_process=0)[0]
        _SHUFFLED_INDICES_CACHE[indices_cache_key] = shuffled_indices

    # How many tasks to finally keep (global target)
    if mode == "train":
        if config.rollout.num_task_per_step == -1:
            target_total = len(all_data)
        else:
            target_total = min(config.rollout.num_task_per_step, len(all_data))
    else:
        target_total = len(all_data)

    # 顺序遍历策略：每个epoch按顺序遍历数据集，不打乱
    # 从 data_cursor 位置开始，遍历完后从头循环
    world_size = accelerator.num_processes
    rank = accelerator.process_index
    
    # 根据数据集确定采样倍数（考虑DAPO过滤率）
    if "GSM8K" in dataset_name:
        sampling_multiplier = 6
    elif "BigMath" in dataset_name:
        sampling_multiplier = 3
    else:
        sampling_multiplier = 2
    # 测试，不过滤
    # sampling_multiplier = 1
    if config.rollout.num_response_per_task >= 16:
        sampling_multiplier = 2
    
    if not filter:
        sampling_multiplier = 1

    total_to_sample = target_total * sampling_multiplier
    
    if mode == "train":
        # 使用传入的 data_cursor
        start_cursor = data_cursor
        
        # 计算统计信息
        total_data_size = len(all_data)
        epochs_per_pass = math.ceil(total_data_size / total_to_sample) if total_to_sample > 0 else 1
        completed_passes = start_cursor // total_data_size
        current_position = start_cursor % total_data_size
        current_pass_progress = current_position / total_data_size * 100
        end_position = (start_cursor + total_to_sample) % total_data_size
        
        if accelerator.is_main_process:
            print(f"\n[Epoch {epoch} Rollout] Dataset: {dataset_name} ({total_data_size} samples)")
            print(f"  Target: {target_total} | Multiplier: {sampling_multiplier}x | Will sample: {total_to_sample}")
            print(f"  Cursor: {start_cursor} -> position {current_position} ({current_pass_progress:.1f}% of dataset)")
            if (start_cursor + total_to_sample) // total_data_size > start_cursor // total_data_size:
                print(f"  ** Will complete full dataset pass in this epoch **")
        
        # 使用打乱后的索引
        ordered_indices = _SHUFFLED_INDICES_CACHE[indices_cache_key]
        global_cursor = start_cursor
    else:
        # 评估模式不需要cursor
        ordered_indices = list(range(len(all_data)))
        global_cursor = 0
    
    def get_next_batch(batch_size):
        """按顺序循环获取下一批数据"""
        nonlocal global_cursor
        batch = []
        for _ in range(batch_size):
            # 循环索引
            idx = ordered_indices[global_cursor % len(ordered_indices)]
            batch.append(idx)
            global_cursor += 1
        return batch

    k_sample = getattr(config.rollout if mode == "train" else config.evaluation, 'num_response_per_task', 1)
    # 采样4倍数量，后续选择最短的k个
    k_sample_actual = k_sample

    def process_batch(batch_indices):
        import time
        batch_time_stats = {'sampling': 0.0, 'reward': 0.0, 'sampling_tokens': 0}
        
        if len(batch_indices) == 0:
            return [], batch_time_stats
        idx_pairs = [all_data[i] for i in batch_indices]
        _, prompt_gt_pairs = zip(*idx_pairs)
        batch_prompts, batch_gts = zip(*prompt_gt_pairs)
        prompts_list = [reason_prompt.format(problem=p) for p in batch_prompts]

        # 统一调用 sample 函数，都传 config 和 accelerator   
        # 同步调用，采样2倍数量
        try:
            outputs = sample(client, model_name, gen_config, prompts_list, k_sample_actual, config, accelerator, use_tqdm=True, dynamic_threshold=dynamic_threshold)
            if accelerator.is_main_process and len(outputs) > 0:
                print(f"[Rank {rank}] Generated {len(outputs)} responses for {len(prompts_list)} prompts (k={k_sample_actual})")
        except Exception as e:
            outputs = []
        
        # 从outputs中统计采样时间和token数
        for o in outputs:
            batch_time_stats['sampling'] += o.generation_time
            batch_time_stats['sampling_tokens'] += o.token_length


        expected = len(prompts_list) * k_sample_actual
        actual = len(outputs)
        if actual == 0:
            return []
        if actual != expected:
            # 截断到整组
            usable = (actual // k_sample_actual) * k_sample_actual
            outputs = outputs[:usable]

        batch_selected = []
        for original_idx in range(0, len(outputs) // k_sample_actual):
            question = batch_prompts[original_idx]
            prompt = prompts_list[original_idx]
            gt = batch_gts[original_idx]

            combined_texts = []
            combined_step_maps = []
            extracted_answers = []
            correctness_list = []
            raw_rewards = []
            input_ids_list = []
            token_ids_list = []
            generation_times = []
            prompt_lengths = []
            token_lengths = []

            # 先收集所有2*k_sample个样本的信息
            all_samples = []
            for i in range(k_sample_actual):
                output_idx = original_idx * k_sample_actual + i
                o = outputs[output_idx]
                text = o.text
                token_ids = o.token_ids
                step_map = o.step_map
                prompt_ids = o.prompt_ids
                generation_time = o.generation_time
                prompt_length = o.prompt_length
                token_length = o.token_length
                # assert "<|im_end|>" in text or "<|endoftext|>" in text or len(token_ids)>=config.rollout.max_token, f"len: {len(token_ids)}, eos: {token_ids[-1]}"
                
                extracted_answer = extract_final_boxed_answer(text)

                # ===== Reward计算计时 =====
                reward_start = time.time()
                
                if reward == "math":
                    correctness = math_judge(completions=[text], solution=[gt], response_token_ids=[token_ids], max_tokens=config.rollout.max_token)[0]

                    # 计算 speed_metric（训练和评估都计算）
                    response_length = len(token_ids) if token_ids is not None else 0
                    unique_steps = len(set(step_map)) if len(step_map) > 0 else 1
                    speed_metric = response_length / unique_steps if unique_steps > 0 else 1

                    if mode == "train":
                        # 使用新的reward_funcs系统计算reward
                        total_reward, reward_components = compute_reward_from_funcs(
                            text=text, 
                            gt=gt, 
                            token_ids=token_ids,
                            step_map=step_map
                        )
                        
                        # 记录每个reward function的分数（按reward_func_list的顺序）
                        reward_scores = [reward_components.get(func_name, 0.0) for func_name in reward_func_list]
                    else:
                        # 评测模式：使用 math_judge 的结果，与 eval_lmdeploy.py 保持一致
                        reward_scores = [correctness]
                else:
                    correctness = 0
                    reward_scores = [0.0] * len(reward_func_list)
                    speed_metric = 0.0
                
                batch_time_stats['reward'] += time.time() - reward_start
                
                # 保存样本信息
                all_samples.append({
                    'text': text,
                    'token_ids': token_ids,
                    'step_map': step_map,
                    'prompt_ids': prompt_ids,
                    'generation_time': generation_time,
                    'prompt_length': prompt_length,
                    'token_length': token_length,
                    'extracted_answer': extracted_answer,
                    'correctness': correctness,
                    'reward_scores': reward_scores,
                    'speed_metric': speed_metric,
                })
            
            # 按token_length排序，选择最短的k_sample个
            # all_samples.sort(key=lambda x: x['token_length'])
            selected_samples = all_samples[:k_sample]
            
            # 从选择的样本中提取信息
            speed_metrics = []
            for s in selected_samples:
                combined_texts.append(s['text'])
                combined_step_maps.append(s['step_map'])
                extracted_answers.append(s['extracted_answer'])
                input_ids_list.append(s['prompt_ids'])
                token_ids_list.append(s['token_ids'])
                generation_times.append(s['generation_time'])
                prompt_lengths.append(s['prompt_length'])
                token_lengths.append(s['token_length'])
                correctness_list.append(s['correctness'])
                raw_rewards.append(s['reward_scores'])
                speed_metrics.append(s['speed_metric'])

            mean_correctness = sum(correctness_list) / len(correctness_list)
            
            actual_k = len(combined_texts)
            # raw_rewards 是二维列表: [[r1_f1, r1_f2, ...], [r2_f1, r2_f2, ...], ...]
            # 使用 reward_weights 加权求和
            raw_rewards_scalar = [sum(r * w for r, w in zip(x, reward_weights)) for x in raw_rewards]
            if mode == "train":
                if normalize:
                    rewards = z_score_normalize(raw_rewards_scalar)
                else:
                    rewards = raw_rewards_scalar
            else:
                rewards = raw_rewards_scalar

            # 计算 extract_reward: 提取成功为1，失败为0
            extract_rewards = [1.0 if ext != EXTRACT_FAILED else 0.0 for ext in extracted_answers]

            # 评测模式使用与 eval_lmdeploy.py 一致的数据格式，训练模式保持原有格式
            if mode == "eval":
                batch_selected.append({
                    "question": question,
                    "prompt": prompt,
                    "ground_truth_answer": gt,
                    "full_output": combined_texts,
                    "extracted_output": extracted_answers,
                    "rewards": rewards,
                    "extract_reward": extract_rewards,
                    "speed_metrics": speed_metrics,
                    "step_map": combined_step_maps,
                    "prompt_ids": input_ids_list,
                    "token_ids": token_ids_list,
                    "generation_time": generation_times,
                    "prompt_length": prompt_lengths,
                    "token_length": token_lengths,
                })
            else:
                batch_selected.append({
                    "question": [question] * actual_k,
                    "prompt": [prompt] * actual_k,
                    "response": combined_texts,
                    "ground_truth_answer": [gt] * actual_k,
                    "full_output": combined_texts,
                    "extracted_output": extracted_answers,
                    "correctness": correctness_list,
                    "mean_correctness": mean_correctness,  # 添加用于后续过滤
                    "rewards": rewards,
                    "raw_rewards": raw_rewards_scalar,
                    "reward_components": raw_rewards,
                    "extract_reward": extract_rewards,
                    "speed_metrics": speed_metrics,
                    "step_map": combined_step_maps,
                    "prompt_ids": input_ids_list,
                    "token_ids": token_ids_list,
                    "generation_time": generation_times,
                    "prompt_length": prompt_lengths,
                    "token_length": token_lengths,
                })
        return batch_selected, batch_time_stats


    if mode == "train":
        # 简化逻辑：每轮生成 -> gather -> 过滤 -> 判断是否继续
        batch_size = config.rollout.num_task_per_step if config.rollout.num_task_per_step != -1 else len(all_data)
        round_num = 0
        selected_data = []  # main process 累积过滤后的数据
        
        # 时间统计累加器
        import time
        total_sampling_time = 0.0
        total_reward_time = 0.0
        total_sampling_tokens = 0
        
        # 循环采样直到达到目标数量
        while True:
            accelerator.wait_for_everyone()
            round_num += 1
            
            # 每轮采样 sampling_multiplier 倍的任务量（考虑DAPO过滤）
            sample_size = batch_size * sampling_multiplier
            # 获取全局批次，然后按 rank 分片
            global_batch = get_next_batch(sample_size)
            local_batch = get_data_chunk(global_batch, world_size, rank)
            local_data, batch_time_stats = process_batch(local_batch)
            
            # 累加时间和token统计
            total_sampling_time += batch_time_stats['sampling']
            total_reward_time += batch_time_stats['reward']
            total_sampling_tokens += batch_time_stats['sampling_tokens']
            
            # Gather 所有 rank 的数据
            all_round_data = gather_object(local_data)
            
            # 清理local_data以释放内存
            del local_data
            
            # Main process 进行 DAPO 过滤并累积
            if accelerator.is_main_process:
                # 展平多 rank 数据
                if isinstance(all_round_data[0], list):
                    all_round_data = [item for rank_data in all_round_data for item in rank_data]
                
                # DAPO 过滤：仅保留 0 < mean_correctness < 1.0 且至少有一个正确的，同时过滤长度和 step_map
                filtered_data = []
                for item in all_round_data:
                    mean_corr = item.get("mean_correctness", 0.0)
                    correctness_list = item.get("correctness", [])
                    token_lengths = item.get("token_length", [])
                    step_maps = item.get("step_map", [])
                    
                    if token_lengths:
                        min_len = min(token_lengths)
                        max_len = max(token_lengths)
                        if filter:
                            # # # 过滤条件：correctness + 长度范围 + step_map 均匀性
                            if (0 < mean_corr < 1.0 and max(correctness_list) == 1.0):
                                filtered_data.append(item)
                        else:
                            filtered_data.append(item)
                selected_data.extend(filtered_data)
                print(f"[Round {round_num}] Filtered: {len(filtered_data)}/{len(all_round_data)} | Total: {len(selected_data)}/{target_total}")
                
                # 清理已处理的数据以释放内存
                del all_round_data, filtered_data
                
                # 检查是否达到目标
                continue_sampling = len(selected_data) < target_total
            else:
                # 非主进程释放gather的数据以节省内存
                del all_round_data
                continue_sampling = None
            
            # Broadcast 是否继续采样
            from accelerate.utils import broadcast_object_list
            continue_sampling = broadcast_object_list([continue_sampling], from_process=0)[0]
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if not continue_sampling:
                break
        
        # 循环结束后同步
        accelerator.wait_for_everyone()
    
    if mode == "train":
        # 训练模式：Main process 已经有过滤后的数据，现在截断并统计
        if accelerator.is_main_process:
            # 截断到 target_total
            if len(selected_data) > target_total:
                random.shuffle(selected_data)
                selected_data = selected_data[:target_total]
            
            all_data_collected = selected_data
            
            # 统计 correctness 分布
            count_low = count_mid = count_high = 0
            for item in all_data_collected:
                mean_corr = item.get("mean_correctness", 0.0)
                if mean_corr < 0.2:
                    count_low += 1
                elif mean_corr <= 0.8:
                    count_mid += 1
                else:
                    count_high += 1
            
            total = count_low + count_mid + count_high
            if total > 0:
                print(f"[Epoch {epoch}] Collected {len(all_data_collected)} tasks | <0.2: {count_low} ({count_low/total*100:.1f}%) | 0.2-0.8: {count_mid} ({count_mid/total*100:.1f}%) | >0.8: {count_high} ({count_high/total*100:.1f}%)")
        else:
            # 非主进程初始化为 None
            all_data_collected = None
        
        # 使用 broadcast 而不是 gather，避免内存浪费
        # 只需要 rank0 的数据广播给其他 rank，不需要 gather 所有的 None
        from accelerate.utils import broadcast_object_list
        all_data_collected = broadcast_object_list([all_data_collected], from_process=0)[0]
        
        # 每个 rank 获取部分切片
        local_data_collected = get_data_chunk(all_data_collected, world_size, rank)
        
        # 清理完整数据以释放内存（每个rank只保留自己的切片）
        del all_data_collected
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 返回数据和更新后的cursor
        updated_cursor = global_cursor
        if accelerator.is_main_process:
            print(f"[Epoch {epoch}] Cursor: {start_cursor} -> {updated_cursor} (advanced {updated_cursor - start_cursor})")
        
        # 检查本epoch是否完成了一次完整的数据集遍历
        # 判断条件：跨越了数据集边界（完成了一个完整周期）
        if (updated_cursor // total_data_size) > (start_cursor // total_data_size):
            if accelerator.is_main_process:
                print(f"[Epoch {epoch}] ** Completed full dataset pass! Reshuffling for next pass **")
                shuffled_indices = list(range(len(all_data)))
                random.shuffle(shuffled_indices)
                _SHUFFLED_INDICES_CACHE[indices_cache_key] = shuffled_indices
            accelerator.wait_for_everyone()
            # 同步打乱后的索引到所有进程
            if not accelerator.is_main_process:
                _SHUFFLED_INDICES_CACHE[indices_cache_key] = None
            from accelerate.utils import broadcast_object_list
            shuffled_indices = broadcast_object_list([_SHUFFLED_INDICES_CACHE[indices_cache_key]], from_process=0)[0]
            _SHUFFLED_INDICES_CACHE[indices_cache_key] = shuffled_indices
        
        # 构建时间统计字典
        time_breakdown = {
            'sampling': total_sampling_time,
            'reward': total_reward_time,
            'sampling_tokens': total_sampling_tokens
        }
        
        if not split_rank:
            return local_data_collected, updated_cursor, time_breakdown
        else:
            return all_data_collected, updated_cursor, time_breakdown



def rollout_sampling_eval(client, dataset_name, epoch, config, model_path, model_name, tokenizer, accelerator, reward_function=None, normalize=True, filter=True, mode="train"):
    
    # epoch = 1
    # 每轮控制采样的子集数据集不一样，需要设置不同的随机种子
    if config.training.seed is not None:
        random.seed(config.training.seed)

    project_name = config.experiment.project
    # experiment.project: 实验名；experiment.output_dir: 真实输出目录（可能是绝对路径）
    output_dir = getattr(config.experiment, "output_dir", None) or project_name
    output_dir = str(output_dir)
    reward = config.dataset.data_type
    math_judge = orms['accuracy']()
    cosine_reward = orms['cosine'](cosine_max_len=config.rollout.max_token, accuracy_orm=math_judge)
    speed_reward = orms['speed'](speed_max_len=config.rollout.block_size, accuracy_orm=math_judge)
    speed_penalty = orms['speed_penalty'](speed_max_len=config.rollout.block_size, accuracy_orm=math_judge)
    repetition_reward = orms['repetition']()
    format_reward = orms['format']()

    backend_config = PytorchEngineConfig(
        dtype="auto",  # 与训练时的mixed_precision保持一致
        cache_max_entry_count=0.95,
        dllm_block_length=config.evaluation.block_size,
        dllm_denoising_steps=config.evaluation.denoising_steps_per_block,
        dllm_unmasking_strategy=config.evaluation.remasking_strategy,
        dllm_confidence_threshold=config.evaluation.dynamic_threshold,
        # max_prefill_token_num=config.rollout.max_token*64,
    )

    gen_config = GenerationConfig(
        top_p=config.evaluation.top_p,
        top_k=config.evaluation.top_k,
        temperature=config.evaluation.temperature,
        do_sample=config.evaluation.do_sample,
        # min_new_tokens=128, # 防止突然早停
        max_new_tokens=config.evaluation.max_token,
        skip_special_tokens=False,
        # random_seed=config.training.seed,  # eval模式使用固定seed，不使用epoch
        # stop_words=["<|endoftext|>"],
    )

    start_with_think = config.rollout.start_with_think
    # dataset_name = config.dataset.train_dataset

    outputs_name = "eval-" + model_path.replace("/", ".") + "-" + dataset_name + "-" + str(epoch)
    # Load data
    if accelerator.is_main_process:
        # rank0 负责构建数据集并写缓存
        dataset = load_dataset("json", data_files=f"data/{dataset_name}.json")

    # 等 rank0 处理完再继续
    accelerator.wait_for_everyone()
    # 所有 rank 都能安全读取缓存
    if not accelerator.is_main_process:
        dataset = load_dataset("json", data_files=f"data/{dataset_name}.json")

    ds = dataset['train']
    prompts = ds['question']
    gts = ds['ground_truth_answer']

    reason_prompt = "<|im_start|>user\n{problem}\nPlease reason step by step, and put your final answer within $\\boxed{{}}$.<|im_end|>\n<|im_start|>assistant\n"

    if start_with_think:
        reason_prompt = "<|im_start|>system\nPlease reason step by step, and put your final answer within $\\boxed{{}}$.<|im_end|>\n<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n<think>"

    # 构建带原始索引的数据
    all_data = list(enumerate(zip(prompts, gts))) # [(idx, (prompt, gt)), ...]
    # random.seed(config.training.seed+epoch)
    # random.seed(config.training.seed)
    # Directly sample and shuffle the original data
    num_tasks_total = len(all_data)

    rank = accelerator.process_index
    num_processes = accelerator.num_processes
    local_data = get_data_chunk(all_data, accelerator.num_processes, accelerator.process_index)

    # 初始化时间统计
    eval_sampling_time = 0.0
    eval_reward_time = 0.0
    
    if len(local_data) == 0:
        accelerator.print(f"[Rank {rank}] No prompts assigned.")
        # 仍需返回空结果以便 gather
        data = []
        count_low = 0
        count_mid = 0
        count_high = 0
    else:
        indices, prompt_gt_pairs = zip(*local_data)
        local_prompts, local_gts = zip(*prompt_gt_pairs)

        # Apply chat template
        prompts_list = [reason_prompt.format(problem=p) for p in local_prompts]

        # sample函数现在返回完整的data列表，支持k次采样
        k_sample = getattr(config.evaluation, 'num_response_per_task', 1)  # 从配置中获取k值

        # 统一调用 sample 函数，都传 config 和 accelerator   
        # 同步调用
        import time
        eval_sampling_start = time.time()
        try:
            # eval时不使用动态阈值，使用配置中的固定阈值
            outputs = sample(client, model_name, gen_config, prompts_list, k_sample, config, accelerator, use_tqdm=True, dynamic_threshold=None)
            accelerator.print(f"[Rank {rank}] Generated {len(outputs)} responses for {len(prompts_list)} prompts (k={k_sample})")
        except Exception as e:
            outputs = []
        eval_sampling_time = time.time() - eval_sampling_start

        data = []
        # 统计correctness分布
        count_low = 0  # <0.2
        count_mid = 0  # 0.2-0.8
        count_high = 0  # >0.8
        
        # Reward计算时间统计
        eval_reward_time = 0.0
        
        for original_idx in range(0, len(outputs) // k_sample):
            question = local_prompts[original_idx]
            prompt = prompts_list[original_idx]
            gt = local_gts[original_idx]

            combined_texts = []
            combined_step_maps = []
            extracted_answers = []
            correctness_list = []
            raw_rewards = []
            speed_rewards = []
            speed_metrics = []
            input_ids_list = []
            token_ids_list = []
            generation_times = []
            prompt_lengths = []
            token_lengths = []
            for i in range(k_sample):
                output_idx = original_idx * k_sample + i
                o = outputs[output_idx]
                text = o.text
                token_ids = o.token_ids
                step_map = o.step_map
                prompt_ids = o.prompt_ids
                generation_time = o.generation_time
                prompt_length = o.prompt_length
                token_length = o.token_length

                combined_texts.append(text)
                combined_step_maps.append(step_map)
                extracted_answers.append(extract_final_boxed_answer(text))
                input_ids_list.append(prompt_ids)
                token_ids_list.append(token_ids)
                generation_times.append(generation_time)
                prompt_lengths.append(prompt_length)
                token_lengths.append(token_length)

                # ===== Reward计算计时 =====
                reward_start = time.time()
                
                if reward == "math":
                    correctness = math_judge([text], [gt], response_token_ids=[token_ids], max_tokens=config.evaluation.max_token)[0]
                    
                    # 计算 speed_metric 和 speed_reward（一直计算，不管是否开启）
                    response_length = len(token_ids) if token_ids is not None else 0
                    unique_steps = len(set(step_map)) if len(step_map) > 0 else 1
                    speed_metric = response_length / unique_steps if unique_steps > 0 else 1
                    
                    # 使用 speed_reward ORM 计算奖励
                    speed_reward_value = speed_reward(
                        completions=[text], 
                        solution=[gt], 
                        response_token_ids=[token_ids],
                        step_map=[step_map]
                    )[0]
                    
                    if correctness == 0:
                        if "boxed" in text:
                            correct_reward = 0.1
                        else:
                            correct_reward = 0
                    else:
                        correct_reward = 1
                        
                    correctness_list.append(correctness)
                    raw_rewards.append([correct_reward])
                    speed_rewards.append(speed_reward_value)
                    speed_metrics.append(speed_metric)
                else:
                    correctness_list.append(0)
                    speed_rewards.append(0.0)
                    speed_metrics.append(0.0)
                
                eval_reward_time += time.time() - reward_start

            # 计算mean_correctness并统计分布
            mean_correctness = sum(correctness_list) / len(correctness_list)
            if mean_correctness < 0.2:
                count_low += 1
            elif mean_correctness <= 0.8:
                count_mid += 1
            else:
                count_high += 1

            raw_rewards = [sum(x) for x in raw_rewards]

            rewards = raw_rewards

            data.append({
                "question": [question]*k_sample,
                "prompt": [prompt]*k_sample,
                "response": combined_texts,
                "ground_truth_answer": [gt]*k_sample,
                "full_output": combined_texts,
                "extracted_output": extracted_answers,
                "prompt_length": prompt_lengths,
                "token_length": token_lengths,
                "correctness": correctness_list,
                "rewards": rewards,
                "raw_rewards": raw_rewards,
                "speed_rewards": speed_rewards,
                "speed_metrics": speed_metrics,
                "prompt_ids": input_ids_list,
                "token_ids": token_ids_list,
                "generation_time": generation_times,
                "step_map": combined_step_maps,
            })

        accelerator.print(f"[Rank {rank}] Correctness: <0.2: {count_low}, 0.2-0.8: {count_mid}, >0.8: {count_high}")
        
    accelerator.wait_for_everyone()

    # Gather局部统计
    all_counts = gather_object([count_low, count_mid, count_high])
    
    # 直接gather data
    all_data = gather_object(data)
    
    # Gather时间统计
    all_time_stats = gather_object([{
        'sampling': eval_sampling_time,
        'reward': eval_reward_time
    }])
    
    # 在主进程打印全局统计并保存结果
    if accelerator.is_main_process:
        # 展平多 rank 数据
        if isinstance(all_data[0], list):
            all_data_flattened = [item for rank_data in all_data for item in rank_data]
        else:
            all_data_flattened = all_data
        
        # 全局统计
        total_low = sum(all_counts[i] for i in range(0, len(all_counts), 3))
        total_mid = sum(all_counts[i] for i in range(1, len(all_counts), 3))
        total_high = sum(all_counts[i] for i in range(2, len(all_counts), 3))
        total = total_low + total_mid + total_high
        print(f"[Evaluation Step {epoch}] Total: {total} | <0.2: {total_low} ({total_low/total*100:.1f}%) | 0.2-0.8: {total_mid} ({total_mid/total*100:.1f}%) | >0.8: {total_high} ({total_high/total*100:.1f}%)")
        
        # 计算accuracy和pass@k
        total_correct = sum(sum(item.get("correctness", [])) for item in all_data_flattened)
        total_responses = sum(len(item.get("correctness", [])) for item in all_data_flattened)
        pass_at_k = sum(1 for item in all_data_flattened if any(c > 0 for c in item.get("correctness", [])))
        accuracy = (total_correct / total_responses * 100) if total_responses > 0 else 0.0
        pass_at_k_rate = (pass_at_k / len(all_data_flattened) * 100) if len(all_data_flattened) > 0 else 0.0
        
        print(f"[Evaluation Step {epoch}] Accuracy: {accuracy:.2f}% | Pass@K: {pass_at_k_rate:.2f}%")
        
        # 保存结果到 JSON 文件
        import json
        output_data = {
            "accuracy": accuracy,
            "pass_at_k": pass_at_k_rate,
            "data": all_data_flattened
        }
        output_file_name = os.path.join(output_dir, "temp_data", "outputs-" + outputs_name + ".json")
        os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
        with open(output_file_name, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_file_name}")
    
    # 确保主进程完成文件保存后所有进程再一起返回
    accelerator.wait_for_everyone()
    
    return all_data

