import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from typing import List
from dataclasses import dataclass
from openai import OpenAI
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm
def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


@dataclass
class GenerationOutput:
    """模拟lmdeploy的输出格式"""
    text: str
    prompt_ids: List[int]
    token_ids: List[int]
    step_map: List[int]
    generation_time: float = 0.0
    prompt_length: int = 0
    token_length: int = 0


def _single_batch_request(client, model_name, gen_config, current_batch, batch_idx, total_batches, rank, dynamic_threshold=None):
    """单个批次的API请求（用于并发）"""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            extra_body = {"top_k": gen_config.top_k}
            if dynamic_threshold is not None:
                extra_body["dllm_confidence_threshold"] = dynamic_threshold
            extra_body["skip_special_tokens"] = False
            response = client.completions.create(
                model=model_name,
                prompt=current_batch,
                temperature=gen_config.temperature,
                top_p=gen_config.top_p,
                max_tokens=gen_config.max_new_tokens,
                stop=["<|im_end|>","<|endoftext|>"],
                extra_body=extra_body,
                timeout=3000.0,  # 30分钟超时
            )
            
            # 检查响应
            if not hasattr(response, 'choices') or not response.choices:
                raise ValueError("Invalid response: no choices")
            
            # 解析响应
            batch_outputs = []
            for idx, g in enumerate(response.choices):
                # 检查必要字段
                if not hasattr(g, 'text') or g.text is None:
                    continue
                if not hasattr(g, 'input_ids') or g.input_ids is None:
                    continue
                    
                output = GenerationOutput(
                    text=g.text, 
                    prompt_ids=g.input_ids, 
                    token_ids=g.token_ids if hasattr(g, 'token_ids') and g.token_ids else [], 
                    step_map=g.step_map if hasattr(g, 'step_map') and g.step_map else [], 
                    generation_time=g.generation_time if hasattr(g, 'generation_time') else 0.0,
                    prompt_length=len(g.input_ids),
                    token_length=len(g.token_ids) if hasattr(g, 'token_ids') and g.token_ids else 0
                )
                batch_outputs.append(output)
            
            if not batch_outputs:
                raise ValueError(f"No valid outputs from {len(response.choices)} choices")
            
            return batch_idx, batch_outputs, None
            
        except Exception as e:
            if attempt == max_retries - 1:
                return batch_idx, [], str(e)
            time.sleep(2 ** attempt)  # 指数退避
    
    return batch_idx, [], "Max retries exceeded"


def sample(client, model_name, gen_config, prompts, k_sample=1, config=None, accelerator=None, use_tqdm=True, dynamic_threshold=None):
    """
    使用并发异步调用来生成所有响应，提高推理速度
    
    Args:
        client: OpenAI客户端
        model_name: 模型名称
        gen_config: 生成配置
        prompts: 输入提示列表
        k_sample: 每个提示生成k个响应
        config: 全局配置对象
        accelerator: Accelerator对象
        use_tqdm: 是否使用进度条
    
    Returns:
        List[GenerationOutput]: 生成的输出列表
    """
    
    # 构建批量请求：每个 prompt 重复 k_sample 次
    batch_prompts = []
    for prompt in prompts:
        batch_prompts.extend([prompt] * k_sample)
    
    # 分批处理，避免单次请求过大
    max_batch_size = getattr(config.rollout if config else None, 'api_batch_size', 2048)
    
    # 并发线程数（同时发送的批次数）
    max_workers = getattr(config.rollout if config else None, 'api_max_workers', 16)
    
    total_batches = (len(batch_prompts) + max_batch_size - 1) // max_batch_size
    
    rank = accelerator.process_index if accelerator else 0
    
    # 准备所有批次
    batches = []
    for batch_idx in range(total_batches):
        start_idx = batch_idx * max_batch_size
        end_idx = min(start_idx + max_batch_size, len(batch_prompts))
        current_batch = batch_prompts[start_idx:end_idx]
        batches.append((batch_idx, current_batch))
    
    # 使用线程池并发处理，结果按 batch_idx 排序
    results = [None] * total_batches
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_batch = {
            executor.submit(_single_batch_request, client, model_name, gen_config, batch, idx, total_batches, rank, dynamic_threshold): idx
            for idx, batch in batches
        }
        
        # 收集结果，添加进度条
        if use_tqdm and (accelerator is None or accelerator.is_main_process):
            pbar = tqdm(total=total_batches, desc=f"Rank {rank} Sampling", unit="batch")
        else:
            pbar = None
            
        for future in as_completed(future_to_batch):
            batch_idx, batch_outputs, error = future.result()
            results[batch_idx] = (batch_outputs, error)
            if pbar:
                pbar.update(1)
        
        if pbar:
            pbar.close()
    
    # 按顺序合并所有结果
    all_outputs = []
    failed_batches = 0
    for batch_idx, (batch_outputs, error) in enumerate(results):
        if error:
            failed_batches += 1
        else:
            all_outputs.extend(batch_outputs)
    
    return all_outputs

def main():
    model_name = "test"
    model_path = "/inspire/hdd/global_user/liuxiaoran-240108120089/public/SDAR-8B-Chat"
    local_ip = get_host_ip()
    rank = 0
    port = 12349
    base_url = f'http://{local_ip}:{port}/v1'
    api_key = "sk-"
    client = OpenAI(api_key="sk", base_url=f"http://0.0.0.0:{port}/v1")

    prompts = ["<|im_start|>user\nFind the greatest integer less than $(\\sqrt{7} + \\sqrt{5})^6.$  (Do not use a calculator!)\nPlease reason step by step, and put your final answer within $\\boxed{}$.<|im_end|>\n<|im_start|>assistant\n"]*10
    k_sample = 16
    config = None
    accelerator = type('obj', (object,), {'process_index': 0})()
    use_tqdm = True
    outputs = sample(client, model_name, gen_config=type('obj', (object,), {
        'temperature': 1.0,
        'top_p': 1.0,
        'top_k': 50,
        'max_new_tokens': 2048,
        'stop_words': None
    })(), prompts=prompts, k_sample=k_sample, config=config, accelerator=accelerator, use_tqdm=use_tqdm)

if __name__ == "__main__":
    main()
