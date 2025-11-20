from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig, ChatTemplateConfig
from lmdeploy.pytorch.tools.utils import Timer, visualize_pipe_out
from accelerate import Accelerator, utils
import accelerate
from transformers import AutoTokenizer
import gc
import torch

if __name__ == '__main__':
    # Initialize accelerator for multi-node multi-GPU support
    accelerator = Accelerator()
    
    model_path = 'xxx/'
    print(model_path)

    question = "What is $1^{(2^{235423523})}$?"
    prompts = [
        [dict(role="user", content=question+"\nPlease reason step by step, and put your final answer within $\\boxed{}$.")]
    ]
    # prompts = [
        # [dict(role="user", content="生命的意义在于什么？")]
    # ]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompts = tokenizer.apply_chat_template(prompts, tokenize=False,add_generation_prompt=True)
    prompts = [x for x in prompts]
    print(prompts)
    print(tokenizer.encode(prompts[0]))
    backend_config = PytorchEngineConfig(
        dtype="bfloat16",
        session_len=32678+2048,
        max_prefill_token_num=8192,
        cache_max_entry_count=0.8,
        dllm_block_length=4,
        dllm_denoising_steps=4,
        dllm_unmasking_strategy="low_confidence_dynamic",
        dllm_confidence_threshold=0.9,
    )
    with pipeline(model_path, backend_config=backend_config) as pipe:
        gen_config = GenerationConfig(
            n=1,
            top_p=1.0,
            top_k=50,
            temperature=1.0,
            do_sample=False, # greedy decoding
            max_new_tokens=8192,
            skip_special_tokens=False,
            repetition_penalty=1.0,
        )
        print(gen_config)
        outputs = pipe(prompts, gen_config=gen_config)
        
        # 打印输出和 step_map
        for idx, output in enumerate(outputs):
            # Save the output to a file
            decoded_output = tokenizer.decode(output.token_ids)
            with open("output.txt", "w", encoding="utf-8") as f:
                f.write(decoded_output)
                f.write("Gen Length: " + str(len(output.token_ids)) + "\n")

        # for item in pipe.stream_infer(prompts, gen_config=gen_config):
            # print(item.text,end="",flush=True)
        print("\nInference completed")
    gc.collect()
    torch.cuda.empty_cache()

