import os
import pandas as pd
from typing import Callable, List, Dict, Any, Union, Tuple
from vllm import LLM, SamplingParams
import json

def load_math_data(path: str) -> List[Dict[str, Any]]:
    print(f"Loading Parquet from {path}...")
    df = pd.read_parquet(path)
    return df.to_dict(orient='records')

def format_prompt(data_path: str, prompt_path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    examples = load_math_data(data_path)
    
    with open(prompt_path, "r") as f:
        r1_zero_template = f.read()

    # Apply the template and pre-fill with <think>
    formatted_prompts = [
        r1_zero_template.format(question=ex["problem"])
        for ex in examples
    ]
    
    return examples, formatted_prompts

def generation_hyperparameters() -> SamplingParams:
    return SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], Union[float, bool]], 
    examples: List[Dict[str, Any]],
    prompts: List[str],
    eval_sampling_params: SamplingParams,
    output_path: str
) -> None:
    print(f"Generating responses for {len(prompts)} examples")
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    results = []
    total_correct = 0
    total_reward = 0.0
    total_format = 0.0
    total_answer = 0.0

    for example, output in zip(examples, outputs):
        generated_text = output.outputs[0].text
        ground_truth = example.get("solution", "") 
        
        reward_dict = reward_fn(generated_text, ground_truth)
        current_reward = float(reward_dict.get("reward", 0.0))
        current_format = float(reward_dict.get("format_reward", 0.0))
        current_answer = float(reward_dict.get("answer_reward", 0.0))
        total_reward += current_reward
        total_format += current_format
        total_answer += current_answer
        
        results.append({
            "problem": example.get("problem", ""),
            "expected_answer": ground_truth,
            "generated_text": generated_text,
            "reward": current_reward,
            "format_reward": current_format,
            "answer_reward": current_answer
        })
    
    avg_reward = total_reward / len(examples) if examples else 0.0
    avg_fmt_reward = total_format / len(examples) if examples else 0.0
    avg_ans_reward = total_answer / len(examples) if examples else 0.0
    print(f"\n--- Evaluation Metrics ---")
    print(f"Average Reward: {avg_reward:.4f}")
    print(f"Average Format Reward: {avg_fmt_reward:.4f}")
    print(f"Average Answer Reward: {avg_ans_reward:.4f}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + '\n')
    print(f"Results saved to: {output_path}")