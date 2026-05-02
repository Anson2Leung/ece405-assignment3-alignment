import argparse
import os
from unittest.mock import patch

import pandas as pd
import torch
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from math_verify import parse, verify

from cs336_alignment.drgrpo_grader import extract_boxed_answer, question_only_reward_fn, grade
from cs336_alignment.math_baseline import format_prompt

# Default file paths
DEFAULT_MATH_PATH   = "/home/ansonl32/ECE405/ece405-assignment3-alignment/data/math/math.parquet"
DEFAULT_PROMPT_FILE = "/home/ansonl32/ECE405/ece405-assignment3-alignment/cs336_alignment/prompts/r1_zero.prompt"
DEFAULT_MODEL_ID    = "/home/ansonl32/koa_scratch/ECE405/assignment3/Qwen/Qwen2.5-Math-1.5B"
DEFAULT_OUTPUT = "/home/ansonl32/ECE405/ece405-assignment3-alignment/data/math/math_filtered_correct.parquet"

def init_vllm(
    model_id: str,
    device: str,
    seed: int,
    gpu_memory_utilization: float = 0.75,
) -> LLM:

    vllm_set_random_seed(seed)

    if torch.cuda.is_bf16_supported():
        target_dtype = "bfloat16"
    else:
        target_dtype = "float16"

    print(f"Initializing vLLM with dtype: {target_dtype} on {device}")

    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=target_dtype,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


# call format_prompt from mathbaseline
def _format_prompt(math_path: str, prompt_file: str) -> tuple[pd.DataFrame, list[str]]:
    with open(prompt_file) as f:
        template = f.read()

    # Debugging
    print(f"[format_prompt] Raw template (first 300 chars):")
    print(repr(template[:300]))
    print()

    examples, formatted_prompts = format_prompt(math_path, prompt_file)
    df = pd.DataFrame(examples)
    print(f"[format_prompt] Loaded {len(df)} examples from {math_path}")
    return df, formatted_prompts



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter MATH dataset to correctly-answered examples")
    p.add_argument("--math_path",   type=str, default=DEFAULT_MATH_PATH)
    p.add_argument("--prompt_file", type=str, default=DEFAULT_PROMPT_FILE)
    p.add_argument("--model_id",    type=str, default=DEFAULT_MODEL_ID)
    p.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT)
    p.add_argument("--device",      type=str, default="cuda:0")
    p.add_argument("--vllm_mem",    type=float, default=0.70)
    p.add_argument("--batch_size",  type=int,   default=32,
                   help="vLLM generation batch size (does not affect memory much)")
    p.add_argument("--max_tokens",  type=int,   default=1024,
                   help="Max tokens for generation during filtering")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature (0 = greedy, recommended)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # format prompts
    df, formatted_prompts = _format_prompt(args.math_path, args.prompt_file)
    n_total = len(df)
    print(f"\nTotal examples in dataset : {n_total}")

    if "solution" in df.columns:
        solutions = [extract_boxed_answer(s) for s in df["solution"].tolist()]
        print(f"  Sample extracted answers: {solutions[:3]}")
    else:
        raise ValueError("DataFrame does not have 'solution' column.")

    print(f"\nInitialising vLLM ({args.model_id}) on {args.device} …")
    llm = init_vllm(args.model_id, args.device, args.seed, args.vllm_mem)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    print(f"\nGenerating completions for {n_total} examples (batch_size={args.batch_size})")
    correct_mask: list[bool] = [] # mask to filter correct responses

    for start in range(0, n_total, args.batch_size):
        end          = min(start + args.batch_size, n_total)
        batch_p      = formatted_prompts[start:end]
        batch_s      = solutions[start:end]

        outputs = llm.generate(batch_p, sampling_params)
        fallback_count = 0
 
        for output, sol in zip(outputs, batch_s):
            completion  = output.outputs[0].text
            
            # safety for malformed dataset rows
            if sol is None:
                correct_mask.append(False)
                continue
                
            is_correct = False
            
            # standard grader (looks for \boxed)
            try:
                reward_info = question_only_reward_fn(completion, sol)
                is_correct  = reward_info.get("reward", 0.0) > 0.5
            except Exception:
                is_correct = False
            
            # fallback extract from <answer> tags (was having issues with getting answer)
            if not is_correct and "<answer>" in completion and "</answer>" in completion:
                try:
                    # Extract what is inside the <answer> tags
                    raw_ans = completion.split("<answer>")[-1].split("</answer>")[0].strip()
                    
                    # Grade extracted string against the ground truth
                    is_correct = grade(raw_ans, sol, fast=True)
                    
                    if is_correct:
                        # print(f"\n  --> [Strict Fallback SAVED] Extracted: '{raw_ans}' | GT: '{sol}'")
                        fallback_count += 1
                except Exception:
                    is_correct = False
                
            correct_mask.append(is_correct)

        n_done = end
        n_correct = sum(correct_mask)
        print(
            f"  Processed {n_done:5d}/{n_total}  |  "
            f"correct so far: {n_correct} ({100*n_correct/n_done:.1f}%)"
        )

    # Filter DataFrame for correct 
    df_filtered = df[correct_mask].reset_index(drop=True)
    n_filtered  = len(df_filtered)

    print(f"\n{'='*60}")
    print(f"Filtering complete.")
    print(f"  Total examples    : {n_total}")
    print(f"  Correct (kept)    : {n_filtered}  ({100*n_filtered/n_total:.1f}%)")
    print(f"  Correct (fallback): {fallback_count}")
    print(f"  Removed           : {n_total - n_filtered}")
    print(f"{'='*60}\n")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    df_filtered.to_parquet(args.output_path, index=False)
    print(f"Filtered dataset saved → {args.output_path}")


if __name__ == "__main__":
    main()