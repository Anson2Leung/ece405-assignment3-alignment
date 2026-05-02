import argparse
import os
import random
import re
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import wandb
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from sft_helpers import (
    compute_entropy,
    get_response_log_probs,
    log_generations,
    sft_microbatch_train_step,
    tokenize_prompt_and_output,
    evaluate_accuracy,
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, extract_answer, question_only_reward_fn, grade
from cs336_alignment.math_baseline import format_prompt


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class EIDataset(Dataset):
    """Dataset built from correct model-generated reasoning traces."""
    def __init__(self, dsft_pairs: list[dict]):
        self.data = dsft_pairs

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]


def collate_fn(batch: list[dict]) -> dict:
    return {
        "prompts":   [ex["prompt"]   for ex in batch],
        "responses": [ex["response"] for ex in batch],
    }


def load_policy_into_vllm(policy: PreTrainedModel, llm: LLM) -> None:
    state_dict = policy.state_dict()
    llm_model  = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def evaluate_val_loss(
    policy,
    val_samples: list[dict],
    tokenizer,
    device: str,
    max_seq_len: int,
    batch_size: int = 4,
) -> float:
    """
    Compute average per-token cross-entropy loss on the validation set
    """
    policy.eval()
    total_loss   = 0.0
    total_tokens = 0

    with torch.no_grad():
        for start in range(0, len(val_samples), batch_size):
            batch = val_samples[start : start + batch_size]
            prompts   = [s["formatted_prompt"] for s in batch]
            responses = [s["solution"]         for s in batch]

            tokens = tokenize_prompt_and_output(prompts, responses, tokenizer)
            tokens = {k: v[:, :max_seq_len] for k, v in tokens.items()}

            input_ids     = tokens["input_ids"].to(device)
            labels        = tokens["labels"].to(device)
            response_mask = tokens["response_mask"].to(device)

            n_resp = response_mask.sum().item()
            if n_resp == 0:
                continue

            lp_out = get_response_log_probs(policy, input_ids, labels)
            # per-token NLL, masked to response only
            per_token_nll = -lp_out["log_probs"] * response_mask
            total_loss   += per_token_nll.sum().item()
            total_tokens += n_resp

    return total_loss / total_tokens if total_tokens > 0 else float("nan")

def build_val_samples(examples: list[dict], formatted_prompts: list[str], val_size: int, seed: int) -> list[dict]:
    # Builds validation set from the formatted dataset.
    rng = random.Random(seed)
    
    # Zip for the shuffle
    paired_data = list(zip(examples, formatted_prompts))
    rng.shuffle(paired_data)
    
    val_samples = []
    # Take the first `val_size` items after shuffling
    for ex, prompt in paired_data[:val_size]:
        sol = ex.get("solution", "")
        # Extract the short answer
        short_ans = ex.get("answer", extract_answer(sol)) 
        
        val_samples.append({
            "formatted_prompt": prompt,
            "solution":         sol,
            "answer":           short_ans,
        })
    return val_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id",       type=str, required=True)
    parser.add_argument("--math_path",      type=str, required=True)
    parser.add_argument("--prompt_file",    type=str, required=True)
    parser.add_argument("--output_dir",     type=str,
                        default="/home/ansonl32/koa_scratch/ECE405/assignment3/expert_iteration")

    parser.add_argument("--n_ei_steps",   type=int,   default=5)
    parser.add_argument("--db_size",      type=int,   default=512)
    parser.add_argument("--num_rollouts", type=int,   default=4)
    parser.add_argument("--sft_epochs",   type=int,   default=1)

    parser.add_argument("--lr",          type=float, default=2e-5)
    parser.add_argument("--batch_size",  type=int,   default=1)
    parser.add_argument("--grad_accum",  type=int,   default=8)
    parser.add_argument("--max_seq_len", type=int,   default=1024)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--val_size",    type=int,   default=200)

    parser.add_argument("--policy_device",  type=str, default="cuda:0")
    parser.add_argument("--vllm_device",    type=str, default="cuda:1")
    parser.add_argument("--vllm_mem",       type=float, default=0.6)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--wandb_project",  type=str, default="ece405_assignment3")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    run_name = args.wandb_run_name or (
        f"ei_db{args.db_size}_G{args.num_rollouts}_E{args.sft_epochs}"
    )
    wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    wandb.define_metric("ei_step")
    wandb.define_metric("train_step")
    wandb.define_metric("ei/*",    step_metric="ei_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*",  step_metric="ei_step")


    print(f"Loading and formatting prompts from {args.math_path}")
    examples, formatted_prompts = format_prompt(args.math_path, args.prompt_file)
    val_indices  = set(range(args.val_size))
    
    val_size = min(200, int(len(examples) * 0.1))
    val_samples = build_val_samples(examples, formatted_prompts, val_size, seed=42)

    train_paired = list(zip(examples, formatted_prompts))[val_size:]
    train_df = pd.DataFrame([{
        "formatted_prompt": p, 
        "solution": ex.get("solution", ""),
        "answer": ex.get("answer", extract_answer(ex.get("solution", "")))
    } for ex, p in train_paired])

    # Policy and Optimizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
    ).to(args.policy_device)

    policy.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    # 4. Init vLLM
    llm = LLM(
        model=args.model_id,
        dtype="bfloat16",
        gpu_memory_utilization=args.vllm_mem,
        seed=42,
    )

    sampling_params = SamplingParams(
            temperature=1.0,
            max_tokens=args.max_seq_len,
            min_tokens=4,
            n=args.num_rollouts,
            stop=["</answer>"],
            include_stop_str_in_output=True,
        )

    # Expert Iteration Loop
    for ei_step in range(1, args.n_ei_steps + 1):
        print(f"\n{'='*60}\nExpert Iteration Step {ei_step}/{args.n_ei_steps}")
        
        # Sample a batch of questions
        db_df = train_df.sample(n=min(args.db_size, len(train_df)))
        prompts = db_df["formatted_prompt"].tolist()
        ground_truths = db_df["answer"].tolist()

        load_policy_into_vllm(policy, llm)

        # Step 5: Rollouts
        print(f"Generating {args.num_rollouts} rollouts for {len(prompts)} questions...")
        outputs = llm.generate(prompts, sampling_params)

        # Evaluate and filter
        dsft_pairs = []
        correct_count = 0
        total_rollouts = len(prompts) * args.num_rollouts

        for output, gt_ans in zip(outputs, ground_truths):
            q = output.prompt
            
            # Skip if dataset ground truth is malformed
            if gt_ans is None:
                continue

            for gen in output.outputs:
                completion = gen.text
                is_correct = False
                
                # grader
                reward_info = r1_zero_reward_fn(completion, gt_ans)
                if reward_info.get("reward", 0.0) > 0.5:
                    is_correct = True
                
                # try other function
                if not is_correct:
                    try:
                        fallback_info = question_only_reward_fn(completion, gt_ans)
                        if fallback_info.get("reward", 0.0) > 0.5:
                            is_correct = True
                    except Exception:
                        pass
                
                # fallback get <answer>...</answer>
                if not is_correct and "<answer>" in completion and "</answer>" in completion:
                    try:
                        raw_ans = completion.split("<answer>")[-1].split("</answer>")[0].strip()
                        if grade(raw_ans, gt_ans, fast=True):
                            is_correct = True
                    except Exception:
                        pass

                # save to dataset
                if is_correct:
                    # If base model got math right but missed tags, inject them
                    if "</think> <answer>" not in completion:
                        if "<answer>" in completion:
                            completion = completion.replace("<answer>", "</think> <answer>")
                        else:
                            completion = completion + f"\n</think> <answer>{gt_ans}</answer>"
                    
                    correct_count += 1
                    dsft_pairs.append({
                        "prompt": q,
                        "response": completion 
                    })

        print(f"Rollouts: {total_rollouts} | Correct: {correct_count} | Dsft size: {len(dsft_pairs)}")

        if len(dsft_pairs) == 0:
            print("No correct trajectories found, skipping SFT inner loop.")
            continue

        # SFT Training Loop
        dataset = EIDataset(dsft_pairs)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

        policy.train()
        train_step = 0

        for epoch in range(args.sft_epochs):
            optimizer.zero_grad()
            for step, batch in enumerate(dataloader):
                train_step += 1
                
                # Tokenize
                prompts = batch["prompts"]
                responses = batch["responses"]
                tokens = tokenize_prompt_and_output(prompts, responses, tokenizer)
                tokens = {k: v[:, :args.max_seq_len] for k, v in tokens.items()}
                
                input_ids = tokens["input_ids"].to(args.policy_device)
                labels = tokens["labels"].to(args.policy_device)
                response_mask = tokens["response_mask"].to(args.policy_device)
                
                num_tokens = response_mask.sum().item()
                if num_tokens == 0:
                    continue
                
                # Forward pass sft
                logits = policy(input_ids).logits
                
                # Calculate policy_log_probs
                log_probs_all = torch.nn.functional.log_softmax(logits, dim=-1)
                policy_log_probs = torch.gather(
                    log_probs_all, dim=-1, index=labels.unsqueeze(-1)
                ).squeeze(-1)
                
                # Calculate Token Accuracy
                with torch.no_grad():
                    preds = logits.argmax(dim=-1)
                    correct_mask = (preds == labels) * response_mask
                    tok_acc = correct_mask.sum().item() / num_tokens
                    # Calculate Entropy
                    token_entropy = compute_entropy(logits) # Get entropy for all tokens
                    masked_entropy = (token_entropy * response_mask).sum() / num_tokens # Mask and average
                
                # train step
                loss, metadata = sft_microbatch_train_step(
                    policy_log_probs=policy_log_probs,
                    response_mask=response_mask,
                    gradient_accumulation_steps=args.grad_accum,
                    normalize_constant=float(num_tokens)
                )
                
                # Optimizer step
                if (step + 1) % args.grad_accum == 0 or (step + 1) == len(dataloader):
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=args.max_grad_norm)
                    try:
                        del logits, log_probs_all, policy_log_probs, token_entropy
                    except NameError:
                        pass
                    torch.cuda.empty_cache()
                    optimizer.step()
                    optimizer.zero_grad()

                wandb.log({
                    "train/loss":           metadata["loss"].item(),
                    "train/token_accuracy": tok_acc,
                    "train/entropy":        masked_entropy.item(),
                    "train/num_tokens":     num_tokens,
                    "train/ei_step":        ei_step,
                    "train/epoch":          epoch + 1,
                    "train_step":           train_step,
                })

        # Validation
        print(f"  Running validation after EI step {ei_step} …")
        policy.eval()

        val_loss = evaluate_val_loss(
            policy, val_samples, tokenizer, args.policy_device, args.max_seq_len,
        )
        load_policy_into_vllm(policy, llm)
        val_acc = evaluate_accuracy(llm, val_samples, r1_zero_reward_fn)

        print(
            f"  Val loss: {val_loss:.4f}  |  "
            f"Val accuracy: {val_acc:.4f}  ({val_acc*100:.1f}%)"
        )

        wandb.log({
            "eval/loss":     val_loss,
            "eval/accuracy": val_acc,
            "ei_step":       ei_step,
        })

        policy.train()

    final_path = os.path.join(args.output_dir, "final_expertIteration_model")
    print(f"\nSaving final model to {final_path}")
    policy.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    wandb.finish()

if __name__ == "__main__":
    main()