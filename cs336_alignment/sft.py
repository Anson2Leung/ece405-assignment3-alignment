import argparse
import os
import random
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
    masked_normalize,
    sft_microbatch_train_step,
    tokenize_prompt_and_output,
    evaluate_accuracy,
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


# ─────────────────────────────────────────────
# Default paths
# ─────────────────────────────────────────────

DEFAULT_MATH_PATH   = "/home/ansonl32/ECE405/ece405-assignment3-alignment/data/math/math.parquet"
DEFAULT_PROMPT_FILE = "/home/ansonl32/ECE405/ece405-assignment3-alignment/cs336_alignment/prompts/r1_zero.prompt"
DEFAULT_OUTPUT_DIR  = "/home/ansonl32/koa_scratch/ECE405/assignment3/sft_experiment"
DEFAULT_MODEL_ID    = "/home/ansonl32/koa_scratch/ECE405/assignment3/Qwen/Qwen2.5-Math-1.5B"


# ─────────────────────────────────────────────
# Starter Code
# ─────────────────────────────────────────────

def init_vllm(
    model_id: str,
    device: str,
    seed: int,
    dtype: str = "auto",
    gpu_memory_utilization: float = 0.70,
) -> LLM:
    """Start a vLLM inference engine on a dedicated GPU."""
    vllm_set_random_seed(seed)

    if dtype == "auto":
        dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    print(f"Initializing vLLM with dtype: {dtype} on {device}")

    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=dtype,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM) -> None:
    """Sync current policy weights into the vLLM inference engine in-place."""
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def format_prompt(math_path: str, prompt_file: str) -> tuple[list[dict], list[str]]:
    with open(prompt_file, "r") as f:
        template = f.read()
    df = pd.read_parquet(math_path)

    if df.empty:
        raise ValueError(f"The dataset at {math_path} is empty. Check your data source or filtering.")

    examples: list[dict] = []
    formatted_prompts: list[str] = []

    for _, row in df.iterrows():
        ex = row.to_dict()
        problem_text = ex.get("problem", ex.get("question", ""))
        formatted = template.replace("{question}", problem_text)
        examples.append(ex)
        formatted_prompts.append(formatted)

    print(f"[format_prompt] Loaded {len(examples)} examples from {math_path}")
    return examples, formatted_prompts


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class MATHSFTDataset(Dataset):
    """
    Wraps formatted MATH examples for SFT.

    Each __getitem__ returns a dict with:
      "formatted_prompt" – full r1_zero prompt string
      "response"         – gold solution (SFT training target)
      "solution"         – same as response (for val compatibility with log_generations)
    """

    def __init__(
        self,
        examples: list[dict],
        formatted_prompts: list[str],
        split: str = "train",
        train_ratio: float = 0.9,
        max_samples: int = -1,
        seed: int = 42,
    ):
        assert len(examples) == len(formatted_prompts), "examples / prompts length mismatch"
        paired = list(zip(examples, formatted_prompts))

        rng = random.Random(seed)
        rng.shuffle(paired)

        n_train = int(len(paired) * train_ratio)
        paired = paired[:n_train] if split == "train" else paired[n_train:]

        # Sub-sampling for ablations: --max_samples 128|256|512|1024|-1
        if max_samples > 0 and max_samples < len(paired):
            paired = rng.sample(paired, max_samples)

        self.data: list[dict] = []
        for ex, fp in paired:
            solution = ex.get("solution", ex.get("answer", ""))
            answer = ex.get("answer", solution)
            sft_target = f"{solution}\n</think> <answer>{answer}</answer>"
            self.data.append({
                "formatted_prompt": fp,
                "response":         sft_target,
                "solution":         solution,
                "answer":           answer,
            })

        print(f"[MATHSFTDataset] split={split}, n={len(self.data)}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]


def collate_fn(batch: list[dict]) -> dict:
    return {
        "prompts":   [ex["formatted_prompt"] for ex in batch],
        "responses": [ex["response"]         for ex in batch],
    }


# ─────────────────────────────────────────────
# Validation loss
# ─────────────────────────────────────────────

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
    using the policy model directly.
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
            # Truncate to same length used during training
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


# ─────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT for Qwen Math on MATH dataset")
    p.add_argument("--model_id",    type=str, default=DEFAULT_MODEL_ID)
    p.add_argument("--math_path",   type=str, default=DEFAULT_MATH_PATH,
                   help="Path to math.parquet")
    p.add_argument("--prompt_file", type=str, default=DEFAULT_PROMPT_FILE,
                   help="Path to r1_zero.prompt template")
    p.add_argument("--output_dir",  type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--train_ratio", type=float, default=0.9,
                   help="Fraction used for training; remainder is validation")
    p.add_argument("--n_sft_steps",      type=int,   default=200,
                   help="Optimizer steps. ~200 ≈ 30 min on a T4 with these settings.")
    p.add_argument("--batch_size",       type=int,   default=1,
                   help="Sequences per microbatch (1 as required)")
    p.add_argument("--grad_accum_steps", type=int,   default=8,
                   help="Microbatches per optimiser step")
    p.add_argument("--lr",               type=float, default=2e-5)
    p.add_argument("--max_grad_norm",    type=float, default=1.0,
                   help="Gradient clipping norm")
    p.add_argument("--weight_decay",     type=float, default=0.01)
    p.add_argument("--warmup_steps",     type=int,   default=20)
    p.add_argument("--max_seq_len",      type=int,   default=512,
                   help="Hard truncation length to protect T4 memory")
    # Dataset sub-sampling {128, 256, 512, 1024, -1=full}
    p.add_argument("--max_samples", type=int, default=-1,
                   help="Max training examples. -1 = full dataset.")
    p.add_argument("--eval_every",       type=int, default=50)
    p.add_argument("--num_val_generate", type=int, default=200,
                   help="Val examples used for accuracy sweep")
    p.add_argument("--num_to_log",       type=int, default=8,
                   help="Examples written to the wandb generation table")
    p.add_argument("--policy_device", type=str, default="cuda:0")
    p.add_argument("--vllm_device",   type=str, default="cuda:1")
    p.add_argument("--vllm_mem",      type=float, default=0.70)
    p.add_argument("--seed",           type=int,  default=42)
    p.add_argument("--wandb_project",  type=str,  default="ece405_assignment3")
    p.add_argument("--wandb_run_name", type=str,  default=None)
    p.add_argument("--no_wandb",       action="store_true")
    p.add_argument("--save_every",     type=int,  default=0,
                   help="Intermediate checkpoint interval (0 = final only)")

    return p.parse_args()


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=1e-7)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


def truncate_batch(tokens: dict[str, torch.Tensor], max_len: int) -> dict[str, torch.Tensor]:
    # Truncate all tokenised tensors to max_len
    return {k: v[:, :max_len] for k, v in tokens.items()}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # WANDB setup init
    use_wandb = not args.no_wandb
    if use_wandb:
        run_name = args.wandb_run_name or (
            f"sft_n{args.max_samples}_lr{args.lr}"
            f"_bs{args.batch_size}x{args.grad_accum_steps}"
        )
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
        wandb.define_metric("train_step")
        wandb.define_metric("eval_step")
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("eval/*",  step_metric="eval_step")

    # Load and format datasets into prompt 
    print(f"Formatting prompts from {args.math_path}")
    examples, formatted_prompts = format_prompt(args.math_path, args.prompt_file)

    train_dataset = MATHSFTDataset(
        examples, formatted_prompts,
        split="train",
        train_ratio=args.train_ratio,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    val_dataset = MATHSFTDataset(
        examples, formatted_prompts,
        split="val",
        train_ratio=args.train_ratio,
        max_samples=-1,
        seed=args.seed,
    )
    # Only use a capped subset for fast periodic evaluation
    val_samples_for_eval = val_dataset.data[: args.num_val_generate]

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # Tokenizer and policy model
    print(f"Loading {args.model_id} on {args.policy_device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use bflaot16 if the GPU is capable
    major, _ = torch.cuda.get_device_capability(args.policy_device)
    if major >= 8:
        policy_dtype = torch.bfloat16
        vllm_dtype_str = "bfloat16"
    else:
        policy_dtype = torch.float16
        vllm_dtype_str = "float16"

    policy = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=policy_dtype,
        trust_remote_code=True,
        use_cache=False,
    ).to(args.policy_device)

    policy.gradient_checkpointing_enable()
    policy.train()

    # Optimizer and LR scheduler
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = build_lr_scheduler(optimizer, args.warmup_steps, args.n_sft_steps)

    # Load vLLM on the second GPU 
    print(f"Initialising vLLM on {args.vllm_device}")
    llm = init_vllm(
        args.model_id, args.vllm_device, args.seed,
        dtype=vllm_dtype_str,
        gpu_memory_utilization=args.vllm_mem,
    )

    # SFT training loop 
    print(
        f"\nStarting SFT training: {args.n_sft_steps} steps | "
        f"batch_size=1 × grad_accum={args.grad_accum_steps} | "
        f"lr={args.lr} | max_seq_len={args.max_seq_len}\n"
    )

    global_step = 0   # optimiser steps completed
    train_step  = 0   # wandb x-axis for train/*
    eval_step   = 0   # wandb x-axis for eval/*
    data_iter   = iter(train_loader)

    while global_step < args.n_sft_steps:

        # Accumulate gradients over grad_accum_steps microbatches 
        optimizer.zero_grad()

        accum_loss_sum        = 0.0
        accum_tokens          = 0
        accum_microbatches    = 0
        step_correct_tokens   = 0
        step_total_tokens     = 0

        for _ in range(args.grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            prompts   = batch["prompts"]    # list of 1 string (batch_size=1)
            responses = batch["responses"]  # list of 1 string

            # Tokenise prompt + response, build response_mask
            tokens = tokenize_prompt_and_output(prompts, responses, tokenizer)

            # Hard-truncate to protect memory
            tokens = truncate_batch(tokens, args.max_seq_len)

            input_ids     = tokens["input_ids"].to(args.policy_device)
            labels        = tokens["labels"].to(args.policy_device)
            response_mask = tokens["response_mask"].to(args.policy_device)

            n_resp_tokens = response_mask.sum().item()
            # Response was cut off by truncation
            if n_resp_tokens == 0:
                continue

            with torch.set_grad_enabled(True):
                logits = policy(input_ids).logits

            log_probs_all = torch.nn.functional.log_softmax(logits, dim=-1)
            policy_log_probs = torch.gather(
                log_probs_all, dim=-1, index=labels.unsqueeze(-1)
            ).squeeze(-1)

            # Train token accuracy: top-1 prediction vs label on response tokens
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                correct_mask = (preds == labels) * response_mask
                step_correct_tokens += correct_mask.sum().item()
                step_total_tokens   += n_resp_tokens

            # Backward: masked CE loss, normalised per-token + grad_accum
            _, meta = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=args.grad_accum_steps,
                normalize_constant=float(n_resp_tokens),
            )

            accum_loss_sum     += meta["loss"].item()
            accum_tokens       += meta["num_tokens"].item()
            accum_microbatches += 1

        # if all samples in microbatch wege degenerate in this step
        if accum_microbatches == 0:
            continue  

        # Gradient clip and optimizer step
        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        global_step   += 1
        train_step    += 1
        avg_loss       = accum_loss_sum / accum_microbatches
        current_lr     = scheduler.get_last_lr()[0]
        train_tok_acc  = (
            step_correct_tokens / step_total_tokens
            if step_total_tokens > 0 else 0.0
        )

        print(
            f"[step {global_step:4d}/{args.n_sft_steps}]  "
            f"loss={avg_loss:.4f}  tok_acc={train_tok_acc:.3f}  "
            f"tokens={accum_tokens}  lr={current_lr:.2e}"
        )

        if use_wandb:
            wandb.log({
                "train/loss":           avg_loss,
                "train/token_accuracy": train_tok_acc,
                "train/lr":             current_lr,
                "train/num_tokens":     accum_tokens,
                "train_step":           train_step,
            })

        # Periodic evaluation 
        if global_step % args.eval_every == 0:
            print(f"\n{'='*60}")
            print(f"Evaluating at step {global_step} …")
            policy.eval()

            # Validation loss
            val_loss = evaluate_val_loss(
                policy, val_samples_for_eval, tokenizer,
                args.policy_device, args.max_seq_len,
            )
            print(f"  Validation loss     : {val_loss:.4f}")

            # Validation accuracy
            load_policy_into_vllm_instance(policy, llm)
            accuracy = evaluate_accuracy(llm, val_samples_for_eval, r1_zero_reward_fn)
            print(f"  Validation accuracy : {accuracy:.4f}  ({accuracy*100:.1f}%)")

            # Qualitative generation table (log_generations from sft_helpers)
            log_out = log_generations(
                llm=llm,
                policy_model=policy,
                tokenizer=tokenizer,
                val_samples=val_samples_for_eval,
                reward_fn=r1_zero_reward_fn,
                num_to_log=args.num_to_log,
            )

            # Remap "val/*" → "eval/*"
            eval_metrics = {
                k.replace("val/", "eval/"): v
                for k, v in log_out["metrics"].items()
            }
            eval_metrics["eval/accuracy"] = accuracy
            eval_metrics["eval/loss"]     = val_loss

            eval_step += 1
            if use_wandb:
                table = wandb.Table(
                    columns=["raw_prompt", "question", "solution", "generated answer", "reward"],
                    data=log_out["table_data"],
                )
                
                wandb.log({
                    **eval_metrics, 
                    "eval/generations": table, 
                    "eval_step": eval_step,
                    "step": global_step 
                })

            policy.train()
            print(f"{'='*60}\n")

        # Intermediate checkpoint
        if args.save_every > 0 and global_step % args.save_every == 0:
            ckpt = os.path.join(args.output_dir, f"checkpoint-step{global_step}")
            print(f"Saving checkpoint → {ckpt}")
            policy.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)

    # Final model 
    final_path = os.path.join(args.output_dir, "final")
    print(f"\nSaving final model → {final_path}")
    policy.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    # Final evaluation 
    # Only runs if no evaluation at the final step
    if args.n_sft_steps % args.eval_every != 0:
        print("Running final evaluation")
        policy.eval()
        final_val_loss = evaluate_val_loss(
            policy, val_samples_for_eval, tokenizer,
            args.policy_device, args.max_seq_len,
        )
        load_policy_into_vllm_instance(policy, llm)
        final_acc = evaluate_accuracy(llm, val_samples_for_eval, r1_zero_reward_fn)
        print(f"Final validation loss     : {final_val_loss:.4f}")
        print(f"Final validation accuracy : {final_acc:.4f}  ({final_acc*100:.1f}%)")
    
        if use_wandb:
            eval_step += 1
            wandb.log({
                "eval/accuracy": final_acc,
                "eval/loss":     final_val_loss,
                "eval_step":     eval_step,
            })
            wandb.finish()
            
    else:
        print("Skipping final evaluation.")

    print("\nSFT training complete.")


if __name__ == "__main__":
    main()