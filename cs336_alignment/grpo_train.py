import os
import json
import random
import torch
import wandb
import typer
import pandas as pd
from typing import Literal, List, Dict, Any
from unittest.mock import patch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from cs336_alignment.sft_helpers import tokenize_prompt_and_output, get_response_log_probs, log_generations
from cs336_alignment.grpo import compute_group_normalized_rewards, grpo_microbatch_train_step, masked_mean
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn, extract_answer
from cs336_alignment.math_baseline import format_prompt

app = typer.Typer()

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    vllm_set_random_seed(seed)
    # Auto-detect dtype bfloat16 
    vllm_dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    print(f"vLLM dtype: {vllm_dtype}")
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=vllm_dtype,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


@app.command()
def train_grpo(
    model_id: str = "/home/ansonl32/koa_scratch/ECE405/assignment4/Qwen/Qwen2.5-Math-1.5B",
    data_path: str = "/home/ansonl32/ECE405/ece405-assignment3-alignment/data/math/math.parquet",
    output_dir: str = "/home/ansonl32/koa_scratch/ECE405/assignment3/qwen_grpo_final",
    n_grpo_steps: int = 200,
    learning_rate: float = 1e-5,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 256,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,
    sampling_max_tokens: int = 1024,
    epochs_per_rollout_batch: int = 1,
    train_batch_size: int = 256,
    gradient_accumulation_steps: int = 128,
    gpu_memory_utilization: float = 0.85,
    loss_type: str = "reinforce_with_baseline",
    cliprange: float = 0.2,
    eval_interval: int = 10,
    seed: int = 42,
    prompt_file: str = "/home/ansonl32/ECE405/ece405-assignment3-alignment/cs336_alignment/prompts/r1_zero.prompt",
    wandb_project: str = "ece405_assignment3",
    wandb_run_name: str = "grpo_train",
    use_std_normalization: bool = typer.Option(True, "--use-std-normalization/--no-use-std-normalization"),
    use_length_normalization: bool = typer.Option(False, "--use-length-normalization/--no-use-length-normalization"),
    reward_type: str = typer.Option("r1_zero", "--reward-type"),
):
    # Assertions and setup
    assert train_batch_size % gradient_accumulation_steps == 0, "train_batch_size must be divisible by gradient_accumulation_steps"
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    
    assert rollout_batch_size % group_size == 0, "rollout_batch_size must be divisible by group_size"
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    
    assert train_batch_size >= group_size, "train_batch_size must be greater than or equal to group_size"
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size

    train_device = "cuda:0"
    eval_device = "cuda:1"
    
    wandb.init(project=wandb_project, name=wandb_run_name, config=locals())
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*",  step_metric="eval_step")

    # Format prompt
    examples, formatted_prompts_all = format_prompt(data_path, prompt_file)
    random.seed(seed)
    combined = list(zip(examples, formatted_prompts_all))
    random.shuffle(combined)
    examples_shuffled, prompts_shuffled = zip(*combined)

    # Reserve 1024 for validation
    val_data   = [{"example": e, "prompt": p} for e, p in
                  zip(examples_shuffled[:1024], prompts_shuffled[:1024])]
    train_data = [{"example": e, "prompt": p} for e, p in
                  zip(examples_shuffled[1024:],  prompts_shuffled[1024:])]

    print(f"Train: {len(train_data)}  Val: {len(val_data)}")

    if reward_type == "r1_zero":
        active_reward_fn = r1_zero_reward_fn
    elif reward_type == "question_only":
        active_reward_fn = question_only_reward_fn
    else:
        raise ValueError(f"Unknown reward_type: {reward_type}")
    print(f"Using reward function: {reward_type}")

    # model intialization
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Initializing vLLM on {eval_device}")
    with torch.cuda.device(eval_device):
        vllm_engine = init_vllm(model_id, device=eval_device, seed=seed, gpu_memory_utilization=gpu_memory_utilization)

    train_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.cuda.device(train_device):
        policy_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=train_dtype).to(train_device)
        policy_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        policy_model.config.use_cache = False
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate, weight_decay=0.0, betas=(0.9, 0.95))

    if len(tokenizer) > policy_model.config.vocab_size:
        print(f"Resizing embeddings from {policy_model.config.vocab_size} to {len(tokenizer)}")
        policy_model.resize_token_embeddings(len(tokenizer))
    
    sampling_params = SamplingParams(
        temperature=sampling_temperature,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        n=group_size
    )

    eval_sampling_params = SamplingParams(
        temperature=0.0, 
        max_tokens=sampling_max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    # Train loop
    for step in range(1, n_grpo_steps + 1):
        policy_model.train()
        
        # Sample
        batch = random.sample(train_data, n_prompts_per_rollout_batch)
        prompts = [ex["prompt"] for ex in batch]
        # Extract boxed answer for grading
        gt_answers = [
            ex["example"].get("answer", None) or extract_answer(ex["example"].get("solution", ""))
            for ex in batch
        ]
        
        load_policy_into_vllm_instance(policy_model, vllm_engine)
        
        # Rollout
        vllm_outputs = vllm_engine.generate(prompts, sampling_params)
        
        rollout_responses = []
        repeated_ground_truths = []
        repeated_prompts = []
        
        for i, req_output in enumerate(vllm_outputs):
            for gen in req_output.outputs:
                rollout_responses.append(gen.text)
                repeated_ground_truths.append(gt_answers[i])
                repeated_prompts.append(prompts[i])

        # reward and advantage
        advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
            reward_fn=active_reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=use_std_normalization
        )
        
        wandb.log({f"train/{k}": v for k, v in reward_metadata.items()} | {"train_step": step})

        # Off policy
        old_log_probs_cache = []
        if epochs_per_rollout_batch > 1 or loss_type in ["grpo_clip", "grpo_no_clip"]:
            policy_model.eval()
            with torch.inference_mode():
                for i in range(n_microbatches_per_rollout_batch):
                    start_idx = i * micro_train_batch_size
                    end_idx   = start_idx + micro_train_batch_size
                    
                    mb_prompts = repeated_prompts[start_idx:end_idx]
                    mb_responses = rollout_responses[start_idx:end_idx]
                    mb_tokens = tokenize_prompt_and_output(mb_prompts, mb_responses, tokenizer)
                    
                    log_prob_data = get_response_log_probs(
                        policy_model, 
                        mb_tokens["input_ids"].to(train_device), 
                        mb_tokens["labels"].to(train_device)
                    )
                    old_log_probs_cache.append(log_prob_data["log_probs"].detach())
            policy_model.train()

        # Optimization Loop
        for epoch in range(epochs_per_rollout_batch):
            optimizer.zero_grad()   # reset
            accumulated_loss = 0.0

            for i in range(n_microbatches_per_rollout_batch):
                start_idx = i * micro_train_batch_size
                end_idx   = start_idx + micro_train_batch_size

                mb_prompts   = repeated_prompts[start_idx:end_idx]
                mb_responses = rollout_responses[start_idx:end_idx]
                mb_adv       = advantages[start_idx:end_idx].to(train_device).view(-1, 1)
                mb_raw       = raw_rewards[start_idx:end_idx].to(train_device).view(-1, 1)

                mb_old_log_probs = old_log_probs_cache[i] if old_log_probs_cache else None

                mb_tokens     = tokenize_prompt_and_output(mb_prompts, mb_responses, tokenizer)
                input_ids     = mb_tokens["input_ids"].to(train_device)
                response_mask = mb_tokens["response_mask"].to(train_device)
                labels        = mb_tokens["labels"].to(train_device)

                # forward pass
                logits        = policy_model(input_ids).logits
                log_probs_all = torch.nn.functional.log_softmax(logits, dim=-1)
                policy_log_probs = torch.gather(
                    log_probs_all, dim=-1, index=labels.unsqueeze(-1)
                ).squeeze(-1)

                loss, micro_metadata = grpo_microbatch_train_step(
                    policy_log_probs=policy_log_probs,
                    response_mask=response_mask,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    loss_type=loss_type,
                    raw_rewards=mb_raw,
                    advantages=mb_adv,
                    old_log_probs=mb_old_log_probs,
                    cliprange=cliprange,
                    use_length_normalization=use_length_normalization
                )
                accumulated_loss += loss.item()

                # Log microbatch
                if i == 0:
                    log_dict = {}
                    if "clip_fraction" in micro_metadata:
                        log_dict["train/clip_fraction"] = micro_metadata["clip_fraction"].item()
                    with torch.no_grad():
                        prob_all  = torch.exp(log_probs_all)
                        entropy   = -(prob_all * log_probs_all).sum(dim=-1)
                        log_dict["train/entropy"] = masked_mean(entropy, response_mask).item()
                    wandb.log(log_dict | {"train_step": step, "epoch": epoch})

            # Optimizer step 
            grad_norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            wandb.log({
                "train/loss":      accumulated_loss,
                "train/grad_norm": grad_norm.item(),
                "train_step":      step,
            })

        # Evaluation
        if step % eval_interval == 0:
            print(f"--- Step {step}: Evaluating on {len(val_data)} examples ---")
            policy_model.eval()
            load_policy_into_vllm_instance(policy_model, vllm_engine)
            
            val_prompts  = [ex["prompt"] for ex in val_data]
            val_answers  = [
                ex["example"].get("answer", None) or extract_answer(ex["example"].get("solution", ""))
                for ex in val_data
            ]

            val_outputs = vllm_engine.generate(val_prompts, eval_sampling_params)            
            # --- debug generation failed ---
            if len(val_outputs) == 0:
                print(f"[ERROR] vLLM generated 0 outputs for {len(val_prompts)} prompts!")

            val_rewards, val_format, val_answer, val_correct = [], [], [], []            
            for i, (output, ans) in enumerate(zip(val_outputs, val_answers)):
                completion = output.outputs[0].text
                score = active_reward_fn(completion, ans)
                
                # Directly using the exact keys your function returns
                val_rewards.append(score["reward"])
                val_format.append(score["format_reward"])
                val_answer.append(score["answer_reward"])
                
                # Track if the math answer was correct
                val_correct.append(int(score["answer_reward"] > 0.5))

                # --- debug print generation ---
                if i == 0:
                    print(f"\n[DEBUG EVAL] Ground Truth: {ans}")
                    print(f"[DEBUG EVAL] Model Completion:\n{completion}")
                    print(f"[DEBUG EVAL] Score Dict: {score}\n")

            # Log evaluation
            eval_step = step // eval_interval
            n_eval = len(val_correct)
            
            if n_eval > 0:
                wandb.log({
                    "eval/avg_reward":    sum(val_rewards)  / n_eval,
                    "eval/format_reward": sum(val_format)   / n_eval,
                    "eval/answer_reward": sum(val_answer)   / n_eval,
                    "eval/accuracy":      sum(val_correct)  / n_eval,
                    "eval_step":          eval_step,
                })
                print(f"  eval/accuracy={sum(val_correct)/n_eval:.4f}  "
                      f"eval/avg_reward={sum(val_rewards)/n_eval:.4f}")
            else:
                print("  [ERROR] No evaluations were logged because the evaluation list was empty.")

    # Save
    print(f"Saving to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    policy_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    wandb.finish()

if __name__ == "__main__":
    app()