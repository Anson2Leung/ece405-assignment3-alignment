import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
import numpy as np
from vllm import SamplingParams

def tokenize_prompt_and_output(prompt_strs: list[str], output_strs: list[str], tokenizer) -> dict[str, torch.Tensor]:
    batch_input_ids = []
    batch_labels = []
    batch_masks = []

    # Get padding and eos token, which will be replaced with 0
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_token_id = tokenizer.eos_token_id

    # tokenized prompt and output strings 
    tokenized = []
    prompt_and_output_lens = []
    for prompt_str, output_str in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer.encode(prompt_str)
        output_ids = tokenizer.encode(output_str, add_special_tokens=False)
        prompt_and_output_lens.append((len(prompt_ids) + len(output_ids)))
        tokenized.append((prompt_ids, output_ids))

    # max(prompt_and_output_lens) - 1
    target_len = max(prompt_and_output_lens) - 1

    for prompt_ids, output_ids in tokenized:
        # Concatenate prompt and output and append EOS
        concat = prompt_ids + output_ids + [eos_token_id]
        input_ids = concat[:-1]   # slice off final token
        label_ids = concat[1:]    # no first token

        # Mask prompt with 0 and output with 1 and the eos token with 0 
        mask = [0] * (len(prompt_ids) - 1) + [1] * len(output_ids) + [0]

        # Pad to max(prompt_and_output_lens) - 1
        # add pad token and 0 to mask
        cur_len = len(input_ids)
        if cur_len <= target_len:
            pad = target_len - cur_len
            input_ids = input_ids + [pad_token_id] * pad
            label_ids = label_ids + [pad_token_id] * pad
            mask  = mask  + [0] * pad
        else:
            input_ids = input_ids[:target_len]
            label_ids = label_ids[:target_len]
            mask  = mask[:target_len]

        batch_input_ids.append(input_ids)
        batch_labels.append(label_ids)
        batch_masks.append(mask)

    return {
        "input_ids":     torch.tensor(batch_input_ids, dtype=torch.long),
        "labels":        torch.tensor(batch_labels,    dtype=torch.long),
        "response_mask": torch.tensor(batch_masks,     dtype=torch.long),
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    # H(p) = -sum(p * log(p))
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    
    # Summing across dim=-1 reduces to a single value
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    
    # Get logits for next token (batch_size, sequence_length, vocab_size)
    logits = model(input_ids).logits
    
    # numerically stable log prob
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Gather the log prob for the actual tokens in labels log_probs shape after squeeze (batch_size, sequence_length)
    log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    result = {"log_probs": log_probs}
    
    if return_token_entropy:
        token_entropy = compute_entropy(logits)
        result["token_entropy"] = token_entropy
        
    return result


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:

    # Zero out the elements we want to ignore
    masked_tensor = tensor * mask
    
    # Sum tensor or dimension
    if dim is None:
        sum = torch.sum(masked_tensor)
    else:
        sum = torch.sum(masked_tensor, dim=dim)
        
    # Normalize by dividing by the provided constant
    return sum / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch for SFT.
    """

    # loss per token for response only (masked with 1)
    per_token_loss = -policy_log_probs * response_mask
    
    # Sum the valid losses and apply normalization
    microbatch_loss_sum = torch.sum(per_token_loss)
    normalized_loss = (microbatch_loss_sum / normalize_constant) / 2
    
    # Backward pass adjusted for Gradient Accumulation
    accumulated_loss = normalized_loss / gradient_accumulation_steps
    accumulated_loss.backward()
    
    # metadata with loss
    metadata = {
        "loss": normalized_loss.detach(),
        "num_tokens": response_mask.sum().detach(),
    }
    
    return accumulated_loss.detach(), metadata

def log_generations(
    llm, 
    policy_model, 
    tokenizer, 
    val_samples: list[dict], 
    reward_fn, 
    num_to_log: int = 8
) -> dict:
    """
    Generates completions, evaluates them, and computes statistics for logging.
    """
    # 1. Setup vLLM Generation
    # temperature=0.0 (greedy) is standard for validation to reduce variance
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024, stop=["</answer>"])
    
    prompts = [s["formatted_prompt"] for s in val_samples[:num_to_log]]
    solutions = [s["solution"] for s in val_samples[:num_to_log]]
    
    # 2. Batch Generation
    outputs = llm.generate(prompts, sampling_params)
    
    table_data = []
    all_stats = {
        "lengths": [], 
        "correct_lengths": [], 
        "incorrect_lengths": [], 
        "entropies": [], 
        "rewards": []
    }

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        # Ensure we keep the <think> tag if vLLM stripped it or if it's part of the response
        full_completion = generated_text 
        
        # 3. Reward Calculation
        # Assuming reward_fn returns a dict: {"total": float, "format": float, "answer": float}
        reward_info = reward_fn(full_completion, solutions[i])
        total_reward = reward_info.get("total", 0.0)
        
        # 4. Entropy Calculation
        # We must re-tokenize the generated text to get logits from the policy model
        enc = tokenizer(full_completion, return_tensors="pt").to(policy_model.device)
        with torch.no_grad():
            logits = policy_model(**enc).logits
            entropy_tensor = compute_entropy(logits) 
            avg_entropy = entropy_tensor.mean().item()

        # 5. Length Stats
        resp_len = len(output.outputs[0].token_ids)
        all_stats["lengths"].append(resp_len)
        all_stats["entropies"].append(avg_entropy)
        all_stats["rewards"].append(total_reward)
        
        if total_reward > 0.5: # Threshold for 'correct'
            all_stats["correct_lengths"].append(resp_len)
        else:
            all_stats["incorrect_lengths"].append(resp_len)

        # 6. Format data for a WandB Table
        table_data.append([
            prompts[i], 
            full_completion, 
            solutions[i], 
            total_reward, 
            reward_info.get("format", 0.0), 
            reward_info.get("answer", 0.0), 
            avg_entropy
        ])

    # 7. Aggregate Metrics
    metrics = {
        "val/avg_reward": np.mean(all_stats["rewards"]),
        "val/avg_entropy": np.mean(all_stats["entropies"]),
        "val/avg_response_length": np.mean(all_stats["lengths"]),
        "val/avg_len_correct": np.mean(all_stats["correct_lengths"]) if all_stats["correct_lengths"] else 0.0,
        "val/avg_len_incorrect": np.mean(all_stats["incorrect_lengths"]) if all_stats["incorrect_lengths"] else 0.0,
    }

    return {"table_data": table_data, "metrics": metrics}