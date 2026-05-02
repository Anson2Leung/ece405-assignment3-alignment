import torch
from typing import Callable, List, Dict, Any, Tuple, Literal, Optional

def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], Dict[str, float]],
    rollout_responses: List[str],
    repeated_ground_truths: List[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    
    raw_rewards_list = []
    format_rewards_list = []
    answer_rewards_list = []
    
    for response, truth in zip(rollout_responses, repeated_ground_truths):
        score_dict = reward_fn(response, truth)
        raw_rewards_list.append(score_dict.get("reward", 0.0))
        format_rewards_list.append(score_dict.get("format_reward", 0.0))
        answer_rewards_list.append(score_dict.get("answer_reward", 0.0))

    raw_rewards = torch.tensor(raw_rewards_list, dtype=torch.float32)
    format_rewards = torch.tensor(format_rewards_list, dtype=torch.float32)
    answer_rewards = torch.tensor(answer_rewards_list, dtype=torch.float32)

    grouped_rewards = raw_rewards.view(-1, group_size)
    group_means = grouped_rewards.mean(dim=1, keepdim=True)
    
    # prevents NaN when group_size=1 during evaluation
    if normalize_by_std and group_size > 1:
        group_stds = grouped_rewards.std(dim=1, keepdim=True)
        grouped_advantages = (grouped_rewards - group_means) / (group_stds + advantage_eps)
    else:
        grouped_advantages = grouped_rewards - group_means
    
    advantages = grouped_advantages.reshape(-1)
    
    metadata = {
        "reward/mean": raw_rewards.mean().item(),
        "reward/std": raw_rewards.std().item() if len(raw_rewards) > 1 else 0.0,
        "format_reward/mean": format_rewards.mean().item(),
        "answer_reward/mean": answer_rewards.mean().item(),
        # get average
        "accuracy": (answer_rewards > 0.5).float().mean().item(), 
        "advantage/mean": advantages.mean().item(),
        "advantage/std": advantages.std().item() if len(advantages) > 1 else 0.0,
    }
    
    return advantages, raw_rewards, metadata

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:

    # policy-gradient loss (-A * log_p).
    per_token_loss = -(raw_rewards_or_advantages * policy_log_probs)
    return per_token_loss


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    # Ratio of new policy vs old policy πθ(ot|q, o<t)/ πθold (ot|q, o<t)
    ratio = torch.exp(policy_log_probs - old_log_probs)
    
    # Ratio * Advantage
    left = ratio * advantages
    
    # Clipped ratio * Advantage
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    right = clipped_ratio * advantages
    
    # -min()
    loss = -torch.min(left, right)
    
    # clipping metadata
    # This tensor is 1.0 where right was lower than left (clipping was active) and 0.0 otherwise.
    is_clipped = (right < left).to(torch.float32)
    
    metadata = {
        "clip_fraction": is_clipped.mean(),
        "is_clipped": is_clipped 
    }
    
    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    metadata = {}
    batch_size, seq_len = policy_log_probs.shape

    if loss_type == "no_baseline":
        assert raw_rewards is not None, "'raw_rewards Required if loss_type == \"no_baseline\"; shape (batch_size, 1).'"
        assert raw_rewards.shape == (batch_size, 1), f"Expected (batch, 1), got {raw_rewards.shape}"
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
    
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None, "'advantages Required for \"reinforce_with_baseline\" and \"grpo_clip\"; shape (batch_size, 1).'"
        assert advantages.shape == (batch_size, 1), f"Expected (batch, 1), got {advantages.shape}"
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)

    elif loss_type == "grpo_clip":
        assert advantages is not None, "'advantages Required for \"reinforce_with_baseline\" and \"grpo_clip\"; shape (batch_size, 1).'"
        assert advantages.shape == (batch_size, 1), f"Expected advantages (batch, 1), got {advantages.shape}"
        assert old_log_probs is not None, "'old_log_probs Required for \"grpo_clip\"; shape (batch_size, sequence_length).'"
        assert old_log_probs.shape == (batch_size, seq_len), f"Expected old_log_probs {policy_log_probs.shape}, got {old_log_probs.shape}"
        assert cliprange is not None, "'cliprange Required for \"grpo_clip\"; scalar ϵ used for clipping.'"

        loss, clip_metadata = compute_grpo_clip_loss(
            advantages, 
            policy_log_probs, 
            old_log_probs, 
            cliprange
        )
        metadata.update(clip_metadata)

    elif loss_type == "grpo_no_clip":
        # needs advantage, policy, old policy
        assert advantages is not None, "'advantages Required for \grpo_no_clip\; shape (batch_size, 1).'"
        assert old_log_probs is not None, "'old_log_probs Required for \"grpo_clip\"; shape (batch_size, sequence_length).'"
        
        ratio = torch.exp(policy_log_probs - old_log_probs)
        loss = -(ratio * advantages) 

    else:
        raise ValueError(f"Invalid loss_type: {loss_type}")

    return loss, metadata


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: Optional[int] = None,
) -> torch.Tensor:

    # Multiply mask to get values with mask == 1
    masked_tensor = tensor * mask
    
    if dim is not None:
        # Average loss over dimension
        total_sum = masked_tensor.sum(dim=dim)
        count = mask.sum(dim=dim)
        
        return total_sum / count
    else:
        # Average loss over  all masked elements.
        total_sum = masked_tensor.sum()
        count = mask.sum()
        
        return total_sum / count


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"],
    raw_rewards: Optional[torch.Tensor] = None,
    advantages: Optional[torch.Tensor] = None,
    old_log_probs: Optional[torch.Tensor] = None,
    cliprange: Optional[float] = None,
    use_length_normalization: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    # compute the per-token loss
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    
    # masked_mean to aggregate to a scalar loss per example
    masked_tensor = per_token_loss * response_mask
    if use_length_normalization:
        # Average over the sequence length
        loss_per_example = masked_tensor.sum(dim=1) / response_mask.sum(dim=1)
    else:
        # Sum over the sequence length (Lambert 2024 formulation)
        loss_per_example = masked_tensor.sum(dim=1)
    
    # average over the batch dimension
    avg_batch_loss = loss_per_example.mean()
    
    # adjust for gradient accumulation
    scaled_loss = avg_batch_loss / gradient_accumulation_steps
    
    # backward pass
    scaled_loss.backward()
    
    # metadata for loss 
    metadata["loss/scaled_microbatch"] = scaled_loss.detach()
    metadata["loss/unscaled_microbatch"] = avg_batch_loss.detach()
    
    return scaled_loss, metadata