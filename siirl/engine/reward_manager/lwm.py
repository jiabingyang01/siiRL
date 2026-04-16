# Copyright 2025 LWM-Reward Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
#
# LWM Reward Manager: Distributes step-level dense rewards from LWM-Reward.
#
# Key difference from EmbodiedRewardManager:
#   - EmbodiedRewardManager places a single score at the terminal step
#   - LWMRewardManager places per-step rewards at each action step
#
# This enables step-level advantage estimation (GAE or step-level GRPO),
# which is critical for long-horizon tasks.

import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger
from tensordict import TensorDict
from transformers import PreTrainedTokenizer


class LWMRewardManager:
    """
    Reward manager for LWM-Reward step-level dense rewards.

    Unlike EmbodiedRewardManager which places a single trajectory-level score
    at the terminal step, this manager distributes per-step rewards across
    the full response sequence.
    """

    def __init__(
        self,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        num_examine: int = 1,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = "data_source",
        **reward_kwargs,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.rank = int(os.environ.get("RANK", "0"))
        self.print_count = 0

        # LWM-specific parameters
        self.action_token_len = reward_kwargs.get("action_token_len", 7)
        self.reward_coef = reward_kwargs.get("reward_coef", 1.0)

    def __call__(
        self, data: TensorDict, return_dict: bool = False
    ) -> Union[Dict[str, Any], Tuple[Dict[str, torch.Tensor], Dict[str, float]]]:
        """
        Calculate and distribute step-level rewards.

        The key difference from EmbodiedRewardManager: rewards are placed at
        EACH step, not just the terminal step.
        """
        batch_size = data["responses"].shape[0]

        # --- Step 1: Compute step-level rewards ---
        if self.compute_score is None:
            scores_info = [
                {"score": 0.0, "format_correctness": 1.0, "is_success": False, "step_rewards": []}
                for _ in range(batch_size)
            ]
        else:
            scores_info = self.compute_score(batch_data=data)

        verifier_scores = [info["score"] for info in scores_info]
        format_scores = [info.get("format_correctness", 1.0) for info in scores_info]

        # --- Step 2: Distribute step-level rewards ---
        verifier_rewards = torch.zeros_like(data["responses"], dtype=torch.float32)
        verifier_rewards = verifier_rewards.view(batch_size, -1)

        finish_steps = data["finish_step"].cpu().numpy()

        for i in range(batch_size):
            step_rewards = scores_info[i].get("step_rewards", [])
            n_steps = int(finish_steps[i])

            if step_rewards and n_steps > 0:
                # Distribute step-level rewards across action token positions
                # Each env step corresponds to action_token_len tokens in the response
                for t, r_t in enumerate(step_rewards):
                    if t >= n_steps:
                        break
                    # Place reward at the last token of this step's action chunk
                    token_idx = (t + 1) * self.action_token_len - 1
                    if token_idx < verifier_rewards.shape[1]:
                        verifier_rewards[i, token_idx] = r_t
            else:
                # Fallback: use trajectory-level score at terminal (like SRPO)
                last_step_idx = n_steps * self.action_token_len - 1
                if last_step_idx >= 0 and last_step_idx < verifier_rewards.shape[1]:
                    verifier_rewards[i, last_step_idx] = verifier_scores[i]

        # --- Step 3: Debug logging ---
        if self.rank == 0 and self.print_count < self.num_examine:
            logger.info("--- LWMRewardManager Step-Level Reward Example ---")
            for i in range(min(batch_size, 2)):
                info = scores_info[i]
                n_steps = int(finish_steps[i])
                step_rewards = info.get("step_rewards", [])
                progress = info.get("progress_curve", [])
                logger.info(f"Sample {i} | Success: {info.get('is_success')} | Steps: {n_steps}")
                logger.info(f"  - Traj score: {info.get('score', 0.0):.4f}")
                if step_rewards:
                    logger.info(f"  - Step rewards (first 5): {[f'{r:.4f}' for r in step_rewards[:5]]}")
                    logger.info(f"  - Step rewards (last 5):  {[f'{r:.4f}' for r in step_rewards[-5:]]}")
                if progress:
                    logger.info(f"  - Progress (first 5): {[f'{p:.4f}' for p in progress[:5]]}")
                    logger.info(f"  - Progress (last 5):  {[f'{p:.4f}' for p in progress[-5:]]}")
            self.print_count += 1

        # --- Step 4: Aggregate rewards and metrics ---
        reward_tensor_dict = {"gt_scores": verifier_rewards}
        reward_metrics = {}

        final_reward_tensor = torch.zeros_like(verifier_rewards)
        if self.reward_coef != 0:
            final_reward_tensor += self.reward_coef * reward_tensor_dict["gt_scores"]
            reward_metrics["verifier_mean"] = torch.tensor(verifier_scores).mean().item()
            reward_metrics["format_correctness_mean"] = torch.tensor(format_scores).mean().item()

            # LWM-specific metrics
            all_step_rewards = []
            for info in scores_info:
                all_step_rewards.extend(info.get("step_rewards", []))
            if all_step_rewards:
                reward_metrics["lwm_step_reward_mean"] = float(np.mean(all_step_rewards))
                reward_metrics["lwm_step_reward_std"] = float(np.std(all_step_rewards))

        reward_tensor_dict["all"] = final_reward_tensor
        reward_metrics["reward_all"] = final_reward_tensor.sum(dim=-1).mean().item()

        if return_dict:
            return {
                "reward_tensor": reward_tensor_dict["all"],
                "reward_extra_info": {
                    "verifier_score": verifier_scores,
                    "format_correctness": format_scores,
                },
            }
        else:
            return reward_tensor_dict, reward_metrics

    def verify(self, data: TensorDict):
        """Validation: same as EmbodiedRewardManager (binary success)."""
        completes = data["complete"].tolist()
        batch_size = data["responses"].size(0)
        score = [float(item) for item in completes]

        device = data["responses"].device
        acc_tensor = torch.tensor(score, dtype=torch.float32, device=device)
        format_tensor = torch.ones(batch_size, dtype=torch.float32, device=device)

        data["acc"] = acc_tensor
        data["format_correctness"] = format_tensor

        success_rate = acc_tensor.mean().item()
        reward_metrics = {"all": success_rate}
        format_metrics = {"all": 1.0}
        reward_format_metrics = {"all": success_rate}

        return score, reward_metrics, format_metrics, reward_format_metrics
