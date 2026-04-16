# Copyright 2025 LWM-Reward Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
#
# LWM-Reward: Step-level action-causal dense reward via latent world model.
#
# Reward formula (Policy-Invariant Reward Shaping, Ng 1999 + Robo-Dopamine):
#   r_t = r_gold + gamma * Phi(t) - Phi(t-1)
# where Phi(t) = 1 - d(P_phi(z_t, a_t), z_g) / d(z_0, z_g)
#
# This module computes step-level rewards from:
#   - Per-step V-JEPA 2 clip embeddings (z_t)
#   - Per-step actions (a_t)  
#   - Goal embedding (z_g, from expert demo terminal clip)
#   - Dynamics predictor (P_phi, predicts z_{t+1} from z_t, a_t)

import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from loguru import logger

# Handle different tensordict versions
try:
    from tensordict import NonTensorData
except ImportError:
    from tensordict.tensorclass import NonTensorData


def compute_lwm_reward(
    batch_data,
    gamma: float = 0.99,
    success_threshold: float = 0.05,
    reward_scale: float = 1.0,
    dynamics_predictor=None,
    goal_embeddings: Optional[Dict[str, np.ndarray]] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Compute step-level LWM-Reward for each trajectory in the batch.

    Unlike SRPO's compute_embodied_reward which returns one score per trajectory,
    this returns per-trajectory metadata including step-level rewards.

    The actual step-level reward distribution is handled by LWMRewardManager.

    Args:
        batch_data: TensorDict from rollout containing:
            - "complete": (num_envs,) bool - task completion
            - "finish_step": (num_envs,) int - steps taken
            - "lwm_step_embeddings": (num_envs, max_steps, embed_dim) - per-step V-JEPA clip embeddings
            - "lwm_step_actions": (num_envs, max_steps, action_dim) - per-step raw actions
            - "lwm_goal_embeddings": (num_envs, embed_dim) - goal embeddings
        gamma: Discount factor for PBRS (default 0.99)
        success_threshold: Phi threshold for auto-detecting success (default 0.05)
        reward_scale: Scaling factor for rewards
        dynamics_predictor: Trained LatentDynamicsPredictor (optional, if None uses state-only progress)
        goal_embeddings: Dict mapping task_name -> goal embedding (fallback if not in batch)

    Returns:
        List of dicts with per-trajectory reward info including step-level rewards.
    """
    # --- Extract data ---
    batch_size = batch_data["complete"].shape[0]
    completes = batch_data["complete"].cpu().numpy().astype(bool)
    finish_steps = batch_data["finish_step"].cpu().numpy()

    # Fallback: if LWM keys are missing (e.g. batch re-partitioned and stripped
    # them), degrade to binary outcome reward instead of crashing.
    required_keys = ["lwm_step_embeddings", "lwm_step_actions", "lwm_goal_embeddings"]
    missing = [k for k in required_keys if k not in batch_data.keys()]
    if missing:
        logger.warning(f"[LWM-REWARD] Missing keys {missing}, falling back to binary outcome reward.")
        results = []
        for i in range(batch_size):
            results.append({
                "is_success": bool(completes[i]),
                "score": 1.0 if completes[i] else 0.0,
                "format_correctness": 1.0,
                "step_rewards": [],
                "progress_curve": [],
            })
        return results

    # Step-level embeddings: (num_envs, max_steps, embed_dim)
    step_embeddings = batch_data["lwm_step_embeddings"].cpu().numpy()
    # Step-level actions: (num_envs, max_steps, action_dim)
    step_actions = batch_data["lwm_step_actions"]  # keep as tensor for dynamics predictor
    # Goal embeddings: (num_envs, embed_dim)
    goal_embs = batch_data["lwm_goal_embeddings"].cpu().numpy()

    results = []

    for i in range(batch_size):
        n_steps = int(finish_steps[i])
        if n_steps <= 0:
            results.append({
                "is_success": bool(completes[i]),
                "score": 1.0 if completes[i] else 0.0,
                "format_correctness": 1.0,
                "step_rewards": [],
                "progress_curve": [],
            })
            continue

        z_g = goal_embs[i]  # (embed_dim,)
        z_0 = step_embeddings[i, 0]  # (embed_dim,)
        d_init = np.linalg.norm(z_0 - z_g) + 1e-8

        # Compute progress and rewards for each step
        progress_curve = []
        step_rewards = []
        phi_prev = 0.0  # Phi(0) = 0 by definition

        for t in range(n_steps):
            z_t = step_embeddings[i, t]  # (embed_dim,)

            if dynamics_predictor is not None:
                # Action-causal: use dynamics predictor
                z_t_tensor = torch.tensor(z_t, dtype=torch.float32).unsqueeze(0)
                a_t_tensor = step_actions[i, t].unsqueeze(0)
                if z_t_tensor.device != next(dynamics_predictor.parameters()).device:
                    device = next(dynamics_predictor.parameters()).device
                    z_t_tensor = z_t_tensor.to(device)
                    a_t_tensor = a_t_tensor.to(device)

                with torch.no_grad():
                    z_next_pred = dynamics_predictor(z_t_tensor, a_t_tensor)
                z_next = z_next_pred.squeeze(0).cpu().numpy()
            else:
                # Fallback: use next step's actual embedding (no dynamics)
                if t + 1 < n_steps:
                    z_next = step_embeddings[i, t + 1]
                else:
                    z_next = z_t

            # Progress: Phi(t) = 1 - d(z_next, z_g) / d(z_0, z_g)
            d_next_to_goal = np.linalg.norm(z_next - z_g)
            phi_t = 1.0 - d_next_to_goal / d_init
            phi_t = float(np.clip(phi_t, -1.0, 2.0))  # Soft clip to avoid extreme values

            progress_curve.append(phi_t)

            # Policy-invariant reward shaping: r_t = r_gold + gamma * Phi(t) - Phi(t-1)
            r_gold = 0.0
            if completes[i] and t == n_steps - 1:
                r_gold = 1.0
            elif phi_t >= (1.0 - success_threshold):
                r_gold = 1.0

            r_t = r_gold + gamma * phi_t - phi_prev
            step_rewards.append(float(r_t) * reward_scale)

            phi_prev = phi_t

        # Trajectory-level score (for compatibility with SRPO metrics)
        # Use final progress as trajectory score, mapped to [0, 1]
        final_progress = progress_curve[-1] if progress_curve else 0.0
        traj_score = max(0.0, min(1.0, final_progress))
        if completes[i]:
            traj_score = 1.0

        results.append({
            "is_success": bool(completes[i]),
            "score": traj_score,
            "format_correctness": 1.0,
            "step_rewards": step_rewards,
            "progress_curve": progress_curve,
        })

    # --- Logging ---
    scores = [r["score"] for r in results]
    n_success = sum(1 for r in results if r["is_success"])
    avg_steps = np.mean([len(r["step_rewards"]) for r in results])
    logger.info(
        f"[LWM-REWARD] batch={batch_size}, success={n_success}, "
        f"avg_score={np.mean(scores):.4f}, avg_steps={avg_steps:.1f}, "
        f"dynamics={'yes' if dynamics_predictor is not None else 'no'}"
    )

    return results
