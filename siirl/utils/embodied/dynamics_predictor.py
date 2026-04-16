# Copyright 2025 LWM-Reward Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
#
# Dynamics Predictor: Action-conditioned single-step latent state predictor.
# Given current V-JEPA 2 latent z_t and action a_t, predicts next latent z_{t+1}.
# Trained with L1 loss on DROID/OXE data. Used by LWM-Reward for action-causal reward.

import math
import torch
import torch.nn as nn


class LatentDynamicsPredictor(nn.Module):
    """
    Action-conditioned single-step latent dynamics predictor.

    Architecture: Lightweight transformer that takes patch-level V-JEPA features
    and an action token, predicts next-step patch-level features.

    For pooled (global) embeddings (dim=D), we treat the input as a single token
    and predict the next embedding directly. This is the default mode when used
    with VideoEmbeddingModel which outputs mean-pooled embeddings.
    """

    def __init__(
        self,
        latent_dim: int = 1408,   # V-JEPA 2 ViT-G embedding dim
        action_dim: int = 7,       # End-effector delta (xyz + rot + gripper)
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        # Project action to latent dim
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Transformer layers
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            decoder_layer,
            num_layers=n_layers,
        )

        # Output projection
        self.out_proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, z_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        """
        Predict next latent state from current state and action.

        Args:
            z_t: Current latent state. Shape (B, D) for pooled embeddings.
            a_t: Action. Shape (B, action_dim).

        Returns:
            z_next: Predicted next latent state. Shape (B, D).
        """
        # Project action to token
        a_token = self.action_proj(a_t)  # (B, D)

        if z_t.ndim == 2:
            # Pooled embeddings: (B, D) -> (B, 2, D) with [action_token, state_token]
            tokens = torch.stack([a_token, z_t], dim=1)  # (B, 2, D)
            out = self.transformer(tokens)                # (B, 2, D)
            z_next = self.out_proj(out[:, 1, :])          # (B, D) - take state position
        elif z_t.ndim == 3:
            # Patch-level: (B, N, D) -> (B, N+1, D)
            a_token = a_token.unsqueeze(1)                # (B, 1, D)
            tokens = torch.cat([a_token, z_t], dim=1)     # (B, N+1, D)
            out = self.transformer(tokens)                # (B, N+1, D)
            z_next = self.out_proj(out[:, 1:, :])         # (B, N, D)
        else:
            raise ValueError(f"Expected z_t.ndim in [2, 3], got {z_t.ndim}")

        return z_next


def load_dynamics_predictor(
    checkpoint_path: str,
    latent_dim: int = 1408,
    action_dim: int = 7,
    device: str = "cuda",
) -> LatentDynamicsPredictor:
    """Load a trained dynamics predictor from checkpoint."""
    model = LatentDynamicsPredictor(
        latent_dim=latent_dim,
        action_dim=action_dim,
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval().to(device)
    return model
