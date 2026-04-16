#!/usr/bin/env python3
"""
Train the Latent Dynamics Predictor for LWM-Reward.

This script trains a lightweight transformer that predicts z_{t+1} = P_phi(z_t, a_t)
in V-JEPA 2's latent space. Training data comes from DROID or OXE trajectories.

Pre-requisites:
  1. Pre-compute V-JEPA 2 embeddings for all training trajectory frames
  2. Extract actions from the trajectory dataset

Usage:
  python scripts/train_dynamics.py \
    --data_dir /path/to/precomputed_embeddings \
    --output_dir /path/to/save_checkpoint \
    --latent_dim 1408 \
    --action_dim 7 \
    --epochs 50 \
    --batch_size 256 \
    --lr 3e-4

Data format:
  data_dir should contain .npz files, each with:
    - "embeddings": (T, D) float32 - V-JEPA 2 embeddings per frame
    - "actions": (T-1, action_dim) float32 - actions between frames
  OR a single .npz with "all_embeddings" and "all_actions" pre-concatenated.
"""

import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from loguru import logger


class DynamicsPairDataset(Dataset):
    """Dataset of (z_t, a_t, z_{t+1}) tuples for dynamics training."""

    def __init__(self, data_dir: str):
        self.z_t_list = []
        self.a_t_list = []
        self.z_next_list = []

        npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")

        for f in npz_files:
            data = np.load(f)
            embs = data["embeddings"]   # (T, D)
            acts = data["actions"]       # (T-1, action_dim)
            n = min(len(embs) - 1, len(acts))
            if n <= 0:
                continue
            self.z_t_list.append(embs[:n])
            self.a_t_list.append(acts[:n])
            self.z_next_list.append(embs[1:n + 1])

        self.z_t = np.concatenate(self.z_t_list, axis=0)
        self.a_t = np.concatenate(self.a_t_list, axis=0)
        self.z_next = np.concatenate(self.z_next_list, axis=0)
        logger.info(f"Loaded {len(self.z_t)} transition pairs from {len(npz_files)} files")

    def __len__(self):
        return len(self.z_t)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.z_t[idx], dtype=torch.float32),
            torch.tensor(self.a_t[idx], dtype=torch.float32),
            torch.tensor(self.z_next[idx], dtype=torch.float32),
        )


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Import dynamics predictor
    from siirl.utils.embodied.dynamics_predictor import LatentDynamicsPredictor

    model = LatentDynamicsPredictor(
        latent_dim=args.latent_dim,
        action_dim=args.action_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Dynamics predictor: {param_count / 1e6:.1f}M parameters")

    dataset = DynamicsPairDataset(args.data_dir)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(dataloader),
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for z_t, a_t, z_next in dataloader:
            z_t, a_t, z_next = z_t.to(device), a_t.to(device), z_next.to(device)

            z_pred = model(z_t, a_t)
            loss = nn.functional.l1_loss(z_pred, z_next)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        elapsed = time.time() - t0
        logger.info(f"Epoch {epoch + 1}/{args.epochs} | L1 loss: {avg_loss:.6f} | "
                    f"lr: {scheduler.get_last_lr()[0]:.2e} | time: {elapsed:.1f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(args.output_dir, "dynamics_best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "loss": avg_loss,
                "config": {
                    "latent_dim": args.latent_dim,
                    "action_dim": args.action_dim,
                    "n_layers": args.n_layers,
                    "n_heads": args.n_heads,
                },
            }, save_path)
            logger.info(f"  Saved best checkpoint: {save_path} (loss={avg_loss:.6f})")

    # Save final checkpoint
    save_path = os.path.join(args.output_dir, "dynamics_final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs,
        "loss": avg_loss,
        "config": {
            "latent_dim": args.latent_dim,
            "action_dim": args.action_dim,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
        },
    }, save_path)
    logger.info(f"Training complete. Final checkpoint: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LWM-Reward Dynamics Predictor")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory with precomputed V-JEPA embeddings and actions (.npz)")
    parser.add_argument("--output_dir", type=str, default="checkpoints/dynamics",
                        help="Output directory for checkpoints")
    parser.add_argument("--latent_dim", type=int, default=1408, help="V-JEPA 2 embedding dim")
    parser.add_argument("--action_dim", type=int, default=7, help="Action dimension")
    parser.add_argument("--n_layers", type=int, default=4, help="Transformer layers")
    parser.add_argument("--n_heads", type=int, default=8, help="Attention heads")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    args = parser.parse_args()
    train(args)
