#!/usr/bin/env python3
"""
Pre-compute goal embeddings from expert demonstration terminal clips.

For each LIBERO task, loads the expert demo, extracts the terminal frames,
encodes them with V-JEPA 2, and saves the goal embedding as .npy file.

These pre-computed goal embeddings are used by LWM-Reward so it doesn't
depend on batch-internal success trajectories (unlike SRPO).

Usage:
  python scripts/precompute_goal_embeddings.py \
    --suite libero_10 \
    --vjepa_model_path /path/to/vjepa2/vitg-384.pt \
    --output_dir /path/to/goal_embeddings \
    --img_size 384 \
    --num_terminal_frames 64

Output structure:
  {output_dir}/
    libero_10_task_0.npy   # (embed_dim,) float32
    libero_10_task_1.npy
    ...
    task_names.txt          # mapping: filename -> task language description
"""

import argparse
import os
import sys

import numpy as np


def main(args):
    # Setup environment for headless rendering
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"

    from loguru import logger

    # Import LIBERO
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    # Import V-JEPA embedding model
    from siirl.utils.embodied.video_emb import VideoEmbeddingModel

    logger.info(f"Loading V-JEPA 2 model from {args.vjepa_model_path}")
    emb_model = VideoEmbeddingModel(
        model_path=args.vjepa_model_path,
        img_size=args.img_size,
        device_id=args.gpu_id,
        enable_fp16=True,
    )
    logger.info(f"Embedding dim: {emb_model.embedding_dim}")

    # Load LIBERO suite
    Suite = benchmark.get_benchmark(args.suite)
    suite = Suite()
    n_tasks = suite.get_num_tasks()
    logger.info(f"Suite '{args.suite}' has {n_tasks} tasks")

    os.makedirs(args.output_dir, exist_ok=True)
    task_names_file = os.path.join(args.output_dir, "task_names.txt")

    with open(task_names_file, "w") as f_names:
        for task_idx in range(n_tasks):
            task = suite.get_task(task_idx)
            task_name = f"{args.suite}_task_{task_idx}"
            task_desc = task.language
            logger.info(f"Processing task {task_idx}/{n_tasks}: {task_desc}")

            # Get demo init states and bddl file
            init_states = suite.get_task_init_states(task_idx)
            bddl_file = suite.get_task_bddl_file_path(task_idx)

            # Create environment
            env_args = {
                "bddl_file_name": bddl_file,
                "camera_heights": 256,
                "camera_widths": 256,
            }
            env = OffScreenRenderEnv(**env_args)
            env.seed(0)

            # Reset to first demo init state and collect terminal frames
            # by stepping with zero actions until the task is "done" or max steps
            env.reset()
            if len(init_states) > 0:
                env.set_init_state(init_states[0])
            obs = env.reset()

            # Collect frames by running the demo trajectory
            demo = suite.get_task_demonstration(task_idx)
            frames = []
            if demo is not None and len(demo) > 0:
                # Use the first demo trajectory
                traj = demo[0]
                actions = traj.get("actions", None)
                if actions is not None:
                    env.reset()
                    if len(init_states) > 0:
                        env.set_init_state(init_states[0])
                    obs = env.reset()
                    frames.append(obs["agentview_image"])
                    for act in actions:
                        obs, reward, done, info = env.step(act)
                        frames.append(obs["agentview_image"])
                        if done:
                            break

            if len(frames) < 2:
                # Fallback: just use the initial observation repeated
                logger.warning(f"Task {task_idx}: only {len(frames)} frames, using init frame")
                frames = [obs["agentview_image"]] * args.num_terminal_frames

            env.close()

            # Take terminal frames (last N frames of the successful demo)
            n_term = min(args.num_terminal_frames, len(frames))
            terminal_frames = frames[-n_term:]

            # Encode with V-JEPA 2
            embedding = emb_model.get_embeddings(
                [task_name], [terminal_frames]
            )[0]

            # Save
            save_path = os.path.join(args.output_dir, f"{task_name}.npy")
            np.save(save_path, embedding)
            f_names.write(f"{task_name}\t{task_desc}\n")
            logger.info(f"  Saved: {save_path}, shape={embedding.shape}, "
                        f"norm={np.linalg.norm(embedding):.4f}")

    logger.info(f"Done! Goal embeddings saved to {args.output_dir}")
    logger.info(f"Task names: {task_names_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-compute goal embeddings for LWM-Reward")
    parser.add_argument("--suite", type=str, default="libero_10",
                        help="LIBERO suite name (libero_10, libero_spatial, etc.)")
    parser.add_argument("--vjepa_model_path", type=str, required=True,
                        help="Path to V-JEPA 2 weights (vitg-384.pt)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for goal embeddings")
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_terminal_frames", type=int, default=64,
                        help="Number of terminal frames to use for goal encoding")
    args = parser.parse_args()
    main(args)
