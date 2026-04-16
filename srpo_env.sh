#!/bin/bash
# ===== 统一路径配置 =====

# siiRL 项目路径
export SIIRL_DIR=/DATA/disk1/yjb/projects/VLA/siiRL

# V-JEPA 2 代码路径
export VJEPA2_DIR=/DATA/disk1/yjb/projects/VLA/siiRL/vjepa2

# PYTHONPATH
export PYTHONPATH="$SIIRL_DIR:/root/LIBERO/:$VJEPA2_DIR:$PYTHONPATH"

# 模型路径
export MODEL_PATH=/DATA/disk1/yjb/projects/VLA/siiRL/models/OpenVLA-AC-PD-1traj-libero-long
export VJEPA_MODEL_PATH=/DATA/disk1/yjb/projects/VLA/siiRL/models/vjepa2/vitg-384.pt

# 数据路径（siiRL 会自动生成 parquet 文件）
export TRAIN_DATA_PATH=/DATA/disk1/yjb/projects/VLA/siiRL/data/libero_10/train.parquet
export TEST_DATA_PATH=/DATA/disk1/yjb/projects/VLA/siiRL/data/libero_10/test.parquet

# 输出路径
export BASE_CKPT_PATH=/DATA/disk1/yjb/projects/VLA/siiRL/output/ckpts
export BASE_TENSORBOARD_PATH=/DATA/disk1/yjb/projects/VLA/siiRL/output/tensorboard

# 渲染
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# GPU 数量（你有 8 张 A100）
export N_GPUS_PER_NODE=8

echo "Environment configured: $N_GPUS_PER_NODE GPUs, MODEL=$MODEL_PATH"
