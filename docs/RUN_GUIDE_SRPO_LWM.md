# SRPO & LWM-Reward 完整运行指南

> 适用环境：8 × A100-80GB, Docker 可用
>
> 本文档严格对齐官方文档 `docs/start/install.rst`、`docs/examples/embodied_srpo_example.rst` 和 `examples/embodied_srpo_trainer/` 里的脚本

---

## 阶段 1：环境准备

官方文档 (`docs/examples/embodied_srpo_example.rst`) 指定的 Docker 镜像：

```bash
docker pull siiai/siirl-vla:libero-egl-cu12.6
```

启动容器：

```bash
docker run -it --gpus all \
    --shm-size=64g \
    --network=host \
    -v /DATA/disk1/yjb/projects/VLA/siiRL:/DATA/disk1/yjb/projects/VLA/siiRL \
    -e MUJOCO_GL=egl \
    -e PYOPENGL_PLATFORM=egl \
    --name siirl_train \
    siiai/siirl-vla:libero-egl-cu12.6 \
    /bin/bash
```

容器内安装 siiRL：

```bash
cd /DATA/disk1/yjb/projects/VLA/siiRL
git config --global --add safe.directory /DATA/disk1/yjb/projects/VLA/siiRL
pip config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple/
SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 pip install -e .
pip install numpy==1.26.4
pip install transformers==4.51.0
# flash_attn 预编译版本和 torch 2.10+cu128 不匹配，需从源码编译（15-20 分钟）
pip uninstall flash-attn -y
pip install flash-attn --no-build-isolation --no-cache-dir
```

> LIBERO 已预装在 Docker 镜像 `/root/LIBERO/` 中，不需要手动安装。

---

## 阶段 2：准备模型

对齐官方文档 `docs/examples/embodied_srpo_example.rst` Step 2。

### 2.1 V-JEPA 2 代码库

项目里已经有 `vjepa2/` 目录。如果没有：

```bash
cd /DATA/disk1/yjb/projects/VLA/siiRL
git clone https://github.com/facebookresearch/vjepa2.git
```

### 2.2 下载 V-JEPA 2 权重

```bash
huggingface-cli download Sylvest/vjepa2-vit-g \
    --local-dir /DATA/disk1/yjb/projects/VLA/siiRL/models/vjepa2
```

### 2.3 下载 SFT 模型

根据你要跑的任务选择对应模型。官方提供 4 个（对应 LIBERO 四个 suite）：

```bash
# libero_long（论文称 LIBERO-Long，API 名 libero_10）
huggingface-cli download Sylvest/OpenVLA-AC-PD-1traj-libero-long \
    --local-dir /DATA/disk1/yjb/projects/VLA/siiRL/models/OpenVLA-AC-PD-1traj-libero-long

# 其他 suite（按需）：
# huggingface-cli download Sylvest/OpenVLA-AC-PD-1traj-libero-object --local-dir /DATA/disk1/yjb/projects/VLA/siiRL/models/OpenVLA-AC-PD-1traj-libero-object
# huggingface-cli download Sylvest/OpenVLA-AC-PD-1traj-libero-spatial --local-dir /DATA/disk1/yjb/projects/VLA/siiRL/models/OpenVLA-AC-PD-1traj-libero-spatial
# huggingface-cli download Sylvest/OpenVLA-AC-PD-1traj-libero-goal --local-dir /DATA/disk1/yjb/projects/VLA/siiRL/models/OpenVLA-AC-PD-1traj-libero-goal
```

### 2.4 设置环境变量

对齐官方文档 Step 1：

```bash
export SIIRL_DIR=/DATA/disk1/yjb/projects/VLA/siiRL
export VJEPA2_DIR=/DATA/disk1/yjb/projects/VLA/siiRL/vjepa2
export PYTHONPATH="$SIIRL_DIR:/root/LIBERO/:$VJEPA2_DIR:$PYTHONPATH"
```

---

## 阶段 3：运行 SRPO 训练

**官方提供了现成的训练脚本**，在 `examples/embodied_srpo_trainer/` 目录下：

```
examples/embodied_srpo_trainer/
├── run_openvla_oft_libero_long.sh
├── run_openvla_oft_libero_goal.sh
├── run_openvla_oft_libero_object.sh
└── run_openvla_oft_libero_spatial.sh
```

### 3.1 修改脚本中的路径

对齐官方文档 Step 3 的说明，需要修改脚本开头的以下路径：

```bash
cd /DATA/disk1/yjb/projects/VLA/siiRL
```

以 `run_openvla_oft_libero_long.sh` 为例，修改这几个变量：

| 变量 | 改为 |
|------|------|
| `SIIRL_DIR` | `/DATA/disk1/yjb/projects/VLA/siiRL` |
| PYTHONPATH 中的 vjepa2 路径 | `/DATA/disk1/yjb/projects/VLA/siiRL/vjepa2` |
| `HOME_PATH` | `/DATA/disk1/yjb/projects/VLA/siiRL` |
| `MODEL_PATH` | `/DATA/disk1/yjb/projects/VLA/siiRL/models/OpenVLA-AC-PD-1traj-libero-long` |
| `VJEPA_MODEL_PATH` | `/DATA/disk1/yjb/projects/VLA/siiRL/models/vjepa2/vitg-384.pt` |

具体改法（修改脚本前几行）：

```bash
sed -i 's|SIIRL_DIR="${SIIRL_DIR:your_siirl_path}"|SIIRL_DIR="/DATA/disk1/yjb/projects/VLA/siiRL"|' examples/embodied_srpo_trainer/run_openvla_oft_libero_long.sh
sed -i 's|your_vjepa2_path|/DATA/disk1/yjb/projects/VLA/siiRL/vjepa2|' examples/embodied_srpo_trainer/run_openvla_oft_libero_long.sh
sed -i 's|HOME_PATH=${HOME_PATH:your_home_path}|HOME_PATH=/DATA/disk1/yjb/projects/VLA/siiRL|' examples/embodied_srpo_trainer/run_openvla_oft_libero_long.sh
```

### 3.2 启动训练

```bash
bash examples/embodied_srpo_trainer/run_openvla_oft_libero_long.sh
```

> siiRL 会自动生成 LIBERO 的 task manifest（parquet 文件），不需要手动准备数据集。

### 3.3 监控训练

对齐官方文档 Step 4：

```bash
# TensorBoard
tensorboard --logdir ./tensorboard --bind_all --port 6006

# 关键指标：
# - val/all: 验证成功率（核心指标）
# - reward/verifier_mean: 平均 reward
```

---

## 阶段 4：运行 LWM-Reward 训练

LWM-Reward 在 SRPO 基础上有 4 处配置差异。提供了现成脚本：

```
examples/lwm_reward_trainer/
└── run_lwm_libero_long.sh
```

### 4.1 修改脚本中的路径

和 SRPO 一样，改路径变量：

```bash
sed -i 's|SIIRL_DIR="${SIIRL_DIR:your_siirl_path}"|SIIRL_DIR="/DATA/disk1/yjb/projects/VLA/siiRL"|' examples/lwm_reward_trainer/run_lwm_libero_long.sh
sed -i 's|your_vjepa2_path|/DATA/disk1/yjb/projects/VLA/siiRL/vjepa2|' examples/lwm_reward_trainer/run_lwm_libero_long.sh
sed -i 's|HOME_PATH=${HOME_PATH:your_home_path}|HOME_PATH=/DATA/disk1/yjb/projects/VLA/siiRL|' examples/lwm_reward_trainer/run_lwm_libero_long.sh
```

### 4.2 预计算 goal embeddings（推荐，一次性）

消除对 batch 内成功轨迹的依赖：

```bash
PYTHONPATH=/DATA/disk1/yjb/projects/VLA/siiRL:/root/LIBERO/:/DATA/disk1/yjb/projects/VLA/siiRL/vjepa2:$PYTHONPATH \
python scripts/precompute_goal_embeddings.py \
    --suite libero_10 \
    --vjepa_model_path /DATA/disk1/yjb/projects/VLA/siiRL/models/vjepa2/vitg-384.pt \
    --output_dir /DATA/disk1/yjb/projects/VLA/siiRL/data/goal_embeddings/libero_10
```

然后在 LWM 脚本中设置 `LWM_GOAL_DEMO_DIR`（见 4.1 后的脚本修改）。

### 4.3 模式 A：不带 dynamics predictor

直接跑，`DYNAMICS_MODEL_PATH` 留空即可：

```bash
bash examples/lwm_reward_trainer/run_lwm_libero_long.sh
```

### 4.4 模式 B：带 dynamics predictor

先训练 dynamics predictor：

```bash
# 需要预计算好的 V-JEPA 2 embedding（.npz 格式）
python3 scripts/train_dynamics.py \
    --data_dir /path/to/precomputed_embeddings \
    --output_dir /DATA/disk1/yjb/projects/VLA/siiRL/output/dynamics_ckpt \
    --latent_dim 1408 \
    --action_dim 7 \
    --epochs 50 \
    --batch_size 256 \
    --lr 3e-4
```

然后修改 LWM 脚本中的 `DYNAMICS_MODEL_PATH`：

```bash
# 在 run_lwm_libero_long.sh 中将：
export DYNAMICS_MODEL_PATH=""
# 改为：
export DYNAMICS_MODEL_PATH="/DATA/disk1/yjb/projects/VLA/siiRL/output/dynamics_ckpt/dynamics_best.pt"
```

再启动训练：

```bash
bash examples/lwm_reward_trainer/run_lwm_libero_long.sh
```

### 4.5 SRPO vs LWM 的配置差异（4 处）

| # | SRPO 脚本 | LWM 脚本 | 说明 |
|---|-----------|----------|------|
| 1 | `reward_model.reward_manager=embodied` | `reward_model.reward_manager=lwm` | 切换 reward manager |
| 2 | `algorithm.adv_estimator=grpo` | `algorithm.adv_estimator=grpo_step` | step-level advantage（关键） |
| 3 | 无 | `actor_rollout_ref.embodied.lwm_reward_enabled=True` | 开启 step-level clip 编码 |
| 4 | 无 | `actor_rollout_ref.embodied.lwm_gamma=0.99` | PBRS 折扣因子 |

**注意**：必须用 `grpo_step` 而非 `grpo`。用 `grpo` 时 step rewards 被求和为轨迹标量，与 SRPO 数学等价。

---

## 附录：LIBERO benchmark 名称映射

siiRL 脚本中的 `DATASET` / `env_name` 和 LIBERO API 名、论文名的对应关系：

| 脚本中 env_name | LIBERO API 名 | 论文名 |
|----------------|---------------|--------|
| `libero_10` | `libero_10` | LIBERO-Long |
| `libero_spatial` | `libero_spatial` | LIBERO-Spatial |
| `libero_object` | `libero_object` | LIBERO-Object |
| `libero_goal` | `libero_goal` | LIBERO-Goal |
