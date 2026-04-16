# LWM-Reward 实现文档

## 概述

LWM-Reward 以**非侵入式并行 reward 路径**的方式实现在 siiRL 的 SRPO 基础设施之上。SRPO 的所有代码保持不变。通过一个配置项 `reward_model.reward_manager=lwm` 即可在 SRPO 和 LWM-Reward 之间切换。

## 架构对比

```
SRPO（原有，不动）：
  Rollout → V-JEPA 编码整条轨迹 → DBSCAN 聚类 → 轨迹级标量 reward → 放在终端步

LWM-Reward（新增）：
  Rollout → V-JEPA 累积帧编码每步 → dynamics 预测(可选) → PBRS reward → 每步分配 reward → grpo_step 归一化
```

## 文件变更汇总

### 新建文件（7 个）

| 文件 | 说明 |
|------|------|
| `siirl/utils/embodied/dynamics_predictor.py` | Action-conditioned 潜空间 dynamics 模型（轻量 transformer，~30M 参数） |
| `siirl/utils/reward_score/lwm_reward.py` | LWM-Reward 计算核心：从 clip embeddings + dynamics 输出 step-level PBRS reward |
| `siirl/engine/reward_manager/lwm.py` | Step-level reward 分配管理器（把 reward 放在每步，而非只放终端） |
| `scripts/train_dynamics.py` | Dynamics predictor 训练脚本（L1 loss，基于预计算的 V-JEPA embedding） |
| `scripts/precompute_goal_embeddings.py` | 预计算 expert demo 终态 goal embedding（消除对 batch 成功轨迹的依赖） |
| `examples/lwm_reward_trainer/run_lwm_libero_long.sh` | LIBERO-Long 训练启动脚本 |
| `docs/LWM_REWARD_IMPLEMENTATION.md` | 本文档 |

### 修改文件（7 个，最小改动）

| 文件 | 改动量 | 改了什么 |
|------|-------|---------|
| `siirl/params/embodied_args.py` | +15 行 | `EmbodiedArguments` 中新增 5 个 LWM 配置字段（`lwm_clip_window` 已移除） |
| `siirl/engine/reward_manager/__init__.py` | +3 行 | 注册 `LWMRewardManager` 的 lazy import 和 `__all__` 导出 |
| `siirl/execution/scheduler/reward.py` | +7 行 | 添加 `LWMRewardManager` import、`"lwm"` 路由、lazy load `compute_lwm_reward` |
| `siirl/engine/rollout/embodied_rollout.py` | +80 行 | 累积帧编码 per-step embedding、actions、三级优先 goal embedding |
| `siirl/dag_worker/core_algos.py` | +65 行 | 新增 `compute_grpo_step_advantage()` step-level GRPO + dispatcher 分支 |
| `siirl/execution/scheduler/enums.py` | +1 行 | 新增 `GRPO_STEP = "grpo_step"` 枚举值 |
| `siirl/execution/scheduler/launch.py` | +1 行 | `RayTrainer` 的 critic 判断列表中加入 `GRPO_STEP`（不需要 critic） |
| `siirl/main_dag.py` | ~5 行 | `load_pipeline()` 中允许 `grpo_step` 进入 embodied pipeline |

---

## 各文件详细说明

### 1. `siirl/params/embodied_args.py`（修改）

在 `EmbodiedArguments` dataclass 中新增：

```python
lwm_reward_enabled: bool = False           # 总开关
lwm_dynamics_model_path: Optional[str]     # dynamics checkpoint 路径（为 None 则不用 dynamics）
lwm_goal_demo_dir: Optional[str]           # 预计算 goal embedding 目录（含 .npy 文件）
lwm_gamma: float = 0.99                    # PBRS 折扣因子（避免 semantic trap）
lwm_success_threshold: float = 0.05        # progress >= 1 - threshold 判定为成功
```

### 2. `siirl/execution/scheduler/reward.py`（修改）

三处改动：

1. **Import**：在 `from siirl.engine.reward_manager import ...` 中添加 `LWMRewardManager`
2. **路由表**：`manager_map` 中添加 `"lwm": LWMRewardManager`，`default_compute_score_map` 中添加 `"lwm": None`
3. **Lazy load**：在 embodied 的 lazy import 之后，添加 lwm 的 lazy import：
   ```python
   if compute_score_fn is None and reward_manager_name == "lwm":
       from siirl.utils.reward_score.lwm_reward import compute_lwm_reward
       compute_score_fn = compute_lwm_reward
   ```

### 3. `siirl/engine/rollout/embodied_rollout.py`（修改）

在现有 V-JEPA 轨迹级 embedding 计算（`batch["vjepa_embedding"]` 赋值）之后，新增条件块：

```python
if getattr(self.config.embodied, 'lwm_reward_enabled', False):
    # 1. 用 cumulative sliding window 编码每步：取 frames[0:end_idx]（从头到当前步的所有帧）
    # 2. 用 self.embedding_model（已加载的 V-JEPA 2）批量编码所有 clip
    # 3. 从 vla_history 提取每步 action（mean across chunk sub-steps）
    # 4. Goal embedding 三级优先：预计算 .npy > batch 成功轨迹 > 轨迹级 fallback
    # 5. 存入 batch["lwm_step_embeddings"], batch["lwm_step_actions"], batch["lwm_goal_embeddings"]
```

**关键设计**：
- **Cumulative window 编码**：每步取从头到当前步的所有帧（非 fixed window），后期步帧数充足无 padding
- 复用 `self.embedding_model`（SRPO 已加载的 V-JEPA 2 实例），不额外加载模型
- 批量编码 clip（每批 16 个，避免 OOM）
- **Goal embedding 三级优先**：(1) `lwm_goal_demo_dir` 下的预计算 `.npy` 文件 → (2) batch 内成功轨迹终态 → (3) 退化为轨迹级 embedding
- 整个块被 `if lwm_reward_enabled` 包裹，SRPO 运行时完全不执行

### 4. `siirl/utils/reward_score/lwm_reward.py`（新建）

核心 reward 计算逻辑：

```
对每条轨迹 i 的每步 t：
    z_t = step_embeddings[i, t]              # V-JEPA clip embedding
    
    如果有 dynamics predictor：
        z_next = P_phi(z_t, a_t)             # Action-causal 预测
    否则：
        z_next = step_embeddings[i, t+1]     # 退化：用真实下一步状态
    
    Phi(t) = 1 - d(z_next, z_g) / d(z_0, z_g)     # Progress
    r_t = r_gold + gamma * Phi(t) - Phi(t-1)        # PBRS reward
```

返回格式：每条轨迹一个 dict，包含 `step_rewards`（list）和 `progress_curve`（list），兼容 `score`（轨迹级标量，用于 logging）。

### 5. `siirl/engine/reward_manager/lwm.py`（新建）

与 `EmbodiedRewardManager` 的核心区别：

```python
# EmbodiedRewardManager（SRPO）：只在终端步放 reward
verifier_rewards[i, last_step_idx] = score

# LWMRewardManager：每步放 reward
for t, r_t in enumerate(step_rewards):
    token_idx = (t + 1) * action_token_len - 1
    verifier_rewards[i, token_idx] = r_t
```

`verify()` 方法与 SRPO 一致（用 binary success），保证验证逻辑不变。

### 6. `siirl/utils/embodied/dynamics_predictor.py`（新建）

轻量 transformer 模型：
- 输入：`z_t`（V-JEPA embedding，dim=1408）+ `a_t`（动作，dim=7）
- 架构：action 通过 MLP 投射为 token，与 state token 拼接，送入 4 层 transformer encoder
- 输出：`z_{t+1}` 预测 embedding
- 支持 pooled（B, D）和 patch-level（B, N, D）两种输入模式
- `load_dynamics_predictor()` 辅助函数加载 checkpoint

### 7. `scripts/train_dynamics.py`（新建）

Dynamics predictor 训练脚本：
- 数据格式：`.npz` 文件，每个包含 `embeddings`（T, D）和 `actions`（T-1, action_dim）
- 训练目标：L1 loss `||P_phi(z_t, a_t) - z_{t+1}||_1`
- 优化器：AdamW + cosine schedule
- 保存最优和最终 checkpoint

---

## 如何切换 SRPO 和 LWM-Reward

### 跑 SRPO（原有，不变）
```bash
reward_model.reward_manager=embodied
algorithm.adv_estimator=grpo
# 不需要任何 lwm 相关配置，lwm_reward_enabled 默认为 False
```

### 跑 LWM-Reward（推荐配置）
```bash
reward_model.reward_manager=lwm
algorithm.adv_estimator=grpo_step                    # 关键：step-level advantage
actor_rollout_ref.embodied.lwm_reward_enabled=True
actor_rollout_ref.embodied.lwm_gamma=0.99
actor_rollout_ref.embodied.lwm_goal_demo_dir=/path/to/goal_embeddings  # 预计算 goal
```

### 跑 LWM-Reward（带 dynamics predictor，完整版）
```bash
reward_model.reward_manager=lwm
algorithm.adv_estimator=grpo_step
actor_rollout_ref.embodied.lwm_reward_enabled=True
actor_rollout_ref.embodied.lwm_gamma=0.99
actor_rollout_ref.embodied.lwm_goal_demo_dir=/path/to/goal_embeddings
actor_rollout_ref.embodied.lwm_dynamics_model_path=/path/to/dynamics_best.pt
```

**重要**：必须用 `grpo_step` 而非 `grpo`。用 `grpo` 时 step rewards 会被求和为轨迹标量，与 SRPO 数学等价，step-level 信息丢失。

---

## Dynamics Predictor 训练流程

### 第一步：预计算 V-JEPA 2 Embedding

对 DROID/OXE 数据集中的每条轨迹，用 V-JEPA 2 编码每帧周围的 clip，保存为 `.npz`：

```python
# embeddings: (T, 1408) float32 — 每帧的 V-JEPA 2 clip embedding
# actions: (T-1, 7) float32 — 帧间的动作
np.savez("traj_0001.npz", embeddings=embs, actions=acts)
```

### 第二步：训练

```bash
python scripts/train_dynamics.py \
    --data_dir /path/to/precomputed_embeddings \
    --output_dir checkpoints/dynamics \
    --latent_dim 1408 \
    --action_dim 7 \
    --epochs 50 \
    --batch_size 256 \
    --lr 3e-4
```

预计耗时：DROID 23k 轨迹，8×A100 约 4 小时。

---

## 数据流图

```
                    ┌─────────────────────────────────────────────┐
                    │            Rollout (embodied_rollout.py)     │
                    │                                             │
                    │  VLA 生成动作 → 环境执行 → 收集帧和结果        │
                    │  all_video[task] 积累所有帧                   │
                    │  vla_history 存储每步的 action                │
                    │                                             │
                    │  ┌─ SRPO 路径（始终执行）────────────────────┐ │
                    │  │ V-JEPA 编码整条轨迹                       │ │
                    │  │ → batch["vjepa_embedding"]               │ │
                    │  └──────────────────────────────────────────┘ │
                    │                                             │
                    │  ┌─ LWM 路径（仅 lwm_enabled 时执行）──────┐ │
                    │  │ V-JEPA cumulative window 编码每步         │ │
                    │  │ → batch["lwm_step_embeddings"]           │ │
                    │  │ 提取每步 action                           │ │
                    │  │ → batch["lwm_step_actions"]              │ │
                    │  │ Goal: 预计算 > batch成功 > fallback       │ │
                    │  │ → batch["lwm_goal_embeddings"]           │ │
                    │  └──────────────────────────────────────────┘ │
                    └──────────────────┬──────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────────┐
                    │          Reward 路由 (reward.py)             │
                    │                                             │
                    │  reward_manager == "embodied":               │
                    │    → EmbodiedRewardManager（SRPO）           │
                    │    → DBSCAN 聚类 → 轨迹级 reward             │
                    │    → 只在终端步放 reward                      │
                    │                                             │
                    │  reward_manager == "lwm":                    │
                    │    → LWMRewardManager                        │
                    │    → PBRS step-level reward                  │
                    │    → 每步放 reward                            │
                    └──────────────────┬──────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────────┐
                    │          Advantage 估计                       │
                    │                                             │
                    │  adv_estimator == "grpo":                    │
                    │    → sum(step_rewards) → 轨迹标量 → 归一化    │
                    │    → 与 SRPO 等价，step 信息丢失              │
                    │                                             │
                    │  adv_estimator == "grpo_step":（推荐）        │
                    │    → 每步 reward 独立在 group 内归一化         │
                    │    → 每步有独立 advantage，step-level 有效    │
                    └─────────────────────────────────────────────┘
```

---

## Reward 公式

唯一一个公式：

```
r_t = r_gold + gamma * Phi(t) - Phi(t-1)
```

其中：
- `Phi(t) = 1 - d(P_phi(z_t, a_t), z_g) / d(z_0, z_g)`：progress function
- `r_gold`：sparse outcome reward（完成=1，否则=0）
- `gamma`：折扣因子（默认 0.99），保证 policy invariance（Ng 1999 PBRS）
- `P_phi`：dynamics predictor（如果未提供，退化为用真实 z_{t+1}）
- `z_g`：goal embedding（expert demo 终态 clip 编码）
- `d`：L2 距离

**不加 gamma 会怎样**：Robo-Dopamine 论文（Appendix A.2）证明，直接用 `Phi(t) - Phi(t-1)` 作为 reward 会导致 semantic trap——策略学会停在高 progress 状态不动，而不是完成任务。

---

## TODO

- [x] 预计算 expert demo 的 goal embedding 并缓存 → `scripts/precompute_goal_embeddings.py`
- [x] 添加 step-level advantage 估计 → `grpo_step` in `core_algos.py`
- [x] 修复 clip 编码 padding 问题 → cumulative sliding window in `embodied_rollout.py`
- [ ] Cross-VLA 评测脚本（用同一个 LWM-Reward 评测 OpenVLA / π₀ / Octo）
- [ ] RoboRewardBench 评测集成
- [ ] DROID/OXE 数据的 V-JEPA embedding 预计算脚本
