# LWM-Reward: Step-Level Action-Causal Dense Reward via Latent World Model for Cross-VLA Policy Learning and Evaluation

> **核心主张**：SRPO 证明了 V-JEPA 2 潜空间是机器人 progress estimation 的最优基底（SC=0.998），但只以 trajectory-level 单标量的方式使用了它——整条轨迹一个 reward，所有 step 共享同一个 advantage，且绑在 GRPO loop 里不能跨 VLA 通用。我们将 V-JEPA 2 的利用方式从 trajectory-level 提升为 step-level，引入 action-conditioned dynamics predictor 实现 action-causal reward，并设计为可独立于 VLA 训练 loop 使用的通用 reward function，首次实现跨多种 VLA 架构的统一奖励评测与 RL 训练。

> **副标题**："From One Score Per Trajectory to One Score Per Step"

---

## 1. 问题背景与动机

### 1.1 核心矛盾

| 方法 | Reward 粒度 | 需要 batch 内成功轨迹 | 跨 VLA 可用 | Action-conditioned | 推理开销 |
|------|-----------|-------------------|-----------|-------------------|---------|
| GRPO (binary) | Trajectory (0/1) | ❌ | ❌ (需 log-prob) | ❌ | ~0 |
| SRPO | Trajectory (连续标量) | ✅ 必须 | ❌ (绑 GRPO loop) | ❌ | ~0 |
| RoboReward 8B | Episode (1-5 discrete) | ❌ | ✅ | ❌ | ~100ms |
| Robo-Dopamine GRM | Step (continuous) | ❌ | ✅ | ❌ | ~100ms |
| πRL (PPO+GAE) | Step (Critic) | ❌ | ❌ (需 Critic) | ❌ | ~2x 训练 |
| **LWM-Reward** | **Step (continuous)** | **❌** | **✅** | **✅** | **~2ms** |

**关键观察**：

1. **SRPO 的表征质量已被验证**：V-JEPA 2 latent space 做 progress estimation 碾压 pixel-level（SC 0.998 vs 0.125）和 ImageBind（0.957），见 SRPO Table 3。

2. **但 SRPO 只给整条轨迹打一个分**：V-JEPA 2 编码整条轨迹 $o_{0:T}$ → 一个 embedding $h_i$ → 算到成功 cluster center 的 L2 距离 → 一条轨迹一个标量 $g_i$。所有 step 共享同一个 advantage $\hat{A}_i = (g_i - \mu_g) / \sigma_g$。Step 间的 progress 差异信息被完全丢弃。

3. **Step-level 信号在 long-horizon 任务上至关重要**：πRL 实验表明 PPO（step-level GAE）在 LIBERO-Long 上 90.2% vs GRPO（trajectory-level）81.4%，差距 9 个点。SRPO 在 Long 上 98.6% 是因为 reward 质量远好于 binary，但如果配上 step-level advantage，Long 上可能更高。

4. **SRPO 绑在 GRPO loop 里**：(a) 需要 batch 内有成功轨迹做 DBSCAN 聚类；(b) 低成功率时 batch 内可能无成功轨迹，reward 退化；(c) 不能脱离训练 loop 独立评测其他 VLA。

5. **VLM-based reward（RoboReward, Robo-Dopamine）推理太重**：RoboReward 8B 和 GRM 8B 都基于 VLM，推理 ~100ms/step。且 RoboRewardBench 显示即使 GPT-5 也存在大量 false positive/negative（Figure 7 的 Gemini Robotics-ER 失败案例），VLM 的空间推理能力不足以可靠地评估精细操作。

**矛盾总结**：

> V-JEPA 2 潜空间做 progress estimation 效果极好（SRPO 已证明），但 SRPO 只以 trajectory-level 使用了它，且绑在 GRPO loop 里。VLM-based reward model 虽然是 step-level 且可独立使用，但推理重、空间推理不可靠。**没有人用 V-JEPA 2 latent 做 step-level 的、action-conditioned 的、可独立使用的轻量 reward function。**

### 1.2 结构性机会

V-JEPA 2 有两个被 SRPO 忽视的能力：

**能力 1：Clip-level encoding**。V-JEPA 2 是视频模型，输入本来就是多帧 clip。SRPO 的 Appendix B 中做 reward quality 评测时，实际上就是用 cumulative sliding window（逐帧扩展的 clip）编码的，证明了 clip-level embedding 能有效反映 progress。但 SRPO 在 RL 中没有用这个——它在 RL 中用的是 full-trajectory encoding。

**能力 2：Action-conditioned dynamics prediction**。V-JEPA 2-AC 证明了在 DROID 23k 轨迹上训练 action-conditioned latent predictor 后，可以在从未见过的实验室做零样本 planning（reach 100%, pick-and-place 65%）。这个 dynamics predictor 能预测"执行动作 $a_t$ 后潜态会怎么变"。

将这两个能力组合起来，就是 LWM-Reward：用 clip-level encoding 获得 step-level 潜表征，用 dynamics predictor 预测动作效果，用预测结果与目标的距离变化作为 reward。

---

## 2. 预备知识

### 2.1 V-JEPA 2

- **编码器** $W$：输入视频 clip → pooled embedding $h \in \mathbb{R}^D$。在 100万+ 小时互联网视频上自监督预训练
- **Action-conditioned predictor** $P_\phi$：$\hat{z}_{t+1} = P_\phi(z_t, a_t)$。在冻结 encoder 表征空间中做单步状态转移预测

### 2.2 SRPO 的 Reward 机制（精确描述）

1. 编码：$h_i = W(o_{0:T}^{(i)})$，**整条轨迹** → **一个 embedding**
2. 聚类：$C = \text{DBSCAN}(\{h_j : \text{trajectory } j \text{ is success}\})$
3. 距离：$d_i = \min(\{\|h_i - h_c\|_2 : h_c \in C\})$
4. Reward：成功 $g_i = 1.0$，失败 $g_i = \phi(\frac{d_i - \bar{d}}{\sigma_d})$
5. Advantage：$\hat{A}_i = (g_i - \mu_g) / \sigma_g$，**整条轨迹所有 step 共享**

注意：SRPO 在 real-world offline RL（Appendix G, Equation 15）中实际上用了 step-level 的 progress difference $D_{i,t} = R_{i,t} - R_{i,t-1}$ 做 advantage。但在 online RL 模式下退回了 trajectory-level。

### 2.3 Policy-Invariant Reward Shaping（Robo-Dopamine）

直接用 progress 变化量 $\Phi(s_{t+1}) - \Phi(s_t)$ 作为 reward 会导致 **semantic trap**（Robo-Dopamine Appendix A.2 证明）：策略学会到达高 progress 状态后停住不动，而不是完成任务。

正确做法（Ng 1999 PBRS + Robo-Dopamine Equation 14）：

$$r_t = r_{gold} + \gamma \Phi(s_{t+1}) - \Phi(s_t)$$

加 $\gamma$ 折扣和 sparse outcome reward $r_{gold}$（完成任务=1，否则=0），保证 policy invariance。

### 2.4 RoboRewardBench

RoboReward 建立的标准化评测：2,831 条 human-verified episode，14 种 embodiment，22 个 VLM 排名。使用 1-5 discrete progress score，MAE 作为主指标。RoboReward 8B MAE=0.665（Rank 1），GPT-5 mini MAE=0.691（Rank 2）。

RoboReward Figure 3 证明：reward 准确度与 downstream RL 成功率强相关（$r=0.83$）。

---

## 3. 方法

### 3.1 概览

LWM-Reward 由三个模块组成，训练完成后作为**独立的外部 reward function**，可插入任意 VLA 的 RL loop：

```
Module 1: Step-Level Latent Encoder    V-JEPA 2 编码当前 clip → z_t（frozen，零训练）
Module 2: Action-Conditioned Dynamics  P_ϕ(z_t, a_t) → ẑ_{t+1}（训练 ~4h）
Module 3: Goal Reference              V-JEPA 2 编码 expert demo 终态 clip → z_g（一次性计算）

Progress:  Φ(t) = 1 - d(ẑ_{t+1}, z_g) / d(z_0, z_g)
Reward:    r_t = r_gold + γ · d(z_t, z_g) - d(P_ϕ(z_t, a_t), z_g)
```

### 3.2 Module 1: Step-Level Latent Encoding

对每个环境步 $t$，用 **cumulative sliding window** 编码从初始帧到当前步的所有帧：

$$z_t = W(o_{0:t}), \quad z_t \in \mathbb{R}^D$$

V-JEPA 2 完全冻结，零额外训练。输入帧数根据步数自然增长：早期步帧数少（V-JEPA 2 内部通过循环 padding 补齐到 64 帧），后期步帧数充足（通过 linspace 均匀下采样到 64 帧，无 padding）。

**与 SRPO 的区别**：SRPO 编码整条轨迹 $W(o_{0:T})$ → 一个 embedding。LWM-Reward 编码每步的累积帧 $W(o_{0:t})$ → 每步一个 embedding。

**为什么用 cumulative 而非 fixed window**：fixed window（如取最近 8 帧）导致每步只有极少帧，padding 到 64 帧后 embedding 质量严重降低。Cumulative window 与 SRPO Appendix B 做 reward quality 评测时使用的方式一致，已被验证有效。

### 3.3 Module 2: Action-Conditioned Dynamics Predictor

$$\hat{z}_{t+1} = P_\phi(z_t, a_t)$$

**架构**：轻量 transformer，约 30M 参数。
- 输入：$z_t$（pooled V-JEPA embedding, dim=1408）+ $a_t$（末端执行器 delta, dim=7，通过 MLP 映射为额外 token）
- 4 层 transformer encoder，8 heads
- 输出：$\hat{z}_{t+1}$

**训练**：DROID 23k 轨迹（含成功和失败），L1 latent prediction loss：

$$\mathcal{L}_{dyn} = \| P_\phi(z_t, a_t) - \text{sg}(W(o_{t+1})) \|_1$$

8×A100 约 4 小时。对每帧预计算 V-JEPA 2 encoding 并缓存。

**只做单步预测**：训练目标和使用目标完全对齐，无 compounding error。V-JEPA 2-AC 已验证单步 latent prediction 在 DROID 数据上有效。

**Dynamics predictor 提供的核心价值——Action-Causal Attribution**：

没有 dynamics 时，reward 只看 $z_t$（状态），不看 $a_t$（动作）。同一个 $z_t$ 下两个不同动作给相同 reward。

有 dynamics 后，reward 通过 $P_\phi(z_t, a_t)$ 区分不同动作：好的动作让 $\hat{z}_{t+1}$ 靠近目标，坏的动作让 $\hat{z}_{t+1}$ 远离目标。这是所有现有 robot reward model（SRPO, RoboReward, Robometer, TOPReward, Robo-Dopamine）都不具备的。

### 3.4 Module 3: Goal Reference

利用每个任务预先提供的一条 expert demo 的终态 clip：

$$z_g = W(o_{T-W+1:T}^{\text{expert}})$$

一次性计算，整个 RL 训练过程中固定。

**为什么合理**：SRPO 的 one-shot SFT 设定（每个任务一条 demo）是标准配置。这条 demo 的终态天然可用。不需要 batch 内有成功轨迹（vs SRPO）。

**可选扩展**：训练 language-conditioned goal encoder $g_\psi(l, z_0) \to z_g$，去掉对 demo 终态图像的依赖。但 image-goal 模式已足够支撑全部实验。

### 3.5 Progress 与 Reward 计算

#### 3.5.1 Progress Function

$$\Phi(t) = 1 - \frac{d(P_\phi(z_t, a_t),\ z_g)}{d(z_0,\ z_g)}$$

其中 $d$ 为 L2 距离（与 SRPO 一致）。

成功轨迹上 $\Phi(t)$ 从 ~0 单调涨到 ~1。失败轨迹上 $\Phi(t)$ 中途停滞或下降。直接对标 SRPO Figure 3 的 progress curve 可视化。

#### 3.5.2 Reward Function（Policy-Invariant Reward Shaping）

$$\boxed{r_t = r_{gold} + \gamma \cdot \Phi(t) - \Phi(t-1)}$$

展开：

$$r_t = r_{gold} + \gamma \cdot d(z_t, z_g) - d(P_\phi(z_t, a_t), z_g)$$

（归一化因子 $d(z_0, z_g)$ 在 $\Phi(t)$ 和 $\Phi(t-1)$ 中抵消后，简化为上式。）

其中：
- $r_{gold} = 1$ 若任务完成（$\Phi(t) \geq 1 - \delta$，$\delta=0.05$），否则 $r_{gold} = 0$
- $\gamma$：折扣因子（如 0.99）

**为什么必须加 $\gamma$ 和 $r_{gold}$**：Robo-Dopamine Appendix A.2 证明，不加 $\gamma$ 的 progress 差分 reward 会导致 semantic trap——策略学会停在高 progress 状态不动。加 $\gamma$ 后形成标准 PBRS（Ng 1999），保证不改变最优策略。

#### 3.5.3 Progress Curve 如何出来

对一条轨迹的每步 $t$，计算 $\Phi(t)$。成功轨迹：$\Phi$ 从 0 平滑上升到 1。失败轨迹：$\Phi$ 停滞或下降。Reward $r_t$ 是 progress 的（折扣后的）变化量——正表示这步靠近了目标，负表示远离了。

### 3.6 部署：作为 Reward Function 的使用方式

```python
class LWMReward:
    def __init__(self, encoder, dynamics, z_g, gamma=0.99, delta=0.05):
        self.encoder = encoder      # Frozen V-JEPA 2
        self.dynamics = dynamics    # Trained P_phi
        self.z_g = z_g             # Goal embedding (precomputed)
        self.gamma = gamma
        self.delta = delta
        self.d_init = None         # d(z_0, z_g), set at episode start
        self.phi_prev = 0.0        # Φ(t-1)

    def reset(self, initial_clip):
        z_0 = self.encoder(initial_clip)
        self.d_init = l2(z_0, self.z_g)
        self.phi_prev = 0.0

    @torch.no_grad()
    def __call__(self, obs_clip, action):
        z_t = self.encoder(obs_clip)
        z_next = self.dynamics(z_t, action)

        # Progress
        phi_t = 1.0 - l2(z_next, self.z_g) / (self.d_init + 1e-6)

        # Sparse outcome reward
        r_gold = 1.0 if phi_t >= (1.0 - self.delta) else 0.0

        # Policy-invariant shaped reward
        reward = r_gold + self.gamma * phi_t - self.phi_prev

        self.phi_prev = phi_t
        return reward
```

**推理开销**：每步一次 V-JEPA 2 forward (~2ms) + 一次 dynamics forward (~0.3ms) = **~2.3ms/step**。对比：Robometer-4B ~100ms，RoboReward-8B ~100ms，Robo-Dopamine GRM-8B ~100ms。快约 40 倍。

---

## 4. 理论分析

### 4.1 Policy Invariance

**命题**：LWM-Reward 的 shaped reward 不改变最优策略。

**证明**：$r_t = r_{gold} + \gamma \Phi(s_{t+1}) - \Phi(s_t)$ 是标准 PBRS 形式（Ng 1999），其中 potential function $\Phi(s) = $ LWM-Reward 的 progress function。Robo-Dopamine Appendix A.5 证明了离散时间的 telescoping sum 收敛到 $-\Phi(s_0)$，与策略无关。因此 $\arg\max_a Q^*_{GRM}(s,a) = \arg\max_a Q^*_{gold}(s,a)$。$\square$

### 4.2 Action Discrimination

**命题**：在状态 $z_t$ 固定的条件下，LWM-Reward 对不同动作 $a_t^A, a_t^B$ 的 reward 差异为：

$$r_t(a_t^A) - r_t(a_t^B) = d(\hat{z}_{t+1}^B, z_g) - d(\hat{z}_{t+1}^A, z_g)$$

其中 $\hat{z}_{t+1}^{A/B} = P_\phi(z_t, a_t^{A/B})$。

**含义**：所有现有 robot reward model（SRPO, RoboReward, Robometer, TOPReward, Robo-Dopamine GRM）的 reward 都不依赖 $a_t$——同一状态下不同动作给相同 reward。LWM-Reward 通过 dynamics predictor 区分动作优劣，使 GRPO 的 group-wise comparison 更有效。

### 4.3 方差缩减（vs 不用 dynamics 的 latent potential shaping）

**命题**：在环境转移含噪声的条件下，dynamics 版 reward 的方差严格不大于直接用真实 $z_{t+1}$ 的 reward 方差。

**直觉**：$P_\phi(z_t, a_t)$ 预测的是条件期望（训练数据的平均效果），过滤了传感器噪声、光照变化等非动作因素。Clean 仿真中两者接近，noisy 环境中 dynamics 版方差更低。

### 4.4 Step-Level vs Trajectory-Level 的信息论分析

**命题**：设 $G = \sum_t r_t$（step reward 求和），则 $H(\{r_t\}) \geq H(G)$。SRPO 把 step-level 信息压成 trajectory-level scalar 时，必然丢失 step 间的 credit assignment 信号。

---

## 5. 完整算法

### 5.1 离线准备（一次性）

1. **Goal Encoding**：用 `scripts/precompute_goal_embeddings.py` 编码每个任务的 expert demo 终态 clip → $z_g^{(k)}$，保存为 `.npy` 文件（~0.1 秒/任务）
2. **Dynamics Training**（可选）：DROID 数据上训练 $P_\phi$（8×A100，~4 小时）

### 5.2 在线 RL 中的 Reward 计算

```
For each RL iteration:
    For each parallel environment (task k):
        z_g = load_precomputed_goal(task_k)  ← 预计算，不依赖 batch 成功轨迹
        For each step t:
            a_t = VLA(o_t, l_k)              ← 任意 VLA 架构
            o_{t+1} = env.step(a_t)
            z_t = V-JEPA(o_{0:t})            ← cumulative window 编码
            r_t = PBRS(z_t, a_t, z_g)        ← LWM-Reward

    Advantage estimation（推荐模式B）:
        模式A (GRPO trajectory-level): G_i = Σ r_t^(i), Â_i = (G_i - μ) / σ    ← 与 SRPO 等价
        模式B (GRPO step-level):       Â_t^(i) = (r_t^(i) - μ_r) / σ_r          ← 已实现: grpo_step
        模式C (PPO + GAE):             标准 GAE（需要 Critic，未测试）

    Policy update: 任意 RL 算法
```

**注意**：模式 A 下 step rewards 被求和为轨迹标量，step-level 信息丢失，与 SRPO 数学等价。**必须用模式 B（`grpo_step`）才能发挥 step-level reward 的优势。**

---

## 6. 与现有方法的对比

| 维度 | SRPO | RoboReward 8B | Robo-Dopamine GRM | Robometer 4B | TOPReward | **LWM-Reward** |
|------|------|-------------|-------------------|------------|-----------|---------------|
| 表征空间 | V-JEPA 2 latent | VLM token | VLM token | VLM token | VLM logits | **V-JEPA 2 latent** |
| Reward 粒度 | Trajectory | Episode | Step | Step | Step | **Step** |
| Action-conditioned | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** |
| 需要 batch 内成功轨迹 | ✅ | ❌ | ❌ | ❌ | ❌ | **❌** |
| 需要 VLM 推理 | ❌ | ✅ | ✅ | ✅ | ✅ | **❌** |
| 额外训练数据 | 无 | 45k episodes | 35M samples | 1M+ traj | 无 | **23k traj (DROID)** |
| 额外训练时间 | 无 | 数小时 | 14 天 (128×H100) | 数天 | 无 | **~4h (8×A100)** |
| 推理开销/step | ~0 | ~100ms | ~100ms | ~100ms | ~50ms | **~2.3ms** |
| 跨 VLA 可用 | ❌ | ✅ | ✅ | ✅ | ✅ | **✅** |
| Policy invariance 保证 | 无 | 无 | ✅ (PBRS) | 无 | 无 | **✅ (PBRS)** |

### 6.1 与 SRPO 的精确关系

使用**完全相同的表征**（V-JEPA 2），区别在于使用方式：

| | SRPO | LWM-Reward |
|---|---|---|
| 编码粒度 | 整条轨迹 $W(o_{0:T})$ → 一个 embedding | 每步累积帧 $W(o_{0:t})$ → 每步一个 embedding |
| Reward 粒度 | 一条轨迹一个标量 | 每步一个标量 |
| Reference | Batch 内成功轨迹的 DBSCAN cluster | 预计算的 expert demo 终态（不依赖 batch） |
| Action-conditioned | ❌ | ✅（dynamics predictor，可选） |
| Advantage | Trajectory-level GRPO：所有 step 同权 | Step-level GRPO（`grpo_step`）：每步独立 advantage |
| 是否可独立使用 | ❌（绑定 GRPO loop） | ✅ |
| Reward shaping 理论保证 | 无 | ✅（PBRS + $\gamma$ 折扣） |

---

## 7. 实验设计

### 7.1 Benchmarks

| Benchmark | 用途 | 测试重点 |
|-----------|------|---------|
| SRPO Progress Reward Benchmark | Reward quality | SC, Mono, MMD, JS, SMD（700 success + 300 failure 轨迹） |
| RoboRewardBench | Reward quality（跨 embodiment） | MAE on 1-5 discrete score（2,831 human-verified episodes） |
| LIBERO-Spatial/Object/Goal (各 10 tasks) | RL 性能 | 基础成功率 |
| LIBERO-Long (10 tasks) | RL 性能 | **Step-level credit assignment 的核心战场** |
| LIBERO-Plus | RL 性能 | 泛化鲁棒性（7 维扰动） |

### 7.2 实验 1：Reward Quality

**(a) SRPO Progress Reward Benchmark**：

| Method | SC ↑ | Mono ↑ | MMD ↑ | JS ↑ | SMD ↑ |
|--------|------|--------|-------|------|-------|
| Pixel-level | 0.125 | 0.498 | 0.274 | 0.548 | 2.100 |
| ImageBind | 0.957 | 0.837 | 0.356 | 0.408 | 18.111 |
| SRPO (trajectory-level) | 0.998 | 0.992 | 0.615 | 0.572 | 188.799 |
| **LWM-Reward (step-level)** | **?** | **?** | **?** | **?** | **?** |

**(b) RoboRewardBench**：将 LWM-Reward 的 episode 末尾 progress $\Phi(T)$ 映射到 1-5 discrete score，与 RoboReward 8B（MAE=0.665）、GPT-5 mini（0.691）等 22 个模型做横向对比。

### 7.3 实验 2：LIBERO 主实验（RL 训练）

OpenVLA*（与 SRPO 相同 base model），one-shot SFT warm-start。

| 方法 | Reward | Advantage | 描述 |
|------|--------|-----------|------|
| GRPO (binary) | Binary 0/1 | Trajectory | Baseline |
| SRPO | Trajectory scalar | Trajectory | 复现 |
| **LWM-Reward + traj-adv** | Step-level → 求和 | Trajectory | 控制变量：只换 reward |
| **LWM-Reward + step-adv** | Step-level | Step-level | 完整版 |

**核心对比**：
- LWM-Reward (traj-adv) vs SRPO：隔离"不需要 batch 内成功轨迹 + step-level reward quality"的增量
- LWM-Reward (step-adv) vs LWM-Reward (traj-adv)：隔离 step-level advantage 的增量
- **重点关注 LIBERO-Long**：step-level credit assignment 最关键的任务

### 7.4 实验 3：低成功率场景

LIBERO-Long，one-shot SFT warm-start，初始成功率 ~15%。

每个 RL step 记录：batch 内成功轨迹数量、SRPO 实际可计算 reward 的比例。

预期：训练前 50 步 SRPO 的 batch 内经常无成功轨迹 → 退化为 binary。LWM-Reward 从第 1 步就有 dense signal。

### 7.5 实验 4：Cross-VLA 评测（首次）

**(a) Reward Quality**：收集 OpenVLA-OFT / π₀.5 / Octo 在 LIBERO 10 任务上的 rollout（每策略每任务 50 条）。LWM-Reward / Robometer / RoboReward 分别打分。评测 AUC 和策略排序准确性。

**(b) Downstream RL**：用同一个 LWM-Reward 分别训练 OpenVLA-OFT（autoregressive）和 π₀.5（flow-based）的 RL。对比收益幅度。

**这是前人没做过的实验**——没有人用同一个 reward model 跨多种 VLA 做 downstream RL 对比。

### 7.6 实验 5：LIBERO-Plus 泛化

在 SRPO 的 LIBERO-Plus 设定下（7 维扰动：Camera, Robot-Init, Language, Light, Background, Noise, Layout）评测。

### 7.7 消融实验

| 消融 | 描述 |
|------|------|
| Step-level vs Trajectory-level | 同一 reward，不同 advantage 粒度 |
| 有 dynamics vs 无 dynamics | $P_\phi(z_t, a_t)$ vs 直接用 $z_{t+1}$（clean sim 和 noisy 环境分别测） |
| Encoder 选择 | V-JEPA 2 vs DINOv2 vs SigLIP |
| 编码策略 | Fixed-window ($W=16$) vs cumulative-window ($o_{0:t}$) |
| Goal reference | Expert demo 终态 vs random success 终态 vs language goal |
| $\gamma$ 的影响 | 验证 semantic trap：去掉 $\gamma$ 后策略是否停滞 |

---

## 8. 风险评估

| 风险 | 严重程度 | 缓解 |
|------|---------|------|
| Clip-level V-JEPA 2 的 progress 单调性不够 | 致命 | **Week 1 Go/No-Go**：SC > 0.9 继续。SRPO Appendix B 已用 clip-level 验证过，风险低 |
| LIBERO 天花板低（SRPO 已 99.2%） | 中 | 聚焦 LIBERO-Long、LIBERO-Plus、低成功率场景、cross-VLA |
| Dynamics 在 OOD action 上预测不准 | 中 | DROID 含失败轨迹覆盖范围广；action clipping |
| Dynamics 增量在 clean sim 不显著 | 中（不致命） | Action discrimination 和 noisy 环境是 dynamics 的价值场景 |
| 审稿人说"只是 SRPO 的 incremental" | 中 | Cross-VLA 评测是全新维度；PBRS 理论保证是 SRPO 没有的；action-conditioned 是独立贡献 |

---

## 9. 实现细节

### 9.1 计算开销

| 组件 | 参数量 | 训练开销 | 推理开销/step |
|------|-------|---------|-------------|
| V-JEPA 2 Encoder (frozen) | 307M | 0 | ~2ms |
| Dynamics Predictor | 80M | 4h (8×A100) | ~0.3ms |
| Goal Encoding | — | 0 | 一次性 |
| **总 Reward 推理** | | | **~2.3ms** |

对比：Robometer-4B ~100ms，RoboReward-8B ~100ms，Robo-Dopamine GRM-8B ~100ms。

### 9.2 超参数

| 超参数 | 默认值 |
|--------|--------|
| Encoder | V-JEPA 2 ViT-G (frozen) |
| Clip encoding | Cumulative sliding window $o_{0:t}$ |
| Distance metric | L2 |
| $\gamma$ (reward shaping) | 0.99 |
| $\delta$ (success threshold) | 0.05 |
| Advantage estimator | `grpo_step`（step-level GRPO） |
| Dynamics layers | 4 |
| Dynamics heads | 8 |
| Dynamics latent dim | 1408 |
| Dynamics lr | 3e-4 |
| Dynamics epochs | 50 |

---

## 10. 论文结构

### Title
LWM-Reward: Step-Level Action-Causal Dense Reward via Latent World Model for Vision-Language-Action Policy Learning and Evaluation

### Abstract (~150 words)
V-JEPA 2 的潜空间表征已被证明是机器人 progress estimation 的最优基底（SRPO: SC=0.998）。但 SRPO 只以 trajectory-level 单标量的方式使用了它——整条轨迹一个 reward，所有 step 共享同一个 advantage，且绑在 GRPO loop 里不能跨 VLA 通用。我们提出 LWM-Reward，将 V-JEPA 2 的利用方式从 trajectory-level 提升为 step-level：用 sliding window 编码每步 clip，引入 action-conditioned dynamics predictor 实现 action-causal reward（首个 action-conditioned 的通用 robot reward model），并采用 policy-invariant reward shaping 避免 semantic trap。LWM-Reward 不依赖 batch 内成功轨迹，可作为独立外部 reward function 插入任意 VLA 的 RL loop，推理仅 ~2ms/step（快 VLM-based reward ~40x）。在 SRPO Progress Reward Benchmark、RoboRewardBench、LIBERO 和 LIBERO-Plus 上验证了 step-level reward 相对 trajectory-level 的增量，并首次实现跨多种 VLA 架构（OpenVLA, π₀, Octo）的统一 reward 评测与 RL 训练。

### Sections
1. Introduction：SRPO 的三个保守选择 → step-level action-causal 的机会 → LWM-Reward
2. Related Work：Robot reward models (RoboReward, Robometer, TOPReward, SRPO, Robo-Dopamine, LRM)；Latent world models (V-JEPA 2, DINO-WM, Dreamer)；Reward shaping theory (Ng 1999)
3. Preliminaries：V-JEPA 2, SRPO, PBRS, RoboRewardBench
4. Method：Step-level encoding → Dynamics predictor → Goal reference → Progress & reward → Policy-invariant shaping → Deployment
5. Theoretical Analysis：Policy invariance, action discrimination, 方差缩减, 信息论
6. Experiments：Reward quality (两个 benchmark), LIBERO 主实验, 低成功率, cross-VLA, LIBERO-Plus, 消融
7. Conclusion & Limitations

### 核心 Figures
- **Figure 1** (teaser)：左：SRPO 的 trajectory-level reward（一条轨迹一个标量）；右：LWM-Reward 的 step-level progress curve + action-causal reward
- **Figure 2**：三模块架构图（encoder + dynamics + goal → reward）
- **Figure 3**：Progress curve 对比（LWM-Reward vs SRPO vs pixel-level vs ImageBind，复用 SRPO Figure 3 格式）
- **Figure 4**：LIBERO-Long 训练曲线（LWM-Reward step-adv vs SRPO vs GRPO binary）
- **Figure 5**：低成功率场景学习曲线（前 50 步放大，标注 SRPO 的 batch 内成功轨迹缺失）
- **Figure 6**：Cross-VLA 评测结果

---

## 11. 总结

LWM-Reward 的核心贡献：

> **SRPO 证明了 V-JEPA 2 latent space 是做 robot reward 的最优基底。LWM-Reward 把这个基底的利用方式从"一条轨迹一个分"提升为"每步一个分"，加入 action-conditioned dynamics 实现 action-causal reward，采用 policy-invariant shaping 避免 semantic trap，并解耦为可跨 VLA 通用的独立 reward function。**

方法核心改动简洁（clip-level encoding + dynamics predictor），理论清晰（PBRS 保证不扭曲最优策略），训练轻量（4 小时 vs Robo-Dopamine 的 14 天），推理快速（2ms vs VLM 的 100ms），且打开了 cross-VLA 评测的新维度。
