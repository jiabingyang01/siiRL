#!/usr/bin/env bash
set -e

# ===== 第一步：加载路径配置 =====
source /DATA/disk1/yjb/projects/VLA/siiRL/srpo_env.sh

# ===== 第二步：实验参数 =====
export DATASET=libero_10          # libero_10=Long, 可选: libero_spatial / libero_object / libero_goal
export ALG=srpo

# 训练超参数（与论文一致）
export TRAIN_BATCH_SIZE=64
export ROLLOUT_N_SAMPLES=8
export PPO_MINI_BATCH_SIZE=4
export PPO_EPOCHS=1
export LEARNING_RATE=5e-6
export TEMPERATURE=1.6
export CLIP_RATIO_HIGH=0.28
export CLIP_RATIO_LOW=0.2

# 环境参数
export NUM_ENVS=16                # 每 GPU 16 个并行环境
export MAX_EPISODE_STEPS=512      # 最大步数
export VAL_BATCH_SIZE=496         # 验证 batch size

# 输出
timestamp=$(date +%Y%m%d_%H%M%S)
export CKPT_PATH=${BASE_CKPT_PATH}/srpo_${DATASET}_${timestamp}

# ===== 第三步：创建数据目录 =====
mkdir -p $(dirname $TRAIN_DATA_PATH)
mkdir -p $CKPT_PATH

# ===== 第四步：启动 Ray =====
ray stop --force 2>/dev/null || true
ray start --head \
    --num-gpus $N_GPUS_PER_NODE \
    --object-store-memory 50000000000 \
    --memory 80000000000

echo "Ray started with $N_GPUS_PER_NODE GPUs"

# ===== 第五步：启动训练 =====
python3 -m siirl.main_dag \
    data.train_files=$TRAIN_DATA_PATH \
    data.val_files=$TEST_DATA_PATH \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    data.dataset_type=embodied \
    \
    reward_model.reward_manager=embodied \
    reward_model.reward_kwargs.action_token_len=7 \
    reward_model.reward_kwargs.reward_coef=5.0 \
    \
    algorithm.workflow_type=embodied \
    algorithm.adv_estimator=grpo \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.filter_groups.enable=True \
    algorithm.embodied_sampling.filter_accuracy=True \
    algorithm.embodied_sampling.accuracy_lower_bound=0.1 \
    algorithm.embodied_sampling.accuracy_upper_bound=0.9 \
    algorithm.embodied_sampling.filter_truncated=False \
    algorithm.embodied_sampling.oversample_factor=1 \
    \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.model_type=embodied \
    actor_rollout_ref.model.trust_remote_code=True \
    \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.actor.optim.weight_decay=0.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_epochs=$PPO_EPOCHS \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.clip_ratio_c=10000.0 \
    actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH \
    actor_rollout_ref.actor.clip_ratio_low=$CLIP_RATIO_LOW \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.shuffle=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.n=$ROLLOUT_N_SAMPLES \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.prompt_length=256 \
    actor_rollout_ref.rollout.response_length=512 \
    \
    actor_rollout_ref.embodied.embodied_type=openvla-oft \
    actor_rollout_ref.embodied.action_token_len=7 \
    actor_rollout_ref.embodied.action_chunks_len=8 \
    actor_rollout_ref.embodied.video_embedding_model_path=$VJEPA_MODEL_PATH \
    actor_rollout_ref.embodied.embedding_img_size=384 \
    actor_rollout_ref.embodied.embedding_enable_fp16=True \
    actor_rollout_ref.embodied.embedding_model_offload=False \
    actor_rollout_ref.embodied.center_crop=True \
    actor_rollout_ref.embodied.num_images_in_input=1 \
    actor_rollout_ref.embodied.unnorm_key=$DATASET \
    \
    actor_rollout_ref.embodied.env.env_type=libero \
    actor_rollout_ref.embodied.env.env_name=$DATASET \
    actor_rollout_ref.embodied.env.num_envs=$NUM_ENVS \
    actor_rollout_ref.embodied.env.max_steps=$MAX_EPISODE_STEPS \
    actor_rollout_ref.embodied.env.num_steps_wait=10 \
    actor_rollout_ref.embodied.env.num_trials_per_task=50 \
    actor_rollout_ref.embodied.env.model_family=openvla \
    \
    critic.use_critic_model=False \
    \
    trainer.total_epochs=1000 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.max_actor_ckpt_to_keep=5 \
    "trainer.logger=['console','tensorboard']" \
    trainer.project_name=siirl_srpo_${DATASET} \
    trainer.experiment_name=srpo_${DATASET} \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.default_local_dir=$CKPT_PATH \
    trainer.resume_mode=auto \
    trainer.val_before_train=True

echo "SRPO training complete!"
