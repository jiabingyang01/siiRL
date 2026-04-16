#!/usr/bin/env bash
# ===================================================================================
# ===    LWM-Reward Training with OpenVLA-OFT on LIBERO-LONG                     ===
# ===    Based on SRPO trainer with key differences:                               ===
# ===      - reward_model.reward_manager=lwm (step-level dense reward)             ===
# ===      - actor_rollout_ref.embodied.lwm_reward_enabled=True                    ===
# ===      - Optional dynamics predictor for action-causal reward                  ===
# ===================================================================================

set -e

# --- Environment Setup ---
export SIIRL_DIR="/DATA/disk1/yjb/projects/VLA/siiRL"
export PYTHONPATH="$SIIRL_DIR:/root/LIBERO/:/DATA/disk1/yjb/projects/VLA/siiRL/vjepa2:$PYTHONPATH"

# --- Experiment and Model Definition ---
export DATASET=libero_10
export ALG=lwm_reward
export MODEL_NAME=openvla-oft-7b
export MODEL_TYPE=openvla-oft

# --- Path Definitions (USER PROVIDED) ---
export HOME_PATH=/DATA/disk1/yjb/projects/VLA/siiRL
export TRAIN_DATA_PATH=$HOME_PATH/datasets/vla-oft/libero/$DATASET/train.parquet
export TEST_DATA_PATH=$HOME_PATH/datasets/vla-oft/libero/$DATASET/test.parquet
export MODEL_PATH=$HOME_PATH/models/OpenVLA-AC-PD-1traj-libero-long
export VJEPA_MODEL_PATH=$HOME_PATH/models/vjepa2/vitg-384.pt

# --- LWM-Reward Specific ---
# Set to path of trained dynamics predictor checkpoint, or leave empty to use state-only progress
export DYNAMICS_MODEL_PATH=""  # e.g., "$HOME_PATH/checkpoints/dynamics/dynamics_best.pt"
# Set to directory with pre-computed goal embeddings (.npy files from precompute_goal_embeddings.py)
# Leave empty to fall back to batch-internal success trajectories
export LWM_GOAL_DEMO_DIR=""  # e.g., "$HOME_PATH/data/goal_embeddings/libero_10"
export LWM_GAMMA=0.99
export LWM_SUCCESS_THRESHOLD=0.05

# Base output paths
export BASE_CKPT_PATH=ckpts
export BASE_TENSORBOARD_PATH=tensorboard

# --- Embodied AI Specific Parameters ---
export ACTION_TOKEN_LEN=7
export ACTION_CHUNKS_LEN=8
export NUM_ENVS=16
export MAX_EPISODE_STEPS=512

# --- Data and Sampling Parameters ---
export VAL_BATCH_SIZE=496
export MAX_PROMPT_LENGTH=256
export MAX_RESPONSE_LENGTH=128

# --- Embodied Sampling Parameters ---
export FILTER_ACCURACY=True
export ACCURACY_LOWER_BOUND=0.1
export ACCURACY_UPPER_BOUND=0.9
export FILTER_TRUNCATED=False
export OVERSAMPLE_FACTOR=1

# --- Training Hyperparameters ---
export TRAIN_BATCH_SIZE=64
export PPO_MINI_BATCH_SIZE=4
export ROLLOUT_N_SAMPLES=8
export PPO_EPOCHS=1

# Algorithm parameters
export LEARNING_RATE=5e-6
export WEIGHT_DECAY=0.0
export CLIP_RATIO_HIGH=0.28
export CLIP_RATIO_LOW=0.2
export ENTROPY_COEFF=0.0
export TEMPERATURE=1.6
export GAMMA=1.0    # RL discount (separate from LWM_GAMMA for PBRS)
export LAM=1.0
export GRAD_CLIP=1.0

# --- Image/Video Processing ---
export IMG_SIZE=384
export ENABLE_FP16=True
export EMBEDDING_MODEL_OFFLOAD=False
export CENTER_CROP=True
export NUM_IMAGES_IN_INPUT=1
export NUM_STEPS_WAIT=10

# --- Trainer Configuration ---
export SAVE_FREQ=5
export TEST_FREQ=5
export TOTAL_EPOCHS=1000
export MAX_CKPT_KEEP=5
export VAL_BEFORE_TRAIN=True

# --- Multi-node distributed training ---
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
export NNODES=${PET_NNODES:-1}
export NODE_RANK=${PET_NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29500}

# --- Environment Variables ---
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export GLOO_SOCKET_TIMEOUT=600

# --- Output Paths ---
timestamp=$(date +%Y%m%d_%H%M%S)
export CKPT_PATH=${BASE_CKPT_PATH}/${MODEL_NAME}_${ALG}_${DATASET}_${NNODES}nodes
export PROJECT_NAME=siirl_lwm_${DATASET}
export EXPERIMENT_NAME=openvla_oft_lwm_reward
export TENSORBOARD_DIR=${BASE_TENSORBOARD_PATH}/${MODEL_NAME}_${ALG}_${DATASET}/${timestamp}
export SIIRL_LOGGING_FILENAME=${MODEL_NAME}_${ALG}_${DATASET}_${timestamp}

# --- Build optional args ---
DYNAMICS_ARG=""
if [ -n "$DYNAMICS_MODEL_PATH" ] && [ -f "$DYNAMICS_MODEL_PATH" ]; then
    DYNAMICS_ARG="actor_rollout_ref.embodied.lwm_dynamics_model_path=$DYNAMICS_MODEL_PATH"
fi
GOAL_DEMO_ARG=""
if [ -n "$LWM_GOAL_DEMO_DIR" ] && [ -d "$LWM_GOAL_DEMO_DIR" ]; then
    GOAL_DEMO_ARG="actor_rollout_ref.embodied.lwm_goal_demo_dir=$LWM_GOAL_DEMO_DIR"
fi

# --- Define the Training Command ---
# Key differences from SRPO marked with [LWM]
TRAINING_CMD=(
    python3 -m siirl.main_dag
    # Data configuration
    data.train_files=\$TRAIN_DATA_PATH
    data.val_files=\$TEST_DATA_PATH
    data.train_batch_size=\$TRAIN_BATCH_SIZE
    data.val_batch_size=\$VAL_BATCH_SIZE
    data.max_prompt_length=\$MAX_PROMPT_LENGTH
    data.max_response_length=\$MAX_RESPONSE_LENGTH
    data.dataset_type=embodied

    # [LWM] Reward: use lwm reward manager instead of embodied
    reward_model.reward_manager=lwm
    reward_model.reward_kwargs.action_token_len=7
    reward_model.reward_kwargs.reward_coef=5.0

    # Algorithm configuration
    # [LWM] Use step-level GRPO for per-step credit assignment
    algorithm.workflow_type=embodied
    algorithm.adv_estimator=grpo_step
    algorithm.gamma=\$GAMMA
    algorithm.lam=\$LAM
    algorithm.norm_adv_by_std_in_grpo=True

    # Embodied sampling
    algorithm.filter_groups.enable=True
    algorithm.embodied_sampling.filter_accuracy=\$FILTER_ACCURACY
    algorithm.embodied_sampling.accuracy_lower_bound=\$ACCURACY_LOWER_BOUND
    algorithm.embodied_sampling.accuracy_upper_bound=\$ACCURACY_UPPER_BOUND
    algorithm.embodied_sampling.filter_truncated=\$FILTER_TRUNCATED
    algorithm.embodied_sampling.oversample_factor=\$OVERSAMPLE_FACTOR

    # Model configuration
    actor_rollout_ref.model.path=\$MODEL_PATH
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.model_type=embodied
    actor_rollout_ref.model.trust_remote_code=True
    # Actor configuration
    actor_rollout_ref.actor.optim.lr=\$LEARNING_RATE
    actor_rollout_ref.actor.optim.weight_decay=\$WEIGHT_DECAY
    actor_rollout_ref.actor.ppo_mini_batch_size=\$PPO_MINI_BATCH_SIZE
    actor_rollout_ref.actor.ppo_epochs=\$PPO_EPOCHS
    actor_rollout_ref.actor.grad_clip=\$GRAD_CLIP
    actor_rollout_ref.actor.clip_ratio_c=10000.0
    actor_rollout_ref.actor.clip_ratio_high=\$CLIP_RATIO_HIGH
    actor_rollout_ref.actor.clip_ratio_low=\$CLIP_RATIO_LOW
    actor_rollout_ref.actor.entropy_coeff=\$ENTROPY_COEFF
    actor_rollout_ref.actor.shuffle=True

    # FSDP
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.grad_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False

    # Rollout
    actor_rollout_ref.rollout.name=hf
    actor_rollout_ref.rollout.n=\$ROLLOUT_N_SAMPLES
    actor_rollout_ref.rollout.temperature=\$TEMPERATURE
    actor_rollout_ref.rollout.do_sample=True
    actor_rollout_ref.rollout.prompt_length=\$MAX_PROMPT_LENGTH
    actor_rollout_ref.rollout.response_length=512

    # Embodied AI configuration
    actor_rollout_ref.embodied.embodied_type=\$MODEL_TYPE
    actor_rollout_ref.embodied.action_token_len=\$ACTION_TOKEN_LEN
    actor_rollout_ref.embodied.action_chunks_len=\$ACTION_CHUNKS_LEN
    actor_rollout_ref.embodied.video_embedding_model_path=\$VJEPA_MODEL_PATH
    actor_rollout_ref.embodied.embedding_img_size=\$IMG_SIZE
    actor_rollout_ref.embodied.embedding_enable_fp16=\$ENABLE_FP16
    actor_rollout_ref.embodied.embedding_model_offload=\$EMBEDDING_MODEL_OFFLOAD
    actor_rollout_ref.embodied.center_crop=\$CENTER_CROP
    actor_rollout_ref.embodied.num_images_in_input=\$NUM_IMAGES_IN_INPUT
    actor_rollout_ref.embodied.unnorm_key=\$DATASET

    # [LWM] Enable LWM-Reward step-level dense reward
    actor_rollout_ref.embodied.lwm_reward_enabled=True
    actor_rollout_ref.embodied.lwm_gamma=\$LWM_GAMMA
    actor_rollout_ref.embodied.lwm_success_threshold=\$LWM_SUCCESS_THRESHOLD
    $DYNAMICS_ARG
    $GOAL_DEMO_ARG

    # Environment
    actor_rollout_ref.embodied.env.env_type=libero
    actor_rollout_ref.embodied.env.env_name=\$DATASET
    actor_rollout_ref.embodied.env.num_envs=\$NUM_ENVS
    actor_rollout_ref.embodied.env.max_steps=\$MAX_EPISODE_STEPS
    actor_rollout_ref.embodied.env.num_steps_wait=\$NUM_STEPS_WAIT
    actor_rollout_ref.embodied.env.num_trials_per_task=50
    actor_rollout_ref.embodied.env.model_family=openvla

    # Critic (not used)
    critic.use_critic_model=False

    # Trainer
    trainer.total_epochs=\$TOTAL_EPOCHS
    trainer.save_freq=\$SAVE_FREQ
    trainer.test_freq=\$TEST_FREQ
    trainer.max_actor_ckpt_to_keep=\$MAX_CKPT_KEEP
    trainer.logger=['console','tensorboard']
    trainer.project_name=\$PROJECT_NAME
    trainer.experiment_name=\$EXPERIMENT_NAME
    trainer.nnodes=\$NNODES
    trainer.n_gpus_per_node=\$N_GPUS_PER_NODE
    trainer.default_local_dir=\$CKPT_PATH
    trainer.resume_mode=auto
    trainer.val_before_train=\$VAL_BEFORE_TRAIN
)

# ===================================================================================
# ===                          EXECUTION LOGIC                                    ===
# ===================================================================================
# (Same as SRPO - Ray cluster setup and main execution)

set -e
set -o pipefail
set -x

start_ray_cluster() {
    local RAY_HEAD_WAIT_TIMEOUT=600
    export RAY_RAYLET_NODE_MANAGER_CONFIG_NIC_NAME=${INTERFACE_NAME}
    export RAY_GCS_SERVER_CONFIG_NIC_NAME=${INTERFACE_NAME}
    export RAY_RUNTIME_ENV_AGENT_CREATION_TIMEOUT_S=1200
    export RAY_GCS_RPC_CLIENT_CONNECT_TIMEOUT_S=120

    local ray_start_common_opts=(
        --num-gpus "$N_GPUS_PER_NODE"
        --object-store-memory 60000000000
        --memory 60000000000
    )

    if [ "$NNODES" -gt 1 ]; then
        if [ "$NODE_RANK" = "0" ]; then
            export RAY_ADDRESS="$RAY_MASTER_ADDR:$RAY_MASTER_PORT"
            ray start --head --port="$RAY_MASTER_PORT" --dashboard-port="$RAY_DASHBOARD_PORT" "${ray_start_common_opts[@]}" --system-config='{"gcs_server_request_timeout_seconds": 60, "gcs_rpc_server_reconnect_timeout_s": 60}'
            local start_time=$(date +%s)
            while ! ray health-check --address "$RAY_ADDRESS" &>/dev/null; do
                if [ "$(( $(date +%s) - start_time ))" -ge "$RAY_HEAD_WAIT_TIMEOUT" ]; then exit 1; fi
                sleep 5
            done
        else
            local head_node_address="$MASTER_ADDR:$RAY_MASTER_PORT"
            local start_time=$(date +%s)
            while ! ray health-check --address "$head_node_address" &>/dev/null; do
                if [ "$(( $(date +%s) - start_time ))" -ge "$RAY_HEAD_WAIT_TIMEOUT" ]; then exit 1; fi
                sleep 5
            done
            ray start --address="$head_node_address" "${ray_start_common_opts[@]}"
        fi
    else
        ray start --head "${ray_start_common_opts[@]}"
    fi
}

main() {
    ray stop --force
    export VLLM_USE_V1=1
    export GLOO_SOCKET_TIMEOUT=600
    export GLOO_TCP_TIMEOUT=600
    export RAY_MASTER_PORT=${RAY_MASTER_PORT:-6379}
    export RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
    export RAY_MASTER_ADDR=$MASTER_ADDR

    start_ray_cluster

    if [ "$NNODES" -gt 1 ] && [ "$NODE_RANK" = "0" ]; then
        local TIMEOUT=600; local start_time=$(date +%s)
        while true; do
            if [ "$(( $(date +%s) - start_time ))" -ge "$TIMEOUT" ]; then exit 1; fi
            local ready_nodes=$(ray list nodes --format=json | python3 -c "import sys, json; print(len(json.load(sys.stdin)))")
            if [ "$ready_nodes" -ge "$NNODES" ]; then break; fi
            sleep 5
        done
    fi

    if [ "$NODE_RANK" = "0" ]; then
        eval "${TRAINING_CMD[@]}" "$@"
        sleep 30; ray stop --force >/dev/null 2>&1
    elif [ "$NNODES" -gt 1 ]; then
        local head_node_address="$MASTER_ADDR:$RAY_MASTER_PORT"
        while ray health-check --address "$head_node_address" &>/dev/null; do sleep 15; done
    fi
}

main "$@"
