#!/bin/bash
#SBATCH --job-name=matrix
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=224
#SBATCH --partition=wm
#SBATCH --exclude=g42-h100-instance-[134,135,065,084,085,030,160,049]

# export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
# export TORCHDYNAMO_VERBOSE=1
# export WANDB_MODE="offline"
# export NCCL_P2P_DISABLE=1
# export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=true
# export NCCL_TIMEOUT=1800
# export TORCH_NCCL_BLOCKING_WAIT=1


# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_P2P_DISABLE=1


# if [ -z "$MASTER_ADDR" ]; then
#     export MASTER_ADDR=localhost
# fi
# if [ -z "$MASTER_PORT" ]; then
#     export MASTER_PORT=8000
# fi
# if [ -z "$RANK" ]; then
#     export RANK=0
# fi
# if [ -z "$WORLD_SIZE" ]; then
#     export WORLD_SIZE=1
# fi

eval "$(conda shell.bash hook)"
conda activate yichi_diffusers32

export WANDB_ENTITY='guangyil'
export WANDB_PROJECT='matrix'
export WANDB_API_KEY='ff910f5032281881b854480213e7cf10b5e87f00'

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=8000

echo MASTER_ADDR=$MASTER_ADDR

export GPU_IDS="0,1,2,3,4,5,6,7"
export ACCELERATE_CONFIG_FILE="accelerate_configs/deepspeed.yaml"

export PYTHONUNBUFFERED='TRUE'
export OMP_NUM_THREADS=28

srun sh -c 'accelerate launch --config_file $ACCELERATE_CONFIG_FILE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $SLURM_NODEID \
    --num_machines $SLURM_NNODES \
    --num_processes $((SLURM_NNODES * 8)) \
    --gpu_ids $GPU_IDS \
    training/cogvideox_text_to_video_sft.py --config configs/sft_config_from_stage1_t5_actions.py'