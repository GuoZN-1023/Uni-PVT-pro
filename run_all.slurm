#!/usr/bin/env bash
#SBATCH -J trial_gzn              # 任务名，自行修改，让自己能认得就行，建议任务简介+名字缩写
#SBATCH -p 3090                   # 分区
#SBATCH -N 1                      # 单节点
#SBATCH --gres=gpu:3090:1         # 申请 4 张 3090（想要 1/8 就改这里） # 申请之前先检查节点情况，问问GPT怎么分配比较好
#SBATCH -c 32                     # CPU 核（经验：每卡 6~10 核比较舒服）
#SBATCH --mem=120G                # 内存
#SBATCH -t 24:00:00               # 虽然 infinite，但建议先给个上限方便管理
#SBATCH -o /home/thermo2025/Uni-PVT_results/results_gzn/results_gzn_20260122/logs/%x_%j.out #log日志，记得替换成自己的results路径
#SBATCH -e /home/thermo2025/Uni-PVT_results/results_gzn/results_gzn_20260122/logs/%x_%j.err #log日志，记得替换成自己的results路径

set -euo pipefail
umask 077
mkdir -p logs

echo "JobID=$SLURM_JOB_ID  Host=$(hostname)  Time=$(date)"
echo "WorkDir=$SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR"

# ---- GPU/CPU 信息（确认资源到位）----
nvidia-smi || true
echo "CPUs=$SLURM_CPUS_PER_TASK  GPUs_on_node=${SLURM_GPUS_ON_NODE:-unknown}"

if [ ! -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  echo "ERROR: conda.sh not found at $HOME/miniconda3/etc/profile.d/conda.sh"
  echo "You need to install miniconda under \$HOME/miniconda3 or adjust this path."
  exit 1
fi
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate Uni-PVT

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1

MASTER_PORT=$((10000 + RANDOM % 50000))
GPUS="${SLURM_GPUS_ON_NODE:-4}"

echo "Using MASTER_PORT=$MASTER_PORT  nproc_per_node=$GPUS"

export NCCL_DEBUG=warn
export NCCL_IB_DISABLE=1

# -----------------------------
# 重要改动（保留注释，回到单卡单进程）
# 你现在申请的是 1 张 GPU（--gres=gpu:3090:1），因此不应 torchrun 启 4 进程
# 最稳的做法是：限制只看见 1 张卡，并用 python 直接跑 run_all.py
# -----------------------------
export CUDA_VISIBLE_DEVICES=0

# torchrun --standalone \
#   --nproc_per_node="$GPUS" \
#   --master_port="$MASTER_PORT" \
#   run_all.py --config configs/config_total.yaml

python3.10 run_all.py --config configs/config_total_H.yaml