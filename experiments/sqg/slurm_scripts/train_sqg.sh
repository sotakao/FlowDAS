#!/bin/bash
#
# Submit with: sbatch submit_generate.sbatch
#

#SBATCH --job-name=train_sqg        # Job name
#SBATCH --time=12:00:00              # Walltime
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=1                   # Number of tasks
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --mem=16G                     # Memory per node
#SBATCH --mail-user=sotakao@caltech.edu
#SBATCH --mail-type=BEGIN,END,FAIL   # Notify at start/end/fail

# --- Choose partition / GPUs as needed ---
#SBATCH --partition=gpu              # Use GPU partition
#SBATCH --gres=gpu:1                 # Request 1 GPU
# If you want CPU only, comment the two lines above and uncomment this:
# #SBATCH --partition=cpu

# --- Optional: log files ---
#SBATCH -o slurm/%x-%j.out           # STDOUT
#SBATCH -e slurm/%x-%j.err           # STDERR

# --- Environment setup ---
source ~/.bashrc
cd /resnick/groups/astuart/sotakao/score-based-ensemble-filter/FlowDAS/experiments/sqg
conda activate flowdas

# --- Run program ---
python train.py \
  --train_data_path sqg_pv_train.h5 \
  --val_data_path sqg_pv_valid.h5 \
  --hrly_freq 3 \
  --window 7 \
  --batch_size 32 \
  --sigma_coef 1.0 \
  --C 2 --H 64 --W 64 \
  --epochs 4096 \
  --lr 2e-4 \
  --ckpt_path ./checkpoints/latest.pt \
  --use_wandb 1 \
  --scheduler cosine \
  --normalize 1 \
  --label_noise_std 0.5 \
