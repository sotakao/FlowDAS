#!/bin/bash
#
# Submit with: sbatch submit_generate.sbatch
#

#SBATCH --job-name=sqg_inference        # Job name
#SBATCH --time=06:00:00              # Walltime
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=1                   # Number of tasks
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --mem=4G                     # Memory per node
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
python inference2.py --data_dir /central/scratch/sotakao/sqg_train_data \
                     --train_file sqg_pv_train.h5 \
                     --hrly_freq 3 \
                     --obs_pct 0.25 \
                     --obs_fn exp_scaled \
                     --obs_sigma 1.0 \
                     --guidance_strength 1.0 \
                     --guidance_method DPS \
                     --n_ens 20 \
                     --log_wandb 1 \
                     --true_initial 1 \
                     --mc_samples 1 \
                     --em_steps 200
