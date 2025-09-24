
python inference2.py --data_dir /central/scratch/sotakao/sqg_train_data \
                     --train_file sqg_pv_train.h5 \
                     --hrly_freq 3 \
                     --obs_pct 0.25 \
                     --obs_fn linear \
                     --obs_sigma 3.0 \
                     --guidance_strength 0.1 \
                     --n_ens 20 \
                     --log_wandb 1