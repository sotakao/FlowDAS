# python main_research_lookback_sevir.py \
#   --dataset sevir \
#   --beta_fn t^2 \
#   --sigma_coef 1 \
#   --use_wandb 0 \
#   --debug 0 \
#   --overfit 0 \
#   --task_config ./configs/super_resolution_config.yaml \
#   --sample_only 0 \
#   --sevir_datapath ./data/sevir_lr \
#   --save_checkpoint ./checkpoints

python main_research_sevir_gen_sample4GL_621_opensource_0710_copy.py \
  --dataset sevir \
  --beta_fn t^2 \
  --sigma_coef 1 \
  --use_wandb 0 \
  --debug 0 \
  --overfit 0 \
  --task_config ./configs/super_resolution_config.yaml \
  --sample_only 1 \
  --load_path ./checkpoints/latest.pt \
  --savedir ./results \
  --sevir_datapath ./data/sevir_lr \
  --MC_times 25 \
  --exp_times 100 \
  --auto_step 6 \
  --exp_id_times 1