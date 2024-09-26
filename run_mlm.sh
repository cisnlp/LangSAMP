WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --rdzv_endpoint=0.0.0.0:22000 --nproc_per_node=4 ./run_mlm.py \
  --config_name xlm-roberta-base \
  --train_files_path /mounts/data/proj/yihong/glot500_data_sampled/Glot500_0.3_30 \
  --tokenizer_name cis-lmu/glot500-base \
  --output_dir ./langsamp \
  --cache_dir ./cache \
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 8 \
  --fp16 True \
  --do_train \
  --num_train_epochs 20 \
  --save_steps 5000 \
  --ddp_timeout 259200 \
  --preprocessing_num_workers 8 \
  --logging_steps 500 \
  --use_lang_embedding True \
  --use_script_embedding True \
  --language_script_dict_path /mounts/data/proj/yihong/decoupled_training/full_dicts.pkl \
  --remove_unused_columns False \
  --seed 42 \
  --continued_pretrain True \
  --ddp_find_unused_parameters False