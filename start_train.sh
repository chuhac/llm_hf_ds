deepspeed --include localhost:0,1,2,3,4,5,6,7 \
train_with_trainer.py \
--model_name_or_path ../baichuan2 \
--train_data /path/to/data --per_device_train_batch_size 1 \
--gradient_accumulation_steps 32  \
--gradient_checkpointing --output_dir ./outputs \
--seed 42 --bf16 --deepspeed_file ./ds_cfg.json \
--lr_scheduler_type cosine \
--save_steps 500 \
--learning_rate 3e-6