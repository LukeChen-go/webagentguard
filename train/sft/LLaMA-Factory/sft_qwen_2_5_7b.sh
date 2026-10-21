export HF_HOME=./
# W&B key: run `wandb login` or `export WANDB_API_KEY=<your_key>` before launching
export WANDB_PROJECT="pia_detection"
export WANDB_NAME="qwen2.5-7b"
device="0,1"
save_path="saves/full/Qwen2.5-VL-7B/SFT-7B"

model="Qwen/Qwen2.5-VL-7B-Instruct"
batch_size=8
cutoff_len=8192


DS_SKIP_CUDA_CHECK=1 CUDA_VISIBLE_DEVICES=$device llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path $model \
    --image_max_pixels 802816 \
    --preprocessing_num_workers 16 \
    --finetuning_type full \
    --template qwen2_vl \
    --flash_attn auto \
    --dataset_dir data \
    --dataset detection_sft_data \
    --cutoff_len $cutoff_len \
    --learning_rate 5e-6 \
    --num_train_epochs 4.0 \
    --max_samples 100000 \
    --per_device_train_batch_size $batch_size \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 2 \
    --save_strategy epoch \
    --warmup_ratio 0.1 \
    --packing False \
    --output_dir $save_path \
    --bf16 True \
    --plot_loss True \
    --ddp_timeout 180000000 \
    --optim adamw_torch \
    --deepspeed "examples/deepspeed/ds_z3_offload_config.json" \
    --report_to wandb