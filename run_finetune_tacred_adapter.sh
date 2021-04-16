# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# please use 4 GPU
task=tacred
GPU='4'
CUDA_VISIBLE_DEVICES=$GPU python examples/run_finetune_TACRED_adapter.py \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --config_name roberta-large \
    --do_train  \
    --do_eval   \
    --evaluate_during_training \
    --task_name=$task     \
    --data_dir=/hdd1/seokwon/data/TACRED/data/json  \
    --output_dir=./proc_data  \
    --comment 'combine-adapter-dif-trf' \
    --max_seq_length=184  \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate=1e-5 \
    --gradient_accumulation_steps=1 \
    --max_steps=12000  \
    --model_name=roberta-large  \
    --overwrite_output_dir   \
    --overwrite_cache \
    --warmup_steps=200 \
    --negative_sample=45000 \
    --save_steps=500 \
    --freeze_bert="" \
    --freeze_adapter="True" \
    --adapter_size 768 \
    --adapter_list "0,11,22" \
    --adapter_skip_layers 0 \
    --eval_steps 1000 \
    --logging_steps 100 \
    --meta_fac_adaptermodel="/hdd1/seokwon/K-Adapter/pretrained_models/fac-adapter/pytorch_model.bin" \
    --meta_lin_adaptermodel="/hdd1/seokwon/K-Adapter/pretrained_models/lin-adapter/pytorch_model.bin"