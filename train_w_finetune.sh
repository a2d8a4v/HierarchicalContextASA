#!/usr/bin/env bash

stage=0
stop_stage=10000

data_dir=nictjle
cache_dir=cache/NICTJLE
glove_path=glove/glove.6B.300d.txt
save_root=output/NICTJLE
log_path=log/NICTJLE
sentaspara=para
model_type=HSG
problem_type=regression
head=linear
wandb=
pred_method=
CUDA=0
BCUDA=0

. parse_options.sh
set -euo pipefail

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ] ; then
    lrs="8e-6 6e-6 4e-6 2e-6"
    pred_method=all_s
    count=0
    for lr in $lrs; do
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 train.py --cuda --gpu $CUDA --bert_gpu $BCUDA \
            --data_dir $data_dir \
            --cache_dir $cache_dir \
            --embedding_path $glove_path \
            --model HSDG \
            --save_root $save_root \
            --log_root $log_path \
            --lr_descent \
            --grad_clip \
            --batch_size 64 \
            --gradient_accumulation_steps 2 \
            --n_epochs 25 \
            --wandb \
            --num_workers 0 \
            --sentaspara sent \
            --doc_max_timesteps 88 \
            --problem_type regression \
            --reweight \
            --rw_alpha 2 \
            --head $head \
            --word_embedding \
            --pmi_window_width 5 \
            --interviewer \
            --pred_method $pred_method \
            --cefr_word \
            --cefr_info graph_init \
            --bert \
            --bert_train_finetune \
            --bert_model_path roberta-base \
            --bert_roberta_to_long \
            --sent_max_len 1600 \
            --lr $lr \
            --verbose_show_in_gpumonitor "lr:$lr pm:$pred_method stage:$stage count:$count" \
            --restore_model bestmodel
        count=$((count + 1))
    done
fi


if [ $stage -le 1 ] && [ $stop_stage -ge 1 ] ; then
    lrs="8e-6 6e-6 4e-6 2e-6"
    pred_method=test_wdc2
    count=0
    for lr in $lrs; do
        CUDA_LAUNCH_BLOCKING=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 train.py --cuda --gpu $CUDA --bert_gpu $BCUDA \
            --data_dir $data_dir \
            --cache_dir $cache_dir \
            --embedding_path $glove_path \
            --model HSDG \
            --save_root $save_root \
            --log_root $log_path \
            --lr_descent \
            --grad_clip \
            --batch_size 64 \
            --gradient_accumulation_steps 2 \
            --n_epochs 25 \
            --wandb \
            --num_workers 0 \
            --sentaspara sent \
            --doc_max_timesteps 88 \
            --problem_type regression \
            --reweight \
            --rw_alpha 2 \
            --head $head \
            --word_embedding \
            --pmi_window_width 5 \
            --interviewer \
            --pred_method $pred_method \
            --cefr_word \
            --cefr_info graph_init \
            --bert \
            --bert_train_finetune \
            --bert_model_path roberta-base \
            --bert_roberta_to_long \
            --sent_max_len 1600 \
            --lr $lr \
            --verbose_show_in_gpumonitor "lr:$lr pm:$pred_method stage:$stage count:$count"
        count=$((count + 1))
    done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ] ; then
    # lrs="8e-6 6e-6 4e-6 2e-6"
    lrs="1e-5"
    pred_method=hec_s
    count=0
    for lr in $lrs; do
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 train.py --cuda --gpu $CUDA --bert_gpu $BCUDA \
            --data_dir $data_dir \
            --cache_dir $cache_dir \
            --embedding_path $glove_path \
            --model HSDG \
            --save_root $save_root \
            --log_root $log_path \
            --lr_descent \
            --grad_clip \
            --batch_size 64 \
            --gradient_accumulation_steps 2 \
            --n_epochs 25 \
            --wandb \
            --num_workers 0 \
            --sentaspara sent \
            --doc_max_timesteps 88 \
            --problem_type regression \
            --reweight \
            --rw_alpha 2 \
            --head $head \
            --word_embedding \
            --pmi_window_width 5 \
            --interviewer \
            --pred_method $pred_method \
            --cefr_word \
            --cefr_info graph_init \
            --bert \
            --bert_train_finetune \
            --bert_model_path roberta-base \
            --bert_roberta_to_long \
            --sent_max_len 1600 \
            --lr $lr \
            --verbose_show_in_gpumonitor "lr:$lr pm:$pred_method stage:$stage count:$count"
        count=$((count + 1))
    done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ] ; then
    lrs="8e-6 6e-6 4e-6 2e-6"
    #done 1e-5 
    pred_method=sde_s
    count=0
    for lr in $lrs; do
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 train.py --cuda --gpu $CUDA --bert_gpu $BCUDA \
            --data_dir $data_dir \
            --cache_dir $cache_dir \
            --embedding_path $glove_path \
            --model HSDG \
            --save_root $save_root \
            --log_root $log_path \
            --lr_descent \
            --grad_clip \
            --batch_size 64 \
            --gradient_accumulation_steps 2 \
            --n_epochs 25 \
            --wandb \
            --num_workers 0 \
            --sentaspara sent \
            --doc_max_timesteps 88 \
            --problem_type regression \
            --reweight \
            --rw_alpha 2 \
            --head $head \
            --word_embedding \
            --pmi_window_width 5 \
            --interviewer \
            --pred_method $pred_method \
            --cefr_word \
            --cefr_info graph_init \
            --bert \
            --bert_train_finetune \
            --bert_model_path roberta-base \
            --bert_roberta_to_long \
            --sent_max_len 1600 \
            --lr $lr \
            --verbose_show_in_gpumonitor "lr:$lr pm:$pred_method stage:$stage count:$count" --restore_model bestmodel
        exit
        count=$((count + 1))
    done
fi
