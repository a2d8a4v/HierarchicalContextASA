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
CUDA=0
BCUDA=0

. parse_options.sh
set -euo pipefail

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ] ; then
    lrs="4e-5 2e-5"
    count=0
    for lr in $lrs; do
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 train.py --cuda --gpu $CUDA --bert_gpu $BCUDA \
            --data_dir $data_dir \
            --cache_dir $cache_dir \
            --embedding_path $glove_path \
            --model PAL \
            --save_root $save_root \
            --log_root $log_path \
            --grad_clip \
            --batch_size 16 \
            --gradient_accumulation_steps 8 \
            --n_epochs 25 \
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
            --sent_max_len 1600 \
            --lr $lr \
            --verbose_show_in_gpumonitor "lr:$lr stage:$stage count:$count"
            # --verbose_show_in_gpumonitor "lr:$lr stage:$stage"
        count=$((count + 1))
                # --lr_descent \
                        # --wandb \
    done
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ] ; then
    lrs="8e-5 6e-5 4e-5 2e-5"
    count=0
    for lr in $lrs; do
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 train.py --cuda --gpu $CUDA --bert_gpu $BCUDA \
            --data_dir $data_dir \
            --cache_dir $cache_dir \
            --embedding_path $glove_path \
            --model PAL \
            --save_root $save_root \
            --log_root $log_path \
            --grad_clip \
            --batch_size 16 \
            --gradient_accumulation_steps 8 \
            --n_epochs 25 \
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
            --sent_max_len 1600 \
            --lr $lr \
            --language_use \
            --verbose_show_in_gpumonitor "lr:$lr stage:$stage count:$count"
            # --verbose_show_in_gpumonitor "lr:$lr stage:$stage"
        count=$((count + 1))
                # --lr_descent \
                        # --wandb \
    done
fi
