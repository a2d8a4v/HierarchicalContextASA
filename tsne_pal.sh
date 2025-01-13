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
    # lrs="1e-5 8e-6 6e-6 4e-6 2e-6"
    lrs="4e-5"
    pred_method=all_s
    for lr in $lrs; do
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 tsne.py --cuda --gpu $CUDA --bert_gpu $BCUDA --data_dir $data_dir --cache_dir $cache_dir --embedding_path $glove_path --model PAL --save_root $save_root --log_root $log_path --tsne --batch_size 1 --test_model trainbestmodel \
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
            --lr $lr
    done
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ] ; then
    # lrs=" 8e-6 6e-6 4e-6 2e-6" # need rerun
    lrs="1e-4"
    # rlrs="1e-5"
    pred_method=test_wdc
    for lr in $lrs; do
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 tsne.py --cuda --gpu $CUDA --bert_gpu $BCUDA --data_dir $data_dir --cache_dir $cache_dir --embedding_path $glove_path --model PAL --save_root $save_root --log_root $log_path --tsne --batch_size 1 --test_model trainbestmodel \
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
            --language_use
    done
fi
