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
    lrs=(1e-4 8e-5 6e-5 4e-5 2e-5) # done
    repeat_list=(  1 2 3 4 )
    for repeat in "${repeat_list[@]}"; do
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 evaluation.py --cuda --gpu $CUDA --bert_gpu $BCUDA --data_dir $data_dir --cache_dir $cache_dir --embedding_path $glove_path --model PAL --save_root $save_root --log_root $log_path --tsne --batch_size 1 --test_model trainbestmodel \
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
            --lr "${lrs[$repeat]}"
    done

    input_dirs=""
    root_dir=output/NICTJLE
    lrs=(0.0001 8e-05 6e-05 4e-05 2e-05)
    for repeat in "${repeat_list[@]}"; do
        input_dirs+="$root_dir/PAL.sent.${lrs[$repeat]}.reweight2.0.regression.linear.glove.pmi5.interviewer "
    done
    echo $input_dirs
    python3 collect_summary.py --input_dirs $input_dirs \
                               --output_dir $root_dir/0_PAL
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ] ; then
    lrs=(1e-4 8e-5 6e-5 4e-5 2e-5) # done
    repeat_list=( 0 1 2 3 4 )
    for repeat in "${repeat_list[@]}"; do
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 evaluation.py --cuda --gpu $CUDA --bert_gpu $BCUDA --data_dir $data_dir --cache_dir $cache_dir --embedding_path $glove_path --model PAL --save_root $save_root --log_root $log_path --tsne --batch_size 1 --test_model trainbestmodel \
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
            --lr "${lrs[$repeat]}" \
            --language_use
    done

    input_dirs=""
    root_dir=output/NICTJLE
    lrs=(0.0001 8e-05 6e-05 4e-05 2e-05)
    for repeat in "${repeat_list[@]}"; do
        input_dirs+="$root_dir/PAL.sent.${lrs[$repeat]}.reweight2.0.regression.linear.glove.pmi5.interviewer.lu "
    done
    echo $input_dirs
    python3 collect_summary.py --input_dirs $input_dirs \
                               --output_dir $root_dir/0_PAL_lu
fi
