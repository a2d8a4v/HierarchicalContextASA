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
    lrs=(1e-5 8e-6 6e-6 4e-6 2e-6) # done
    repeat_list=( 0 1 2 3 4 )
    pred_method=sde
    # for repeat in "${repeat_list[@]}"; do
    #     PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 evaluation.py --cuda --gpu $CUDA --bert_gpu $BCUDA --data_dir $data_dir --cache_dir $cache_dir --embedding_path $glove_path --model HSDG --save_root $save_root --log_root $log_path --tsne --batch_size 1 --test_model trainbestmodel \
    #         --sentaspara sent \
    #         --doc_max_timesteps 88 \
    #         --problem_type regression \
    #         --reweight \
    #         --rw_alpha 2 \
    #         --head $head \
    #         --word_embedding \
    #         --pmi_window_width 5 \
    #         --interviewer \
    #         --pred_method $pred_method \
    #         --cefr_word \
    #         --cefr_info graph_init \
    #         --bert \
    #         --bert_model_path roberta-base \
    #         --bert_roberta_to_long \
    #         --sent_max_len 1600 \
    #         --lr "${lrs[$repeat]}"
    # done

    input_dirs=""
    root_dir=output/NICTJLE
    lrs=(1e-05 8e-06 6e-06 4e-06 2e-06)
    for repeat in "${repeat_list[@]}"; do
        input_dirs+="$root_dir/HSDG.sent.${lrs[$repeat]}.reweight2.0.regression.linear.sde.roberta.glove.pmi5.interviewer.cefrgraph_init/ "
    done
    echo $input_dirs
    python3 collect_summary.py --input_dirs $input_dirs \
                               --output_dir $root_dir/0_hier_$pred_method
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ] ; then
    lrs=(1e-5 8e-6 6e-6 4e-6 2e-6) # done
    repeat_list=( 0 1 2 3 4 )
    pred_method=hec
    # for repeat in "${repeat_list[@]}"; do
    #     PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 evaluation.py --cuda --gpu $CUDA --bert_gpu $BCUDA --data_dir $data_dir --cache_dir $cache_dir --embedding_path $glove_path --model HSDG --save_root $save_root --log_root $log_path --tsne --batch_size 1 --test_model trainbestmodel \
    #         --sentaspara sent \
    #         --doc_max_timesteps 88 \
    #         --problem_type regression \
    #         --reweight \
    #         --rw_alpha 2 \
    #         --head $head \
    #         --word_embedding \
    #         --pmi_window_width 5 \
    #         --interviewer \
    #         --pred_method $pred_method \
    #         --cefr_word \
    #         --cefr_info graph_init \
    #         --bert \
    #         --bert_model_path roberta-base \
    #         --bert_roberta_to_long \
    #         --sent_max_len 1600 \
    #         --lr "${lrs[$repeat]}"
    # done

    input_dirs=""
    root_dir=output/NICTJLE
    lrs=(1e-05 8e-06 6e-06 4e-06 2e-06)
    for repeat in "${repeat_list[@]}"; do
        input_dirs+="$root_dir/HSDG.sent.${lrs[$repeat]}.reweight2.0.regression.linear.hec.roberta.glove.pmi5.interviewer.cefrgraph_init/ "
    done
    echo $input_dirs
    python3 collect_summary.py --input_dirs $input_dirs \
                               --output_dir $root_dir/0_hier_$pred_method
fi


if [ $stage -le 2 ] && [ $stop_stage -ge 2 ] ; then
    lrs=(1e-5 8e-6 6e-6 4e-6 2e-6) # done
    repeat_list=( 0 1 2 3 4 )
    pred_method=ehg
    # for repeat in "${repeat_list[@]}"; do
    #     PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 evaluation.py --cuda --gpu $CUDA --bert_gpu $BCUDA --data_dir $data_dir --cache_dir $cache_dir --embedding_path $glove_path --model HSDG --save_root $save_root --log_root $log_path --tsne --batch_size 1 --test_model trainbestmodel \
    #         --sentaspara sent \
    #         --doc_max_timesteps 88 \
    #         --problem_type regression \
    #         --reweight \
    #         --rw_alpha 2 \
    #         --head $head \
    #         --word_embedding \
    #         --pmi_window_width 5 \
    #         --interviewer \
    #         --pred_method $pred_method \
    #         --cefr_word \
    #         --cefr_info graph_init \
    #         --bert \
    #         --bert_model_path roberta-base \
    #         --bert_roberta_to_long \
    #         --sent_max_len 1600 \
    #         --lr "${lrs[$repeat]}"
    # done

    input_dirs=""
    root_dir=output/NICTJLE
    lrs=(1e-05 8e-06 6e-06 4e-06 2e-06)
    for repeat in "${repeat_list[@]}"; do
        input_dirs+="$root_dir/HSDG.sent.${lrs[$repeat]}.reweight2.0.regression.linear.ehg.roberta.glove.pmi5.interviewer.cefrgraph_init/ "
    done
    echo $input_dirs
    python3 collect_summary.py --input_dirs $input_dirs \
                               --output_dir $root_dir/0_hier_$pred_method
fi


if [ $stage -le 3 ] && [ $stop_stage -ge 3 ] ; then
    lrs=(1e-5 8e-6 6e-6 4e-6 2e-6) # done
    repeat_list=( 0  )
    pred_method=acg
    for repeat in "${repeat_list[@]}"; do
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 evaluation.py --cuda --gpu $CUDA --bert_gpu $BCUDA --data_dir $data_dir --cache_dir $cache_dir --embedding_path $glove_path --model HSDG --save_root $save_root --log_root $log_path --tsne --batch_size 1 --test_model trainbestmodel \
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
            --bert_model_path roberta-base \
            --bert_roberta_to_long \
            --sent_max_len 1600 \
            --lr "${lrs[$repeat]}"
    done

    input_dirs=""
    root_dir=output/NICTJLE
    # lrs=(1e-05 8e-06 6e-06 4e-06 2e-06)
    lrs=(1e-05)
    for repeat in "${repeat_list[@]}"; do
        input_dirs+="$root_dir/HSDG.sent.${lrs[$repeat]}.reweight2.0.regression.linear.ehg.roberta.glove.pmi5.interviewer.cefrgraph_init/ "
    done
    echo $input_dirs
    python3 collect_summary.py --input_dirs $input_dirs \
                               --output_dir $root_dir/0_hier_$pred_method
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ] ; then
    lrs=(1e-5 8e-6 6e-6 4e-6 2e-6) # done
    repeat_list=( 0  )
    pred_method=hsag
    for repeat in "${repeat_list[@]}"; do
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 evaluation.py --cuda --gpu $CUDA --bert_gpu $BCUDA --data_dir $data_dir --cache_dir $cache_dir --embedding_path $glove_path --model HSDG --save_root $save_root --log_root $log_path --tsne --batch_size 1 --test_model trainbestmodel \
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
            --bert_model_path roberta-base \
            --bert_roberta_to_long \
            --sent_max_len 1600 \
            --lr "${lrs[$repeat]}"
    done

    input_dirs=""
    root_dir=output/NICTJLE
    # lrs=(1e-05 8e-06 6e-06 4e-06 2e-06)
    lrs=(1e-05)
    for repeat in "${repeat_list[@]}"; do
        input_dirs+="$root_dir/HSDG.sent.${lrs[$repeat]}.reweight2.0.regression.linear.ehg.roberta.glove.pmi5.interviewer.cefrgraph_init/ "
    done
    echo $input_dirs
    python3 collect_summary.py --input_dirs $input_dirs \
                               --output_dir $root_dir/0_hier_$pred_method
fi
