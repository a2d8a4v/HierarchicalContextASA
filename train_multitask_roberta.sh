#!/usr/bin/env bash
##SBATCH -p sm
##SBATCH -x sls-sm-1,sls-2080-[1,3],sls-1080-3,sls-sm-5
#SBATCH -p gpu
#SBATCH -x sls-titan-[0-2]
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=24000
#SBATCH --job-name="gopt"
#SBATCH --output=../exp/log_%j.txt

stage=0
stop_stage=10000

root_dir=/share/nas167/a2y3a1N0n2Yann/BertModel/basic_pretrained_model
output_dir=$root_dir/efcamdat-multitask-roberta-base
repeat_list=(0 1 2 3 4)
seed_list=(825 1225 513 759 985)

CUDA=0

. parse_options.sh
set -euo pipefail

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ] ; then
    lrs=(1e-5 8e-6 6e-6 4e-6 2e-6)

    count=0
    mkdir -pv $output_dir/$repeat
    for repeat in "${repeat_list[@]}"; do
        CUDA_VISIBLE_DEVICES=$CUDA python3 posttraining_multitask_roberta.py \
            --output_dir $output_dir/$repeat --lr "${lrs[$repeat]}" --seed "${seed_list[$repeat]}"
        count=$((count + 1))
    done
fi
