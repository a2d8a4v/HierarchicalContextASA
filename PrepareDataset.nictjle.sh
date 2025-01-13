#!/usr/bin/env bash

dataset="$1"
datadir="$2"
task="$3"
sentaspara="$4"
interviewer="$5"
data_root=/share/nas167/a2y3a1N0n2Yann/speechocean/espnet_amazon/egs/nict_jle/asr3/data

# -u to check bounded args, -e to exit when error
#set -u
set -e

if [ ! -n "$dataset" ]; then
    echo "lose the dataset name!"
    exit
fi

if [ ! -n "$datadir" ]; then
    echo "lose the data directory!"
    exit
fi

if [ ! -n "$task" ]; then
    task=single
fi

if [ ! -n "$sentaspara" ]; then
    sentaspara=sent
fi

if [ ! -n "$interviewer" ]; then
    interviewer=false
else
    echo -e "\033[34m[Shell] Use the response of interviewer! \033[0m"
fi

type=(trn_combo dev_combo eval_combo)

echo -e "\033[34m[Shell] Convert tsv to jsonl! \033[0m"
for i in ${type[*]}
    do
        python script.nictjle/nictjle2jsonl.py --input_text_tsv_file_path $data_root/$i/text.tsv --input_labels_file_path $data_root/all/text.pkl.labels.final --output_dir_path $datadir --sentaspara $sentaspara
        if [ $interviewer == true ]; then
            python script.nictjle/nictjle2jsonl.py --input_text_tsv_file_path $data_root/$i/text.tsv --input_labels_file_path $data_root/all/text.pkl.labels.final --output_dir_path $datadir --sentaspara $sentaspara --interviewer
        fi
    done

echo -e "\033[34m[Shell] Create Fluency Tagged WordList! \033[0m"
python script.nictjle/fluencytagged.py --input_text_tsv_file_path $data_root/all/text.tsv --input_tags_file_path $data_root/all/text.pkl.labels.final --input_vocab_profile_file_path $datadir/cefrj1.6_c1c2.final.txt --output_dir_path $datadir

echo -e "\033[34m[Shell] Create Vocabulary! \033[0m"
for i in ${type[*]}
    do
        python script.nictjle/nictjle2jsonl.py --input_text_tsv_file_path $data_root/$i/text.tsv --input_labels_file_path $data_root/all/text.pkl.labels.final --output_dir_path nictjle --sentaspara $sentaspara --interviewer --combine --bert
    done
if [ $interviewer == true ]; then
    python script.nictjle/nictjle2jsonl.py --input_text_tsv_file_path $data_root/all/text.tsv --input_labels_file_path $data_root/all/text.pkl.labels.final --output_dir_path nictjle --sentaspara $sentaspara --interviewer --combine
    python script.nictjle/createVoc.py --dataset $dataset --data_path $datadir/all.$sentaspara.combine.label.jsonl --combine
else
    python script.nictjle/nictjle2jsonl.py --input_text_tsv_file_path $data_root/all/text.tsv --input_labels_file_path $data_root/all/text.pkl.labels.final --output_dir_path nictjle --sentaspara $sentaspara
    python script.nictjle/createVoc.py --dataset $dataset --data_path $datadir/all.$sentaspara.label.jsonl
fi

echo -e "\033[34m[Shell] Get low tfidf words from training set! \033[0m"
python script.nictjle/lowTFIDFWords.py --dataset $dataset --data_path $datadir/all.$sentaspara.label.jsonl --sentaspara $sentaspara

echo -e "\033[34m[Shell] Get word2sent edge feature! \033[0m"
for i in ${type[*]}
    do
        python script.nictjle/calw2sTFIDF.py --dataset $dataset --data_path $datadir/$i.$sentaspara.label.jsonl --sentaspara $sentaspara
    done

if [ "$task" == "multi" ]; then
    echo -e "\033[34m[Shell] Get word2doc edge feature! \033[0m"
    for i in ${type[*]}
        do
            python script.nictjle/calw2dTFIDF.py --dataset $dataset --data_path $datadir/$i.$sentaspara.label.jsonl --sentaspara $sentaspara
        done
fi

echo -e "\033[34m[Shell] The preprocess of dataset $dataset has finished! \033[0m"


