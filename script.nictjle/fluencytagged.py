#!/bin/env python
#coding:utf-8

import os
import csv
import sys
import json
import pickle
import argparse
import numpy as np


def GetType(path):
    filename = path.split("/")[-2]
    return filename

def args_init():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_text_tsv_file_path', type=str, default='data/CNNDM/train.label.jsonl', help='File to deal with')
    parser.add_argument('--input_tags_file_path', type=str, default='data/CNNDM/text.pkl.labels.final', help='File to deal with')
    parser.add_argument('--input_vocab_profile_file_path', type=str, default='data/CNNDM/text.pkl.labels.final', help='File to deal with')
    parser.add_argument('--output_dir_path', type=str, default='CNNDM', help='dataset name')
    
    args = parser.parse_args()
    return args

def readText(input_file, output_col_names=False, quotechar=None):
    """Reads a tab separated value file."""
    columns = []
    with open(input_file, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for i, line in enumerate(reader):
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            if i == 0:
                columns = line
                continue
            lines.append(line)
        if output_col_names:
            return columns, lines
        return lines

def pikleOpen(filename):
    file_to_read = open( filename , "rb" )
    p = pickle.load( file_to_read )
    return p

def convert_int_if(labels_list):
    labels_list = [int(l) if str(l).isnumeric() else l for l in labels_list]
    return labels_list

def read_nictjle_format(columns, lines, labels_dict):
    rtn = list()
    utt_idx  = columns.index('id')
    content_idx = columns.index('pre_token')
    for line in lines:
        id      = line[utt_idx]
        content = line[content_idx].split()
        fp      = convert_int_if(labels_dict.get(id).get('filled_pauses_simple').get('label').split())
        rtn.append({'id': id, 'text': content, 'filled_pauses': fp})
    return rtn

def read_vocab_profile(input_file, quotechar=None):
    """Reads a tab separated value file."""
    columns = []
    with open(input_file, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for i, line in enumerate(reader):
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            if i == 0:
                columns = line
                continue
            lines.append(line)
    return open_multiple_words(lines)
    
def open_multiple_words(lines):
    s = {}
    for i, line in enumerate(lines):
        word = line[1]
        pos  = line[2]
        cefr = int(line[3])
        s.setdefault(
            word,
            list()
        ).append([pos, cefr])
    return s

if __name__ == '__main__':
    
    args = args_init()
    input_text_tsv_file_path = args.input_text_tsv_file_path
    input_tags_file_path = args.input_tags_file_path
    input_vocab_profile_file_path = args.input_vocab_profile_file_path
    output_dir_path = args.output_dir_path
    
    if not os.path.exists(output_dir_path): os.makedirs(output_dir_path)

    columns, text_lines = readText(input_text_tsv_file_path, output_col_names=True)
    labels_dict = pikleOpen(input_tags_file_path)
    nictjle_data_list = read_nictjle_format(columns, text_lines, labels_dict)
    vocab_profile_list = read_vocab_profile(input_vocab_profile_file_path)

    filled_pauses_tokens = {}
    for paragraph in nictjle_data_list:
        fp = paragraph['filled_pauses']
        text = paragraph['text']
        for f, t in zip(fp, text):
            if f == 1:
                filled_pauses_tokens.setdefault(t, []).append(1)
    filled_pauses_tokens = {f: sum(c_list) for f, c_list in filled_pauses_tokens.items() if sum(c_list) > 1 and (f not in vocab_profile_list)}
    filled_pauses_tokens = dict(sorted(filled_pauses_tokens.items(), key=lambda item: item[1], reverse=True))

    fname = GetType(input_text_tsv_file_path) + ".filled_pauses.txt"
    saveFile = os.path.join(output_dir_path, fname)

    f = open(saveFile, "w")
    for t, c in filled_pauses_tokens.items():
        f.write("{}\n".format(t))
