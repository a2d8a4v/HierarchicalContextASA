#!/bin/env python
#coding:utf-8

import os
import csv
import sys
import json
import pickle
import argparse
import numpy as np

pos_tag_labels = ['X', '.', '\'\'', 'ADD', 'AFX', 'CC', 'CD', 'DT', 'EX', 'FW', 'GW', 'IN', 'JJ', 'JJR', 'JJS', 'LS',
                  'MD', 'NFP', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM',
                  'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
deprel_labels = ['X', 'nsubj', 'obj', 'iobj', 'csubj', 'ccomp', 'xcomp', 'obl', 'vocative', 'expl', 'dislocated',
                 'advcl', 'advmod', 'discourse', 'aux', 'cop', 'mark', 'nmod', 'appos', 'nummod', 'acl', 'amod', 'det',
                 'clf', 'case', 'conj', 'cc', 'fixed', 'flat', 'compound', 'list', 'parataxis', 'orphan', 'goeswith',
                 'reparandum', 'punct', 'root', 'dep']
ged_simple_labels = [0, 1]
ged_median_labels = ['X', 'n', 'v', 'mo', 'aj', 'av', 'prp', 'at', 'pn', 'con', 'rel', 'itr', 'o']
ged_difficult_labels = ['X', 'n_inf', 'n_num', 'n_cs', 'n_cnt', 'n_cmp', 'n_lxc', 'v_inf', 'v_agr', 'v_fml', 'v_tns', 'v_asp',
                        'v_vo', 'v_fin', 'v_ng', 'v_qst', 'v_cmp', 'v_lxc', 'mo_lxc', 'aj_inf', 'aj_us', 'aj_num', 'aj_agr',
                        'aj_qnt', 'aj_cmp', 'aj_lxc', 'av_inf', 'av_us', 'av_pst', 'av_lxc', 'prp_cmp', 'prp_lxc1', 'prp_lxc2',
                        'at', 'pn_inf', 'pn_agr', 'pn_cs', 'pn_lxc', 'con_lxc', 'rel_cs', 'rel_lxc', 'itr_lxc', 'o_je', 'o_lxc',
                        'o_odr', 'o_uk', 'o_uit',]
filled_pauses_labels = ['X', 'F']
disfluency_simple_labels = [0, 1]
disfluency_median_labels = ['X', 'SC', 'R']
native_language_labels = ['X', 'italiano', 'japanese']
sst_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
topic_labels = ['X', 'camping', 'car_accident', 'classroom', 'department_store', 'electric_shop', 'electrical_shop',
                'grocery_store', 'invitation', 'landlord', 'map', 'movie', 'neighborhood', 'restaurant', 'room', 'shopping',
                'ski', 'stray_cat', 'train', 'train_station', 'travel', 'zoo']
w_cefr_labels = ['X', 'a1', 'a2', 'b1', 'b2', 'c1', 'c2']

def get_full_tag(tag_text):
    return "<{}>".format(tag_text.split()[0].replace('</','').replace('<','').replace('>',''))

def get_pos_labels():
    return pos_tag_labels

def get_deprel_labels():
    return deprel_labels

def get_native_language_labels():
    return native_language_labels

def get_sst_labels():
    return sst_labels

def get_topic_labels():
    return topic_labels

def get_gra_sim_labels():
    return ged_simple_labels

def get_gra_med_labels():
    return ged_median_labels

def get_gra_dif_labels():
    return ged_difficult_labels

def get_dis_sim_labels():
    return disfluency_simple_labels

def get_dis_med_labels():
    return disfluency_median_labels

def get_filled_pauses_median_labels():
    return filled_pauses_labels

def get_w_cefr_labels():
    return w_cefr_labels


def GetType(path):
    filename = path.split("/")[-2]
    return filename

def args_init():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_text_tsv_file_path', type=str, default='data/CNNDM/train.label.jsonl', help='File to deal with')
    parser.add_argument('--input_labels_file_path', type=str, default='data/CNNDM/train.label.jsonl', help='File to deal with')
    parser.add_argument('--output_dir_path', type=str, default='CNNDM', help='dataset name')
    parser.add_argument('--sentaspara', type=str, default='sent', choices=['sent', 'para'], help='dataset name')
    parser.add_argument('--interviewer', action='store_true', default=False, help='Interviewer')
    parser.add_argument('--combine', action='store_true', default=False, help='Combine responses of nterviewer and interviewee')
    parser.add_argument('--bert', action='store_true', default=False, help='Feature used for BERT')

    
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

def parse_nictjle_content(content, interviewer=False, combine=False, return_speaker_list=False):
    
    # special tokens
    unk_token = '[UNK]'
    splitter_qa = '[SEP_QA]' # it is a pesudo label for distinct <A> from <B>
    splitter_pair = '[PAIR]' # it is a pesudo label for distinct pairs from each other
    newline_token = '[NEW]'
    
    token_list = content.split()
    special_tokens_list = [splitter_qa, splitter_pair, newline_token]
    
    collect_point = False if not interviewer else True
    sentence  = []
    sentences = []
    which_spk = []
    Can = 1
    Int = 0
    
    if combine:
        for i, t in enumerate(token_list):
            if t in special_tokens_list:
                which_spk.append(Int if t == splitter_qa else Can)
                sentences.append(' '.join(sentence))
                sentence = []
                continue
            elif t == splitter_pair:
                sentences.append(' '.join(sentence))
                sentence = []
                continue
            if t == 'grammarerrorword':
                t = unk_token
            sentence.append(t)
            if len(token_list)-1 == i:
                which_spk.append(Can)
                sentences.append(' '.join(sentence))
        if return_speaker_list:
            return which_spk, sentences
        return sentences
    
    for i, t in enumerate(token_list):
        if t == splitter_qa:
            if interviewer:
                which_spk.append(Int)
                sentences.append(' '.join(sentence))
                sentence = []
            collect_point = True if not interviewer else False
            continue
        elif t == splitter_pair:
            if not interviewer:
                which_spk.append(Can)
                sentences.append(' '.join(sentence))
                sentence = []
            collect_point = False if not interviewer else True
            continue
        if not collect_point:
            continue
        if t == 'grammarerrorword':
            t = unk_token
        sentence.append(t)
        if len(token_list)-1 == i:
            which_spk.append(Can)
            sentences.append(' '.join(sentence))
    if return_speaker_list:
        return which_spk, sentences
    return sentences


def combine_tags(tokens, id, tag_array, get_name=''):
    assert get_name != '', 'get_name should have value!'
    _median = ['X'] * len(tokens.split(' '))
    tokens_list = np.array(tokens.split(' '))
    sep_qa_indexes_list = np.where(tokens_list=='[SEP_QA]')[0].tolist()
    pair_indexes_list   = np.where(tokens_list=='[PAIR]')[0].tolist()
    for sep_qa_idx in sep_qa_indexes_list:
        _median[sep_qa_idx] = '[SEP_QA]'
    for pair_idx in pair_indexes_list:
        _median[pair_idx] = '[PAIR]'
    if 'X' in tag_array:
        tag_array.remove('X')
    for tag in tag_array:
        get_tag_labels = other_labels.get(id).get(get_name).get(get_full_tag(tag)).split()

        tag_sep_qa_indexes_list = np.where(np.array(get_tag_labels)=='[SEP_QA]')[0].tolist()
        tag_pair_indexes_list   = np.where(np.array(get_tag_labels)=='[PAIR]')[0].tolist()
        assert check_same_contents(tag_sep_qa_indexes_list, sep_qa_indexes_list), '{}: {} \n {}: {}'.format(len(tokens.split()), tokens, len(get_tag_labels), get_tag_labels)
        assert check_same_contents(tag_pair_indexes_list, pair_indexes_list), '{}: {} \n {}: {}'.format(len(tokens.split()), tokens, len(get_tag_labels), get_tag_labels)

        get_tag_labels = [int(l) if l.isnumeric() else 0 for l in get_tag_labels]
        get_tag_labels_np = np.array(get_tag_labels)
        get_labels_update_indexes = np.where(get_tag_labels_np==1)[0].tolist()
        for idx in get_labels_update_indexes:
            _median[idx] = tag
    return _median

def convert_int_if(labels_list):
    labels_list = [int(l) if str(l).isnumeric() else l for l in labels_list]
    return labels_list

def check_same_contents(nums1, nums2):
    for x in set(nums1 + nums2):
        if nums1.count(x) != nums2.count(x):
            return False
    return True

def read_nictjle_format(columns, lines, other_labels, interviewer=False, combine=False):
    rtn = list()
    
    # originally exsit in text.tsv file
    utt_idx  = columns.index('id')
    utt_id_idx  = columns.index('utt_index') if 'utt_index' in columns else 'X'
    speaker_idx = columns.index('speaker')
    speaker_id_idx = columns.index('speaker_index') if 'speaker_index' in columns else 'X'
    
    content_idx = columns.index('pre_token')
    w_cefr_idx  = columns.index('pre_token_cefr')
    label_idx   = columns.index('score')
    pos_idx = columns.index('pos')
    dep_rels_idx = columns.index('dep_rel')

    for line in lines:
        id_idx = line[utt_id_idx] if utt_id_idx != 'X' else 'X'
        speaker_id = line[speaker_id_idx] if speaker_id_idx != 'X' else 'X'
        id      = line[utt_idx]
        spks_list, content = parse_nictjle_content(line[content_idx], interviewer=interviewer, combine=combine, return_speaker_list=True)
        content = [t for t in content if t]
        w_cefr  = [t for t in parse_nictjle_content(line[w_cefr_idx], interviewer=interviewer, combine=combine) if t]
        pos     = [t for t in parse_nictjle_content(line[pos_idx], interviewer=interviewer, combine=combine) if t]
        deprel  = [t for t in parse_nictjle_content(line[dep_rels_idx], interviewer=interviewer, combine=combine) if t]

        label   = line[label_idx]
        speaker = line[speaker_idx]
    
        # other labels
        tokens = line[content_idx]
        grammar_simple = convert_int_if(other_labels.get(id).get('grammar_simple').get('label').split()) if other_labels else [0] * len(tokens.split(' '))
        grammar_simple = ' '.join([str(x) for x in grammar_simple])
        grammar_simple = [t for t in parse_nictjle_content(grammar_simple, interviewer=interviewer, combine=combine) if t]

        grammar_median = combine_tags(tokens, id, get_gra_med_labels(), get_name='grammar_median') if other_labels else [0] * len(tokens.split(' '))
        grammar_median = ' '.join([str(x) for x in grammar_median])
        grammar_median = [t for t in parse_nictjle_content(grammar_median, interviewer=interviewer, combine=combine) if t]
        
        grammar_difficult = combine_tags(tokens, id, get_gra_dif_labels(), get_name='grammar_difficult') if other_labels else [0] * len(tokens.split(' '))
        grammar_difficult = ' '.join([str(x) for x in grammar_difficult])
        grammar_difficult = [t for t in parse_nictjle_content(grammar_difficult, interviewer=interviewer, combine=combine) if t]

        disfluency_simple = convert_int_if(other_labels.get(id).get('disfluency_simple').get('label').split()) if other_labels else [0] * len(tokens.split(' '))
        disfluency_simple = ' '.join([str(x) for x in disfluency_simple])
        disfluency_simple = [t for t in parse_nictjle_content(disfluency_simple, interviewer=interviewer, combine=combine) if t]

        disfluency_median = combine_tags(tokens, id, get_dis_med_labels(), get_name='disfluency_median') if other_labels else [0] * len(tokens.split(' '))
        disfluency_median = ' '.join([str(x) for x in disfluency_median])
        disfluency_median = [t for t in parse_nictjle_content(disfluency_median, interviewer=interviewer, combine=combine) if t]

        filled_pauses_simple = convert_int_if(other_labels.get(id).get('filled_pauses_simple').get('label').split()) if other_labels else [0] * len(tokens.split(' '))
        filled_pauses_simple = ' '.join([str(x) for x in filled_pauses_simple])
        filled_pauses_simple = [t for t in parse_nictjle_content(filled_pauses_simple, interviewer=interviewer, combine=combine) if t]

        filled_pauses_median = combine_tags(tokens, id, get_filled_pauses_median_labels(), get_name='filled_pauses_median') if other_labels else [0] * len(tokens.split(' '))
        filled_pauses_median = ' '.join([str(x) for x in filled_pauses_median])
        filled_pauses_median = [t for t in parse_nictjle_content(filled_pauses_median, interviewer=interviewer, combine=combine) if t]

        laughing = convert_int_if(other_labels.get(id).get('laughter_simple').get('label').split()) if other_labels else [0] * len(tokens.split(' '))
        laughing = ' '.join([str(x) for x in laughing])
        laughing = [t for t in parse_nictjle_content(laughing, interviewer=interviewer, combine=combine) if t]

        silence = convert_int_if(other_labels.get(id).get('silence_simple').get('label').split()) if other_labels else [0] * len(tokens.split(' '))
        silence = ' '.join([str(x) for x in silence])
        silence = [t for t in parse_nictjle_content(silence, interviewer=interviewer, combine=combine) if t]

        rtn.append({'id': id, 'id_idx': id_idx, 'label': label, 'speaker': speaker, 'speaker_id': speaker_id, 'pos': pos, 'deprel': deprel,
                    'text': content, 'w_cefr': w_cefr, 'grammar_simple': grammar_simple, 'grammar_median': grammar_median,
                    'grammar_difficult': grammar_difficult, 'disfluency_simple': disfluency_simple, 'disfluency_median': disfluency_median,
                    'filled_pauses_simple': filled_pauses_simple, 'filled_pauses_median': filled_pauses_median, 'laughing': laughing,
                    'silence': silence, 'spks_list': spks_list})
    return rtn

def combine_into_documents(nictjle_data_list, combine=False, bert=False):
    documents = []
    converts_sents = {}
    converts_labels = {}
    converts_wcefrs = {}
    converts_pos = {}
    converts_deprel = {}
    if combine and bert:
        for e in nictjle_data_list:
            speaker = e.get('speaker')
            sents   = e.get('text')
            w_cefr  = e.get('w_cefr')
            pos     = e.get('pos')
            deprel  = e.get('deprel')
            converts_sents.setdefault(
                speaker,
                []
            ).append(sents)
            converts_wcefrs.setdefault(
                speaker,
                []
            ).append(w_cefr)
            converts_pos.setdefault(
                speaker,
                []
            ).append(pos)
            converts_deprel.setdefault(
                speaker,
                []
            ).append(deprel)
            if speaker not in converts_labels:
                converts_labels[speaker] = {'speaker_id': e.get('speaker_id'), 'label': e.get('label')}

        for speaker, document in converts_sents.items():
            e = {}
            e.update(converts_labels[speaker])
            e['text'] = document
            e['speaker'] = speaker
            e['w_cefr']  = converts_wcefrs[speaker]
            e['pos']     = converts_pos[speaker]
            e['deprel']  = converts_deprel[speaker]
            documents.append(e)
    else:
        for e in nictjle_data_list:
            speaker = e.get('speaker')
            sents   = e.get('text')
            w_cefr  = e.get('w_cefr')
            pos     = e.get('pos')
            deprel  = e.get('deprel')
            converts_sents.setdefault(
                speaker,
                []
            ).append(' '.join(sents))
            converts_wcefrs.setdefault(
                speaker,
                []
            ).append(' '.join(w_cefr))
            converts_pos.setdefault(
                speaker,
                []
            ).append(' '.join(pos))
            converts_deprel.setdefault(
                speaker,
                []
            ).append(' '.join(deprel))
            if speaker not in converts_labels:
                converts_labels[speaker] = {'speaker_id': e.get('speaker_id'), 'label': e.get('label')}

        for speaker, document in converts_sents.items():
            e = {}
            e.update(converts_labels[speaker])
            e['text'] = document
            e['speaker'] = speaker
            e['w_cefr']  = converts_wcefrs[speaker]
            e['pos']     = converts_pos[speaker]
            e['deprel']  = converts_deprel[speaker]
            documents.append(e)
        
    return documents


if __name__ == '__main__':
    
    args = args_init()
    input_text_tsv_file_path = args.input_text_tsv_file_path
    input_labels_file_path = args.input_labels_file_path
    output_dir_path = args.output_dir_path
    sentaspara = args.sentaspara
    interviewer = args.interviewer
    combine = args.combine
    bert = args.bert
    
    if not os.path.exists(output_dir_path): os.makedirs(output_dir_path)

    columns, text_lines = readText(input_text_tsv_file_path, output_col_names=True)
    other_labels = pickle.load(open(input_labels_file_path, 'rb'))
    nictjle_data_list   = read_nictjle_format(columns, text_lines, other_labels,interviewer=interviewer, combine=combine)
    if sentaspara == 'para':
        nictjle_data_list = combine_into_documents(nictjle_data_list, combine=combine, bert=bert)
    
    fname = GetType(input_text_tsv_file_path) + ".{}.label.jsonl".format(
        sentaspara if not interviewer else '{}.{}'.format(sentaspara, 'interviewer') \
        if not combine else '{}.{}'.format(sentaspara, 'combine')
    )
    saveFile = os.path.join(output_dir_path, fname)

    f = open(saveFile, "w")
    for e in nictjle_data_list:
        f.write(json.dumps(e) + "\n")