import csv
from multiprocessing import connection
import os
import pickle
import sys
import torch
import numpy as np
from collections import Counter
from torch.utils.data import TensorDataset

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

class InputFeatures(object):
    """A single set of features for a single InputExample."""

    def __init__(self, input_ids, input_mask, segment_ids=None, pos_tags=None, dep_rels=None,
                grammar_simple=None, grammar_median=None, grammar_difficult=None,
                disfluency_simple=None, disfluency_median=None, laughing=None, silence=None, w_cefr=None,
                spks_list=None
                ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.pos_tags = pos_tags
        self.dep_rels = dep_rels
        self.grammar_simple = grammar_simple
        self.grammar_median = grammar_median
        self.grammar_difficult = grammar_difficult
        self.disfluency_simple = disfluency_simple
        self.disfluency_median = disfluency_median
        self.laughing = laughing
        self.silence = silence
        self.w_cefr  = w_cefr
        self.spks_list = spks_list

def build_bert_features(example, max_seq_length, special_tokens, tokenizer=None, is_pretokenized=False):
    special_tokens_count = 2 # CLS and SEP special tokens
    pos_tag_label_map = {label: i for i, label in enumerate(pos_tag_labels)}
    dep_rel_label_map = {label: i for i, label in enumerate(deprel_labels)}
    grammar_median_label_map = {label: i for i, label in enumerate(ged_median_labels)}
    grammar_difficult_label_map = {label: i for i, label in enumerate(ged_difficult_labels)}
    disfluency_median_label_map = {label: i for i, label in enumerate(disfluency_median_labels)}
    word_cefr_label_map = {label: i for i, label in enumerate(w_cefr_labels)}

    new_token_string = '[NEW]'
    sep_token = tokenizer.sep_token
    cls_token = tokenizer.cls_token
    unk_token = tokenizer.unk_token

    features = []
    sent_boundary = []

    id = example.get('id')
    tokens, pos_tags, dep_rels = [], [], []
    grammar_simple, grammar_median, grammar_difficult = [], [], []
    disfluency_simple, disfluency_median = [], []
    laughing, silence = [], []
    w_cefr = []
    segment_ids = []
    segment_incremental_index = 0

    # split the tokens for preprocessing answer-only part and qa part
    # first is for qa part
    sents_list = example.get('text')
    deprel_list = example.get('deprel')
    pos_list = example.get('pos')
    grammar_simple_list = example.get('grammar_simple')
    grammar_median_list = example.get('grammar_median')
    grammar_difficult_list = example.get('grammar_difficult')
    disfluency_simple_list = example.get('disfluency_simple')
    disfluency_median_list = example.get('disfluency_median')
    laughing_list = example.get('laughing')
    silence_list  = example.get('silence')
    w_cefr_list   = example.get('w_cefr')
    spks_list     = example.get('spks_list')
    
    c_start_sent_token_idx = 0 # start with [CLS] token
    c_end_sent_token_idx = 0

    for (sent_, deprel_, pos_tag_, gra_sim_, gra_med_, gra_dif_, dis_sim_, dis_med_, laugh_t_, silence_t_, w_cefr_t_, spk_i) in \
        zip(
            sents_list, deprel_list, pos_list, grammar_simple_list, grammar_median_list, grammar_difficult_list,
            disfluency_simple_list, disfluency_median_list, laughing_list, silence_list, w_cefr_list, spks_list
        ):

        sent_list_ = sent_.split()
        deprel_list_ = deprel_.split()
        pos_tag_list_ = pos_tag_.split()
        gra_sim_list_ = gra_sim_.split()
        gra_med_list_ = gra_med_.split()
        gra_dif_list_ = gra_dif_.split()
        dis_sim_list_ = dis_sim_.split()
        dis_med_list_ = dis_med_.split()
        laugh_t_list_ = laugh_t_.split()
        silence_t_list_ = silence_t_.split()
        w_cefr_t_list_ = w_cefr_t_.split()
        
        c_sent_token_len = 0

        for i, (word, deprel, pos_tag, gra_sim, gra_med, gra_dif, dis_sim, dis_med, laugh_t, silence_t, w_cefr_t) in \
            enumerate(zip(
                sent_list_, deprel_list_, pos_tag_list_, gra_sim_list_, gra_med_list_,
                gra_dif_list_, dis_sim_list_, dis_med_list_, laugh_t_list_,
                silence_t_list_, w_cefr_t_list_
            )):

            if word == '[UNK]':
                word = unk_token

            if len(tokens) == max_seq_length - special_tokens_count:
                break

            word_pieces = tokenizer.tokenize(word) if tokenizer else [word]
            if (len(word_pieces) > 1) and is_pretokenized:
                word_pieces = [unk_token]

            if word == '_':
                word_pieces = [word] if tokenizer else [word]

            c_sent_token_len = c_sent_token_len + len(word_pieces)

            tokens.extend(word_pieces)
            # don't predict on special tokens for auxiliary objjectives
            if word in special_tokens:
                pos_tags.append(-1)
                dep_rels.append(-1)
                grammar_simple.append(-1)
                grammar_median.append(-1)
                grammar_difficult.append(-1)
                disfluency_simple.append(-1)
                disfluency_median.append(-1)
                laughing.append(-1)
                silence.append(-1)
                segment_ids.append(0)
            else:
                pos_tags.append(pos_tag_label_map.get(pos_tag, 0))
                dep_rels.append(dep_rel_label_map.get(deprel, 0))
                grammar_simple.append(int(gra_sim))
                grammar_median.append(grammar_median_label_map.get(gra_med, 0))
                grammar_difficult.append(grammar_difficult_label_map.get(gra_dif, 0))
                disfluency_simple.append(int(dis_sim))
                disfluency_median.append(disfluency_median_label_map.get(dis_med, 0))
                laughing.append(int(laugh_t))
                silence.append(int(silence_t))
                w_cefr.append(word_cefr_label_map.get(w_cefr_t, 0))
                segment_ids.extend([spk_i] * len(word_pieces))
                token_padding = [-1] * (len(word_pieces) - 1)
                pos_tags.extend(token_padding)
                dep_rels.extend(token_padding)
                grammar_simple.extend(token_padding)
                grammar_median.extend(token_padding)
                grammar_difficult.extend(token_padding)
                disfluency_simple.extend(token_padding)
                disfluency_median.extend(token_padding)
                laughing.extend(token_padding)
                silence.extend(token_padding)
                w_cefr.extend(token_padding)

        c_start_sent_token_idx = c_start_sent_token_idx + 1 # start index is for [CLS] or [SEP]
        c_end_sent_token_idx = c_start_sent_token_idx + c_sent_token_len - 1
        sent_boundary.append([c_start_sent_token_idx, c_end_sent_token_idx])
        c_start_sent_token_idx = c_end_sent_token_idx + 1

        # add sep token
        tokens = tokens + [sep_token]
        pos_tags = pos_tags + [-1]
        dep_rels = dep_rels + [-1]
        grammar_simple = grammar_simple + [-1]
        grammar_median = grammar_median + [-1]
        grammar_difficult = grammar_difficult + [-1]
        disfluency_simple = disfluency_simple + [-1]
        disfluency_median = disfluency_median + [-1]
        laughing = laughing + [-1]
        silence  = silence + [-1]
        w_cefr   = w_cefr + [-1]
        segment_ids = segment_ids + [spk_i]

    if len(tokens) > max_seq_length - special_tokens_count:
        tokens = tokens[:(max_seq_length - special_tokens_count)]
        pos_tags = pos_tags[:(max_seq_length - special_tokens_count)]
        dep_rels = dep_rels[:(max_seq_length - special_tokens_count)]
        grammar_simple = grammar_simple[:(max_seq_length - special_tokens_count)]
        grammar_median = grammar_median[:(max_seq_length - special_tokens_count)]
        grammar_difficult = grammar_difficult[:(max_seq_length - special_tokens_count)]
        disfluency_simple = disfluency_simple[:(max_seq_length - special_tokens_count)]
        disfluency_median = disfluency_median[:(max_seq_length - special_tokens_count)]
        laughing = laughing[:(max_seq_length - special_tokens_count)]
        silence  = silence[:(max_seq_length - special_tokens_count)]
        w_cefr   = w_cefr[:(max_seq_length - special_tokens_count)]
        segment_ids = segment_ids[:(max_seq_length - special_tokens_count)]

    # debug: if last token after removing is also [SEP]
    if tokens[-1] == sep_token:
        # remove final [SEP] token
        tokens = tokens[:-1]
        pos_tags = pos_tags[:-1]
        dep_rels = dep_rels[:-1]
        grammar_simple = grammar_simple[:-1]
        grammar_median = grammar_median[:-1]
        grammar_difficult = grammar_difficult[:-1]
        disfluency_simple = disfluency_simple[:-1]
        disfluency_median = disfluency_median[:-1]
        laughing = laughing[:-1]
        silence  = silence[:-1]
        w_cefr   = w_cefr[:-1]
        segment_ids = segment_ids[:-1]
        
    tokens = [cls_token] + tokens + [sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    pos_tags = [-1] + pos_tags + [-1]
    dep_rels = [-1] + dep_rels + [-1]
    grammar_simple = [-1] + grammar_simple + [-1]
    grammar_median = [-1] + grammar_median + [-1]
    grammar_difficult = [-1] + grammar_difficult + [-1]
    disfluency_simple = [-1] + disfluency_simple + [-1]
    disfluency_median = [-1] + disfluency_median + [-1]
    laughing = [-1] + laughing + [-1]
    silence  = [-1] + silence  + [-1]
    w_cefr   = [-1] + w_cefr   + [-1]
    segment_ids = [segment_ids[0]] + segment_ids + [segment_ids[-1]]

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)
    # Zero-pad up to the sequence length.
    example_padding = max_seq_length - len(input_ids)
    tokens = tokens + ([tokenizer.pad_token] * example_padding)
    input_ids = input_ids + ([tokenizer.pad_token_id] * example_padding)
    input_mask = input_mask + ([0] * example_padding)
    pos_tags = pos_tags + ([-1] * example_padding)
    dep_rels = dep_rels + ([-1] * example_padding)
    grammar_simple = grammar_simple + ([-1] * example_padding)
    grammar_median = grammar_median + ([-1] * example_padding)
    grammar_difficult = grammar_difficult + ([-1] * example_padding)
    disfluency_simple = disfluency_simple + ([-1] * example_padding)
    disfluency_median = disfluency_median + ([-1] * example_padding)
    laughing = laughing + ([-1] * example_padding)
    silence  = silence  + ([-1] * example_padding)
    w_cefr   = w_cefr   + ([-1] * example_padding)
    segment_ids = segment_ids + ([0] * example_padding)

    assert len(tokens) == max_seq_length, len(tokens)
    assert len(input_ids) == max_seq_length, len(input_ids)
    assert len(input_mask) == max_seq_length, len(input_mask)
    assert len(pos_tags) == max_seq_length, len(pos_tags)
    assert len(dep_rels) == max_seq_length, len(dep_rels)
    assert len(grammar_simple) == max_seq_length, len(grammar_simple)
    assert len(grammar_median) == max_seq_length, len(grammar_median)
    assert len(grammar_difficult) == max_seq_length, len(grammar_difficult)
    assert len(disfluency_simple) == max_seq_length, len(disfluency_simple)
    assert len(disfluency_median) == max_seq_length, len(disfluency_median)
    assert len(laughing) == max_seq_length, len(laughing)
    assert len(silence)  == max_seq_length, len(silence)
    assert len(w_cefr)   == max_seq_length, len(w_cefr)
    assert len(segment_ids) == max_seq_length, len(segment_ids)

    # collect ans inputs
    features.append(
            InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            pos_tags=pos_tags,
                            dep_rels=dep_rels,
                            grammar_simple=grammar_simple,
                            grammar_median=grammar_median,
                            grammar_difficult=grammar_difficult,
                            disfluency_simple=disfluency_simple,
                            disfluency_median=disfluency_median,
                            laughing=laughing,
                            silence=silence,
                            w_cefr=w_cefr,
                            spks_list=spks_list
                            ))

    return tokens, features, sent_boundary
