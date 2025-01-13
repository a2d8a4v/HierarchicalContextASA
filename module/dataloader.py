#!/usr/bin/python
# -*- coding: utf-8 -*-

# __author__="Danqing Wang"

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains code to read the train/eval/test data from file and process it, and read the vocab data from file and process it"""

import re
import os
from nltk.corpus import stopwords

import glob
import copy
import math
import random
import time
import json
import pickle
import nltk
import collections
import tokenizations
from collections import Counter
from itertools import combinations
import numpy as np
from random import shuffle

import torch
import torch.utils.data
import torch.nn.functional as F

from tools.logger import *
from tools.utils import pikleOpen, pickleStore, POS2INT, DEP2INT, GED2INT
from collections import Counter, defaultdict
from module.bert_data import build_bert_features

import dgl
from dgl.data.utils import save_graphs, load_graphs

FILTERWORD = stopwords.words('english')
punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\'\'', '\'', '`', '``',
                '-', '--', '|', '\/']
FILTERWORD.extend(punctuations)

CEFR2INT = {
    'X' : 7,
    'A1': 1,
    'A2': 2,
    'B1': 3,
    'B2': 4,
    'C1': 5,
    'C2': 6,
    '[PAD]': 0
}

######################################### Example #########################################

class ExampleHierSet(torch.utils.data.Dataset):
    """ Constructor: Dataset of example(object) for single document summarization"""

    def __init__(self, data_path, interviewer_data_path, relation_database_path, relation_data_path, abreast_data_path, bert_data_path, oie_data_path, vocab, doc_max_timesteps, sent_max_len, filter_word_path, w2s_path, pmi_window_width, tokenizer, hps):
        """ Initializes the ExampleSet with the path of data
        
        :param data_path: string; the path of data
        :param interviewer_data_path: string; the path of data from interviewer
        :param bert_data_path: string; the path of data from both interviewer and interviewee
        :param vocab: object;
        :param doc_max_timesteps: int; the maximum sentence number of a document, each example should pad sentences to this length
        :param sent_max_len: int; the maximum token number of a sentence, each sentence should pad tokens to this length
        :param filter_word_path: str; file path, the file must contain one word for each line and the tfidf value must go from low to high (the format can refer to script/lowTFIDFWords.py) 
        :param w2s_path: str; file path, each line in the file contain a json format data (which can refer to the format can refer to script/calw2sTFIDF.py)
        :param pmi_window_width: widow with when calculatin NPMI
        :param tokenizer: WordPeice Tokenizer
        :param hps: parameters
        """

        self.hps = hps
        self.vocab = vocab
        self.sent_max_len = sent_max_len
        self.doc_max_timesteps = doc_max_timesteps
        # relation_database
        relation_index_dict = readsingleJson(relation_database_path)
        index_relation_dict = {v: k for k, v in relation_index_dict.items()}
        self.relation_index_dict = relation_index_dict
        self.index_relation_dict = index_relation_dict
        self.abreast_sents = readDialogueSents(abreast_data_path)
        self.relations_sents = readDialogueRelations(relation_data_path, relation_index_dict)
        self.oie_sents = readJson(oie_data_path)

        logger.info("[INFO] Start reading %s", self.__class__.__name__)
        start = time.time()
        self.example_list = readJson(data_path)
        if bert_data_path:
            self.bert_example_list = readJson(bert_data_path)
            # map two sequences tokenized from differenct source of tokenizer with the same string input
            self.bert_example_list = readJson(bert_data_path)
        logger.info("[INFO] Finish reading %s. Total time is %f, Total size is %d", self.__class__.__name__,
                    time.time() - start, len(self.example_list))
        if interviewer_data_path is not None:
            self.interviewer_example_list = readJson(interviewer_data_path)
            logger.info("[INFO] Finish reading %s. Total time is %f, Total size is %d", self.__class__.__name__,
                        time.time() - start, len(self.interviewer_example_list))
        self.size = len(self.example_list)
        self.data_set = data_path.split('/')[-1].split('.')[0]
        self.doc_pmi_dir = os.path.join(self.hps.cache_dir, 'pmi')
        if not os.path.exists(self.doc_pmi_dir):
            os.makedirs(self.doc_pmi_dir)
        self.G_dir = os.path.join(self.hps.cache_dir, 'G')
        self.BERT_dir = os.path.join(self.hps.cache_dir, 'BERT')
        if not os.path.exists(self.G_dir):
            os.makedirs(self.G_dir)
        if not os.path.exists(self.BERT_dir):
            os.makedirs(self.BERT_dir)
            
        logger.info("[INFO] Loading filter word File %s", filter_word_path)
        tfidf_w = readText(filter_word_path)
        self.filterwords = FILTERWORD
        self.filterids = [vocab.word2id(w.lower()) for w in FILTERWORD]
        self.filterids.append(vocab.word2id("[PAD]"))   # keep "[UNK]" but remove "[PAD]"
        lowtfidf_num = 0
        pattern = r"^[0-9]+$"
        for w in tfidf_w:
            if vocab.word2id(w) != vocab.word2id('[UNK]'):
                self.filterwords.append(w)
                self.filterids.append(vocab.word2id(w))
                # if re.search(pattern, w) == None:  # if w is a number, it will not increase the lowtfidf_num
                    # lowtfidf_num += 1
                lowtfidf_num += 1
            if lowtfidf_num > 5000:
                break

        logger.info("[INFO] Loading word2sent TFIDF file from %s!" % w2s_path)
        self.w2s_tfidf = readJson(w2s_path)
        
        self.pmi_window_width = pmi_window_width
        if pmi_window_width > -1:
            logger.info("[INFO] Use N-PMI!")
        
        self.tokenizer = tokenizer

    def get_example(self, index):
        e = self.example_list[index]
        e["summary"] = e.setdefault("summary", [])
        example = Example(e["text"], e["summary"], self.vocab, self.sent_max_len, e["label"], w_cefr=e["w_cefr"],
                          sentaspara=self.hps.sentaspara, speaker_id=None if not self.hps.eval_speaker_wise else e["speaker_id"],
                            pos=e["pos"], dep=e["deprel"], ged=e["grammar_difficult"])
        return example

    def get_bert_example(self, index):
        e = self.bert_example_list[index]
        # TODO
        is_pretokenized = False if self.hps.pred_method in ['test_wdc', 'test_wdc2', 'test_wdc3', 'acg', 'hsag'] else True
        tokens, features, sb = build_bert_features(e, self.hps.bert_config.max_position_embeddings - 2, [], tokenizer=self.tokenizer, is_pretokenized=is_pretokenized)
        
        # mapping two sequences tokenized from different tokenizers but with the same text input
        ## remove pad_token from inputs
        tokens = [t for t in tokens if t != self.tokenizer.pad_token][1:] # remove cls_token
        assert tokens[-1] == self.tokenizer.sep_token
        ## collapse original sequence input into sentence-wise inputs
        clsp_sents = []
        sent = []
        for i, t in enumerate(tokens): # collapse
            if t != self.tokenizer.sep_token:
                sent.append(t)
            else:
                clsp_sents.append(sent)
                sent = []
            if i == len(tokens) - 1:
                if tokens[-1] != self.tokenizer.sep_token:
                    clsp_sents.append(sent)
                del sent

        oie_data = self.oie_sents[index][e.get('id')]

        """
            line_content.setdefault(
                uttid,
                {}
            ).setdefault(
                line_id,
                {'oie': oie_lines, 'tag': tag_lines, 'snt': sent_lines}
            )
        """
        assert oie_data is not None

        ## mark the indexes of oie tokens but mapped to the original tokens from this work
        oie_alng_bert_oieseq = {}

        for l_i, s_o in enumerate(clsp_sents): # sentence-wise

            # index saved into string format in jsonl 
            l_i = str(l_i)

            # oie may not exist
            s_c = oie_data[l_i]['snt'][0]
            s_c = [self.tokenizer.unk_token if t == '_' else t for t in s_c] # FIX: unk token

            ## align two sentence which are from the same input but conveyed to different tokenizers
            res_bert2s_c, s_c2res_bert = tokenizations.get_alignments(s_o, s_c)

            oie_alng_bert_oieseq[l_i] = {'res_bert2s_c': res_bert2s_c, 's_c2res_bert': s_c2res_bert}

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_pos_tags = torch.tensor([f.pos_tags for f in features], dtype=torch.long)
        all_dep_rels = torch.tensor([f.dep_rels for f in features], dtype=torch.long)
        all_grammar_simple = torch.tensor([f.grammar_simple for f in features], dtype=torch.long)
        all_grammar_median = torch.tensor([f.grammar_median for f in features], dtype=torch.long)
        all_grammar_difficult = torch.tensor([f.grammar_difficult for f in features], dtype=torch.long)
        all_disfluency_simple = torch.tensor([f.disfluency_simple for f in features], dtype=torch.long)
        all_disfluency_median = torch.tensor([f.disfluency_median for f in features], dtype=torch.long)
        all_laughing = torch.tensor([f.laughing for f in features], dtype=torch.long)
        all_silence  = torch.tensor([f.silence  for f in features], dtype=torch.long)
        all_w_cefr   = torch.tensor([f.w_cefr   for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_spks_list = torch.tensor([f.spks_list for f in features], dtype=torch.long)

        r = {
            'input_ids': all_input_ids,
            'attention_mask': all_input_mask,
            'token_type_ids': all_segment_ids,
            'pos_tokens_ids': all_pos_tags,
            'deprel_tokens_ids': all_dep_rels,
            'grammar_simple_tokens_ids': all_grammar_simple,
            'grammar_median_tokens_ids': all_grammar_median,
            'grammar_difficult_tokens_ids': all_grammar_difficult,
            'disfluency_simple_tokens_ids': all_disfluency_simple,
            'disfluency_median_tokens_ids': all_disfluency_median,
            'laughing_tokens_ids': all_laughing,
            'silence_tokens_ids': all_silence,
            'w_cefr_tokens_ids': all_w_cefr,
            'spks_list': all_spks_list
        }

        return r, sb, oie_alng_bert_oieseq

    def get_interviewer_example(self, index):
        e = self.interviewer_example_list[index]
        e["summary"] = e.setdefault("summary", [])
        example = Example(e["text"], e["summary"], self.vocab, self.sent_max_len, e["label"], w_cefr=e["w_cefr"], sentaspara=self.hps.sentaspara, speaker_id=None if not self.hps.eval_speaker_wise else e["speaker_id"])
        return example

    def pad_label_m(self, label_matrix):
        label_m = label_matrix[:self.doc_max_timesteps, :self.doc_max_timesteps]
        # N, m = label_m.shape
        # if m < self.doc_max_timesteps:
        #     pad_m = np.zeros((N, self.doc_max_timesteps - m))
        #     return np.hstack([label_m, pad_m])
        return label_m

    def AddWordNode(self, G, inputid):
        wid2nid = {}
        nid2wid = {}
        nid = 0
        for sentid in inputid:
            for wid in sentid:
                if wid not in self.filterids and wid not in wid2nid.keys():
                    wid2nid[wid] = nid
                    nid2wid[nid] = wid
                    nid += 1

        w_nodes = len(nid2wid)

        G.add_nodes(w_nodes)
        G.set_n_initializer(dgl.init.zero_initializer)
        G.ndata["unit"] = torch.zeros(w_nodes)
        G.ndata["id"] = torch.LongTensor(list(nid2wid.values()))
        G.ndata["dtype"] = torch.zeros(w_nodes)

        return wid2nid, nid2wid

    def CreateGraphC2W(self, input_pad, cefr_input_pad, w2s_w):
        G = dgl.DGLGraph()
        wid2nid, nid2wid = self.AddWordNode(G, input_pad)
        w_nodes = len(nid2wid)
        N = 7 # A1, A2, B1, B2, C1, C2, a trash can
        M = len(input_pad)
        
        G.add_nodes(N)
        G.ndata["unit"][w_nodes:] = torch.ones(N)
        G.ndata["dtype"][w_nodes:] = torch.ones(N)
        cefrid2nid = [i + w_nodes for i in range(N)]
        
        G.add_nodes(M)
        G.ndata["unit"][w_nodes+N:] = torch.full((1, M), 2.).squeeze(0)
        G.ndata["dtype"][w_nodes+N:] = torch.full((1, M), 2.).squeeze(0)
        sentid2nid = [i + w_nodes + N for i in range(M)]
        
        for i in range(M):
            w_l = input_pad[i]
            c_l = cefr_input_pad[i]
            sent_tfw = w2s_w[str(i)]
            for cid, wid in zip(c_l, w_l):
                if wid in wid2nid.keys() and self.vocab.id2word(wid) in sent_tfw.keys():
                    n_cid = cid + w_nodes - 1 # index
                    relation = [1]
                    G.add_edges(wid2nid[wid], n_cid,
                                data={"tffrac": torch.LongTensor(relation), "dtype": torch.Tensor([0])})
                    G.add_edges(n_cid, wid2nid[wid],
                                data={"tffrac": torch.LongTensor(relation), "dtype": torch.Tensor([0])})
                    
        G.nodes[sentid2nid].data["label"] = torch.LongTensor(cefr_input_pad)
        return G

    def CreateGraph(self, input_pad, label, w2s_w, w2w_pmi_info):
        """ Create a graph for each document
        
        :param input_pad: list(list); [sentnum, wordnum]
        :param label: list(list); [sentnum, sentnum]
        :param w2s_w: dict(dict) {str: {str: float}}; for each sentence and each word, the tfidf between them
        :param w2w_pmi_info: dict(dict) {str: {str: int}}; for each word and each word, the n-pmi between them
        :return: G: dgl.DGLGraph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
            edge:
                word2sent, sent2word:  tffrac=int, dtype=0
                word2word:             tffrac=int, dtype=1
        """
        G = dgl.DGLGraph()
        wid2nid, nid2wid = self.AddWordNode(G, input_pad)
        w_nodes = len(nid2wid)

        N = len(input_pad)
        G.add_nodes(N)
        G.ndata["unit"][w_nodes:] = torch.ones(N)
        G.ndata["dtype"][w_nodes:] = torch.ones(N)
        sentid2nid = [i + w_nodes for i in range(N)]
        
        # global node: 2
        M = 1
        G.add_nodes(M)
        G.ndata["unit"][w_nodes+N:] = torch.full((1, M), 2.).squeeze(0)
        G.ndata["dtype"][w_nodes+N:] = torch.full((1, M), 2.).squeeze(0)
        
        # global nodes and sentence ndoes
        for i in sentid2nid:
            G.add_edges(w_nodes+N+M-1, i, data={"dtype": torch.Tensor([2])}) # 2 (global)
            G.add_edges(i, w_nodes+N+M-1, data={"dtype": torch.Tensor([2])}) # 2 (global)
            
        # global nodes and word nodes
        for i in range(w_nodes):
            G.add_edges(w_nodes+N+M-1, i, data={"dtype": torch.Tensor([2])}) # 2 (global)
            G.add_edges(i, w_nodes+N+M-1, data={"dtype": torch.Tensor([2])}) # 2 (global)

        G.set_e_initializer(dgl.init.zero_initializer)
        if self.pmi_window_width > -1:
            max_pmi = w2w_pmi_info.get('max_pmi')
            pmi_mat = w2w_pmi_info.get('pmi')
            for i in range(N):
                c = Counter(input_pad[i])
                sent_nid = sentid2nid[i]
                sent_tfw = w2s_w[str(i)]
                for s_wid in c.keys():
                    if s_wid in wid2nid.keys() and self.vocab.id2word(s_wid) in sent_tfw.keys():
                        for t_wid in c.keys():
                            if t_wid in wid2nid.keys() and self.vocab.id2word(t_wid) in sent_tfw.keys():
                                s2t = pmi_mat[self.vocab.id2word(s_wid)][self.vocab.id2word(t_wid)] / max_pmi
                                t2s = pmi_mat[self.vocab.id2word(t_wid)][self.vocab.id2word(s_wid)] / max_pmi
                                s2t = np.round(s2t * 9)
                                t2s = np.round(t2s * 9)
                                G.add_edges(wid2nid[s_wid], wid2nid[t_wid],
                                            data={"tffrac": torch.LongTensor([s2t]), "dtype": torch.Tensor([1])})
                                G.add_edges(wid2nid[t_wid], wid2nid[s_wid],
                                            data={"tffrac": torch.LongTensor([t2s]), "dtype": torch.Tensor([1])})
        for i in range(N):
            c = Counter(input_pad[i])
            sent_nid = sentid2nid[i]
            sent_tfw = w2s_w[str(i)]
            for wid in c.keys():
                if wid in wid2nid.keys() and self.vocab.id2word(wid) in sent_tfw.keys():
                    tfidf = sent_tfw[self.vocab.id2word(wid)]
                    tfidf_box = np.round(tfidf * 9)  # box = 10
                    G.add_edges(wid2nid[wid], sent_nid,
                                data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
                    G.add_edges(sent_nid, wid2nid[wid],
                                data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
            
            # The two lines can be commented out if you use the code for your own training, since HSG does not use sent2sent edges. 
            # However, if you want to use the released checkpoint directly, please leave them here.
            # Otherwise it may cause some parameter corresponding errors due to the version differences.
            G.add_edges(sent_nid, sentid2nid, data={"dtype": torch.ones(N)})
            G.add_edges(sentid2nid, sent_nid, data={"dtype": torch.ones(N)})
        G.nodes[sentid2nid].data["words"] = torch.LongTensor(input_pad)  # [N, seq_len]
        G.nodes[sentid2nid].data["position"] = torch.arange(1, N + 1).view(-1, 1).long()  # [N, 1]
        G.nodes[sentid2nid].data["label"] = torch.LongTensor(label)  # [N, doc_max]

        return G

    def CreatePOSDEPGEDGraph(self, input_pad, pos_pad, dep_pad, ged_pad):
        G = dgl.DGLGraph()
        wid2nid, nid2wid = self.AddWordNode(G, input_pad)
        w_nodes = len(nid2wid)

        N = len(input_pad)
        G.add_nodes(N)
        G.ndata["unit"][w_nodes:] = torch.ones(N)
        G.ndata["dtype"][w_nodes:] = torch.ones(N)
        sentid2nid = [i + w_nodes for i in range(N)]
        
        G.nodes[sentid2nid].data["pos"] = torch.LongTensor(pos_pad)  # [N, seq_len]
        G.nodes[sentid2nid].data["dep"] = torch.LongTensor(dep_pad)  # [N, seq_len]
        G.nodes[sentid2nid].data["ged"] = torch.LongTensor(ged_pad)  # [N, seq_len]
        G.nodes[sentid2nid].data["position"] = torch.arange(1, N + 1).view(-1, 1).long()  # [N, 1]

        return G

    def CreateItvrGraph(self, input_pad):
        """ Create a graph for each document
        
        :param input_pad: list(list); [sentnum, wordnum]
        :param label: list(list); [sentnum, sentnum]
        :param w2s_w: dict(dict) {str: {str: float}}; for each sentence and each word, the tfidf between them
        :param w2w_pmi_info: dict(dict) {str: {str: int}}; for each word and each word, the n-pmi between them
        :return: G: dgl.graph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
        """
        G = dgl.DGLGraph()
        wid2nid, nid2wid = self.AddWordNode(G, input_pad)
        w_nodes = len(nid2wid)

        N = len(input_pad)
        G.add_nodes(N)
        G.ndata["unit"][w_nodes:] = torch.ones(N)
        G.ndata["dtype"][w_nodes:] = torch.ones(N)
        sentid2nid = [i + w_nodes for i in range(N)]
        
        G.nodes[sentid2nid].data["words"] = torch.LongTensor(input_pad)  # [N, seq_len]
        G.nodes[sentid2nid].data["position"] = torch.arange(1, N + 1).view(-1, 1).long()  # [N, 1]

        return G

    def CreateMeetingGraph(self, input_pad, itvr_input_pad, abreast_sents, relations, sent_speakers):
        G = dgl.DGLGraph()

        # seperate sents and adjust the length of sents embeddings
        ir_count = 0
        ie_count = 0
        r_sid2nid = {}

        for i, sent in enumerate(abreast_sents['context']):
            if sent == 'dummy':
                break
            if sent.split(' , ')[0] == 'IR':
                ir_count += 1
            elif sent.split(' , ')[0] == 'IE':
                ie_count += 1
        input_pad = input_pad[:ie_count]
        itvr_input_pad = itvr_input_pad[:ir_count]
        minus_i = 0
        for i, sent in enumerate(abreast_sents['context']):
            if sent == 'dummy':
                break
            if sent.split(' , ')[0] == 'IE':
                r_sid2nid[i] = i - minus_i
            elif sent.split(' , ')[0] == 'IR':
                r_sid2nid[i] = i + len(input_pad) - minus_i
            if i%2 == 0:
                minus_i += 1
            else:
                continue

        # all sentences / utterances: 1
        N = len(input_pad) + len(itvr_input_pad)
        G.add_nodes(N)
        G.set_n_initializer(dgl.init.zero_initializer)
        G.ndata["unit"] = torch.ones(N)
        G.ndata["dtype"] = torch.ones(N)
        # G.ndata["kind"] = torch.zeros(len(input_pad))
        # G.ndata["kind"][len(input_pad):] = torch.ones(len(itvr_input_pad))
        sentid2nid = [i for i in range(N)]

        # all relation categories: 2
        L = len(self.index_relation_dict)
        G.add_nodes(L)
        G.ndata["unit"][N:] = torch.full((1, L), 2.).squeeze(0)
        G.ndata["dtype"][N:] = torch.full((1, L), 2.).squeeze(0)
        relationid2nid = []
        relationid2nid_dict = {}
        nid2relationid_dict = {}
        for i, t in self.index_relation_dict.items():
            relationid2nid.append(i+N)
            relationid2nid_dict[i] = i+N
            nid2relationid_dict[i+N] = i

        # global node: 0
        M = 1
        G.add_nodes(M)
        G.ndata["unit"][L+N:] = torch.zeros(M)
        G.ndata["dtype"][L+N:] = torch.zeros(M)

        # global nodes and self ndoes
        for i in sentid2nid:
            G.add_edges(N+L+M-1, i, data={"dtype": torch.Tensor([0])}) # 0 (global)
            G.add_edges(i, N+L+M-1, data={"dtype": torch.Tensor([0])}) # 0 (global)
            G.add_edges(i, i, data={"dtype": torch.Tensor([1])}) # 1 (self)
            
        # relation nodes self
        for i in relationid2nid:
            G.add_edges(N+L+M-1, i, data={"dtype": torch.Tensor([0])}) # 0 (global)
            G.add_edges(i, N+L+M-1, data={"dtype": torch.Tensor([0])}) # 0 (global)
            G.add_edges(i, i, data={"dtype": torch.Tensor([1])}) # 1 (self)

        # relation and sent
        for i, relation in enumerate(relations):
            if i == 0:
                continue
            start_sent_id, end_sent_id, dis_relation_idx = list(map(int, relation))
            start_sent_id = start_sent_id - 1
            end_sent_id = end_sent_id - 1
            relation_id = relationid2nid_dict[dis_relation_idx]
            G.add_edges(r_sid2nid[start_sent_id], relationid2nid_dict[dis_relation_idx], data={"dtype": torch.Tensor([2]), "src_unit": torch.Tensor([1]), "des_unit": torch.Tensor([2]), "relation": torch.Tensor([relation_id])}) # default-in-discourse, sentence -> relation
            G.add_edges(relationid2nid_dict[dis_relation_idx], r_sid2nid[end_sent_id], data={"dtype": torch.Tensor([3]), "src_unit": torch.Tensor([2]), "des_unit": torch.Tensor([1]), "relation": torch.Tensor([relation_id])}) # default-out-discourse, reltaion -> sentence
            G.add_edges(relationid2nid_dict[dis_relation_idx], r_sid2nid[start_sent_id], data={"dtype": torch.Tensor([4]), "src_unit": torch.Tensor([2]), "des_unit": torch.Tensor([1]), "relation": torch.Tensor([relation_id])}) # reverse-out-discourse, reltaion -> sentence
            G.add_edges(r_sid2nid[end_sent_id], relationid2nid_dict[dis_relation_idx], data={"dtype": torch.Tensor([5]), "src_unit": torch.Tensor([1]), "des_unit": torch.Tensor([2]), "relation": torch.Tensor([relation_id])}) # reverse-in-discourse, sentence -> relation

        G.nodes[relationid2nid].data['label'] = torch.arange(0, L).view(-1, 1).long()
        G.nodes[sentid2nid].data["words"] = torch.LongTensor(input_pad+itvr_input_pad)  # [N, seq_len]
        G.nodes[sentid2nid].data["position"] = torch.LongTensor(list(r_sid2nid.values())).view(-1, 1)  # [N, 1]

        return G, ie_count, ir_count

    def CreateWordEntityActsGraph(self, index, input_pad, itvr_input_pad):
        
        e = self.bert_example_list[index]
        uttid = e.get('id')
        oie_data = self.oie_sents[index][uttid]
        
        # nodes: subject node, predicate node, object node, triplet (global) node
        # we need to build three dictionaries: dict. for subject, predicate, and object domain
        # we can also use the pre-defined word dictionary from GloVe. It is an easy way rather than constructing another 3 new dic., but it need to be mapped to the oie tokenized tokens
        
        # map oie tokenized sequence to GloVe-based sequence
        dialogue_turns = []

        # accumulate data
        oie_alng_glv_oieseq = {}
        oie_in_idx = {}

        ## merge candidate's responses and intelocuter's responses into dialogue turns
        for can_res_seq, int_res_seq in zip(input_pad, itvr_input_pad):
            dialogue_turns.append(int_res_seq)
            dialogue_turns.append(can_res_seq)

        ## convert word ids back to word token defined in GloVe so as to align with the oie's tokenized sequences
        for line_id, res_glove in enumerate(dialogue_turns):
            line_id = str(line_id)

            pad_id = self.vocab.get_pad_id()
            res_glove = [self.vocab.id2word(wid) for wid in res_glove if wid != pad_id]

            s_c = oie_data[line_id]['snt'][0]
            s_c = [self.vocab.get_unk_token() if t == '_' else t for t in s_c] # FIX: unk token
            res_glove2s_c, s_c2res_glove = tokenizations.get_alignments(res_glove, s_c)

            oie_alng_glv_oieseq[line_id] = {'res_glove2s_c': res_glove2s_c, 's_c2res_glove': s_c2res_glove}
            oie_in_idx[line_id] = oie_data[line_id]['oie_in_idx']

        ## build node: subject node
        G = dgl.DGLGraph()

        # DEBUG: the subject, predicate, and object may not be a single token, such as circustance leads to the obstacle of constructing nodes in graph
        # We still make a sequence of tokens as a node, but the embedding of node needs to collapsed via a average pooling method
        S_bag = set()
        P_bag = set()
        O_bag = set()
        E_bag = {}
        E = 0
        for line_id, l_data in oie_data.items():
            s_c_oies = l_data['oie'] # list of oies
            s_c_oies = [{k:' '.join(v).lower() for k, v in oie.items()} for oie in s_c_oies]
            for io, oie in enumerate(s_c_oies):
                if 'ARG0' in oie:
                    S_bag.add(oie['ARG0'])
                if 'V' in oie:
                    P_bag.add(oie['V'])
                if 'ARG1' in oie:
                    O_bag.add(oie['ARG1'])
                E_bag[E] = ('{}_{}'.format(line_id, io), oie)
                E += 1
        SPO_bag = list(S_bag | P_bag | O_bag) # remove the duplications
        S_bag = list(S_bag)
        P_bag = list(P_bag)
        O_bag = list(O_bag)
        S = len(S_bag)
        P = len(P_bag)
        O = len(O_bag)

        G.add_nodes(S)
        G.set_n_initializer(dgl.init.zero_initializer)

        G.ndata["unit"] = torch.zeros(S)
        G.ndata["dtype"] = torch.zeros(S)
        subject_nid2spo_dict = {i: w for i, w in enumerate(S_bag)}
        subject_spo2nid_dict = {w: i for i, w in enumerate(S_bag)}
        subjectid2nid = [i for i in range(E)]

        G.add_nodes(P)
        G.ndata["unit"][S:] = torch.ones(P)
        G.ndata["dtype"][S:] = torch.ones(P)
        predicate_nid2spo_dict = {i+S: w for i, w in enumerate(P_bag)}
        predicate_spo2nid_dict = {w: i+S for i, w in enumerate(P_bag)}
        predicateid2nid = [i+S for i in range(E)]

        G.add_nodes(O)
        G.ndata["unit"][S+P:] = torch.full((1, O), 2.).squeeze(0)
        G.ndata["dtype"][S+P:] = torch.full((1, O), 2.).squeeze(0)
        object_nid2spo_dict = {i+S+P: w for i, w in enumerate(O_bag)}
        object_spo2nid_dict = {w: i+S+P for i, w in enumerate(O_bag)}
        objectid2nid = [i+S+P for i in range(E)]

        G.add_nodes(E) # this Entity is built for each oie turn, in other words, we need a line_id in dialogue to map back every entity ids.
        G.ndata["unit"][S+P+O:] = torch.full((1, E), 3.).squeeze(0)
        G.ndata["dtype"][S+P+O:] = torch.full((1, E), 3.).squeeze(0)
        entityid2nid = [i+S+P+O for i in range(E)]

        spoe2nid_dict = {}
        for entity_count, (oie_idx, oie) in E_bag.items():

            # full connected graph
            # TODO: if follow S <-> P <-> O metapath?
            s, p, o = oie.get('ARG0', None), oie.get('V', None), oie.get('ARG1', None)
            s_nid, p_nid, o_nid, e_nid = subject_spo2nid_dict.get(s, None), predicate_spo2nid_dict.get(p, None), object_spo2nid_dict.get(o, None), entityid2nid[entity_count]
            if s:
                G.add_edges(s_nid, s_nid, data={"dtype": torch.Tensor([0])}) # S (self)
                G.add_edges(s_nid, e_nid, data={"dtype": torch.Tensor([2])}) # S -> E (global)
            if p:
                G.add_edges(p_nid, p_nid, data={"dtype": torch.Tensor([0])}) # P (self)
                G.add_edges(p_nid, e_nid, data={"dtype": torch.Tensor([2])}) # P -> E (global)
            if o:
                G.add_edges(o_nid, o_nid, data={"dtype": torch.Tensor([0])}) # O (self)
                G.add_edges(o_nid, e_nid, data={"dtype": torch.Tensor([2])}) # O -> E (global)
            G.add_edges(e_nid, e_nid, data={"dtype": torch.Tensor([0])}) # E (self)
            if s and p:
                G.add_edges(s_nid, p_nid, data={"dtype": torch.Tensor([1])}) # S -> P (normal)
                G.add_edges(p_nid, s_nid, data={"dtype": torch.Tensor([1])}) # P -> S (normal)
            if s and o:
                G.add_edges(s_nid, o_nid, data={"dtype": torch.Tensor([1])}) # S -> O (normal)
                G.add_edges(o_nid, s_nid, data={"dtype": torch.Tensor([1])}) # O -> S (normal)
            if p and o:
                G.add_edges(p_nid, o_nid, data={"dtype": torch.Tensor([1])}) # P -> O (normal)
                G.add_edges(o_nid, p_nid, data={"dtype": torch.Tensor([1])}) # O -> P (normal)

            spoe2nid_dict[oie_idx] = {
                'ARG0': s_nid,
                'V': p_nid,
                'ARG1': o_nid,
                'ENTITY': e_nid
            }

            line_id = int(oie_idx.split('_')[0])

            # assign line id for each Subject, Predicate, and Object nodes
            if s:
                G.nodes[s_nid].data["position"] = torch.full((1, 1), line_id).long().squeeze(0)  # [1, 1]
            if p:
                G.nodes[p_nid].data["position"] = torch.full((1, 1), line_id).long().squeeze(0)  # [1, 1]
            if o:
                G.nodes[o_nid].data["position"] = torch.full((1, 1), line_id).long().squeeze(0)  # [1, 1]
            G.nodes[e_nid].data["position"] = torch.full((1, 1), line_id).long().squeeze(0)  # [1, 1]

        return G, oie_alng_glv_oieseq, oie_in_idx, spoe2nid_dict


    def CreateSentWordEntityActsGraph(self, index, input_pad, itvr_input_pad, abreast_sents):
        
        e = self.bert_example_list[index]
        uttid = e.get('id')
        oie_data = self.oie_sents[index][uttid]

        # seperate sents and adjust the length of sents embeddings
        ir_count = 0
        ie_count = 0
        r_sid2nid = {}

        for i, sent in enumerate(abreast_sents['context']):
            if sent == 'dummy':
                break
            if sent.split(' , ')[0] == 'IR':
                ir_count += 1
            elif sent.split(' , ')[0] == 'IE':
                ie_count += 1
        input_pad = input_pad[:ie_count]
        itvr_input_pad = itvr_input_pad[:ir_count]
        minus_i = 0
        for i, sent in enumerate(abreast_sents['context']):
            if sent == 'dummy':
                break
            if sent.split(' , ')[0] == 'IE':
                r_sid2nid[i] = i - minus_i
            elif sent.split(' , ')[0] == 'IR':
                r_sid2nid[i] = i + len(input_pad) - minus_i
            if i%2 == 0:
                minus_i += 1
            else:
                continue

        ## build node: subject node
        G = dgl.DGLGraph()

        E_bag = {}
        E = 0
        for line_id, l_data in oie_data.items():
            s_c_oies = l_data['oie'] # list of oies
            s_c_oies = [{k:' '.join(v).lower() for k, v in oie.items()} for oie in s_c_oies]
            for io, oie in enumerate(s_c_oies):
                E_bag[E] = '{}_{}'.format(line_id, io)
                E += 1

        N = len(input_pad) + len(itvr_input_pad)
        G.set_n_initializer(dgl.init.zero_initializer)

        G.add_nodes(N)
        G.ndata["unit"] = torch.zeros(N)
        G.ndata["dtype"] = torch.zeros(N)
        # subjectid2nid = [i for i in range(N)]

        G.add_nodes(E)
        G.ndata["unit"][N:] = torch.ones(E)
        G.ndata["dtype"][N:] = torch.ones(E)
        entityid2nid = [i+N for i in range(E)]

        A = N
        G.add_nodes(A)
        G.ndata["unit"][N+E:] = torch.full((1, A), 2.).squeeze(0)
        G.ndata["dtype"][N+E:] = torch.full((1, A), 2.).squeeze(0)
        globalid2nid = [i+N+E for i in range(A)]

        se2nid_dict = {}
        for entity_count, oie_idx in E_bag.items():

            line_id = int(oie_idx.split('_')[0])
            # s_nid, e_nid = subjectid2nid[line_id], entityid2nid[entity_count]
            s_nid, e_nid, a_nid = r_sid2nid[line_id], entityid2nid[entity_count], globalid2nid[line_id]
            G.add_edges(e_nid, s_nid, data={"dtype": torch.Tensor([0])}) # E -> S 
            G.add_edges(s_nid, e_nid, data={"dtype": torch.Tensor([1])}) # S -> E
            G.add_edges(s_nid, a_nid, data={"dtype": torch.Tensor([2])}) # S -> G (global)
            G.add_edges(e_nid, a_nid, data={"dtype": torch.Tensor([2])}) # E -> G (global)

            se2nid_dict[oie_idx] = {
                'ENTITY': e_nid
            }

            # assign line id for each Subject, Predicate, and Object nodes
            G.nodes[s_nid].data["position"] = torch.full((1, 1), line_id).long().squeeze(0)  # [1, 1]
            G.nodes[e_nid].data["position"] = torch.full((1, 1), line_id).long().squeeze(0)  # [1, 1]

        return G, se2nid_dict


    def __getitem__(self, index):
        """
        :param index: int; the index of the example
        :return 
            G: graph for the example
            index: int; the index of the example in the dataset
        """
        item = self.get_example(index)
        itvr_item = self.get_interviewer_example(index)
        input_pad = item.enc_sent_input_pad[:self.doc_max_timesteps]
        cefr_pad  = item.enc_sent_wcefr_pad[:self.doc_max_timesteps]
        pos_pad = item.enc_sent_pos_pad[:self.doc_max_timesteps]
        dep_pad = item.enc_sent_dep_pad[:self.doc_max_timesteps]
        ged_pad = item.enc_sent_ged_pad[:self.doc_max_timesteps]
        itvr_input_pad = itvr_item.enc_sent_input_pad[:self.doc_max_timesteps]
        assert len(itvr_input_pad) == len(input_pad), 'Problems in interviewers inputs, {} and {}'.format(len(itvr_input_pad), len(input_pad))
        label = self.pad_label_m(item.label_matrix)
        w2s_w = self.w2s_tfidf[index]

        # PMI information
        doc_pmi_file_name = '.'.join(list(filter(None, 
            [self.data_set, str(index), ('pmi{}'.format(self.pmi_window_width) if self.pmi_window_width > -1 else ''), ('itvr' if self.hps.interviewer else None),
            ('cefrbd' if self.hps.cefr_word and (self.hps.cefr_info == 'embed_init') else ''), ('fpbd' if self.hps.filled_pauses_word and (self.hps.filled_pauses_info == 'embed_init') else '')
            ]
            ))
        )
        doc_pmi_path = os.path.join(self.doc_pmi_dir, doc_pmi_file_name)
        if os.path.exists(doc_pmi_path):
            w2w_pmi_info = pikleOpen(doc_pmi_path)
        else:
            w2w_pmi_info = self._calculate_pmi(item.original_article_sents, pmi_window_width=self.pmi_window_width)
            pickleStore(w2w_pmi_info, doc_pmi_path)

        # word <-> sent graph, candidate
        G_file_name = '.'.join(list(filter(None,
            [self.data_set, str(index), ('pmi{}'.format(self.pmi_window_width) if self.pmi_window_width > -1 else ''), ('itvr' if self.hps.interviewer else None),
            ('cefrbd' if self.hps.cefr_word and (self.hps.cefr_info == 'embed_init') else ''), ('fpbd' if self.hps.filled_pauses_word and (self.hps.filled_pauses_info == 'embed_init') else '')
            ]
            ))
        )
        G_path = os.path.join(self.G_dir, G_file_name)
        if os.path.exists(G_path):
            G = pikleOpen(G_path)
        else:
            G = self.CreateGraph(input_pad, label, w2s_w, w2w_pmi_info)
            pickleStore(G, G_path)

        ### CEFR <-> word graph, candidate
        G_c = None
        if self.hps.cefr_info == 'graph_init':            
            G_c_file_name = '.'.join(list(filter(None,
                [self.data_set, str(index), 'G_c', self.hps.sentaspara, 
                ('cefrgh' if self.hps.cefr_word and (self.hps.cefr_info == 'graph_init') else '')]
                ))
            )
            G_c_path = os.path.join(self.G_dir, G_c_file_name)
            if os.path.exists(G_c_path):
                G_c = pikleOpen(G_c_path)
            else:
                G_c = self.CreateGraphC2W(input_pad, cefr_pad, w2s_w)
                pickleStore(G_c, G_c_path)

        # word <-> sent grpah, interlocutor
        itvr_G = None
        if self.hps.interviewer:
            itvr_G_file_name = '.'.join(list(filter(None,
                [self.data_set, str(index), 'itvr_G', self.hps.sentaspara]
                ))
            )
            itvr_G_path = os.path.join(self.G_dir, itvr_G_file_name)

            if os.path.exists(itvr_G_path):
                itvr_G = pikleOpen(itvr_G_path)
            else:
                itvr_G = self.CreateItvrGraph(itvr_input_pad)
                pickleStore(itvr_G, itvr_G_path)

        # pos, dep, ged
        G_PDG = None
        if self.hps.language_use:
            G_PDG_file_name = '.'.join(list(filter(None,
                [self.data_set, str(index), 'G_PDG', self.hps.sentaspara]
                ))
            )
            G_PDG_path = os.path.join(self.G_dir, G_PDG_file_name)

            if os.path.exists(G_PDG_path):
                G_PDG = pikleOpen(G_PDG_path)
            else:
                G_PDG = self.CreatePOSDEPGEDGraph(input_pad, pos_pad, dep_pad, ged_pad)
                pickleStore(G_PDG, G_PDG_path)

        # Transformer-based LM intializating representation
        f = None
        sb = None
        oie_alng_bert_oieseq = None
        if self.tokenizer is not None:
            bert_file_name = '.'.join(list(filter(None,
                [self.data_set, str(index), 'BERT', self.hps.sentaspara]
                ))
            )
            bert_path = os.path.join(self.BERT_dir, bert_file_name)
            if os.path.exists(bert_path):
                (f, sb, oie_alng_bert_oieseq) = pikleOpen(bert_path)
            else:
                f, sb, oie_alng_bert_oieseq = self.get_bert_example(index)
                pickleStore((f, sb, oie_alng_bert_oieseq), bert_path)

        # Dialogue Acts with Open Information Extraction: Subject, Predicate, Object
        DA_G = None
        oie_alng_glv_oieseq = None
        oie_in_idx = None
        spoe2nid_dict = None
        if self.hps.pred_method in ['test_wdc', 'test_wdc2', 'test_wdc3', 'acg', 'hsag']:
            DA_G_file_name = '.'.join(list(filter(None,
                [self.data_set, str(index), 'DA_G', self.hps.sentaspara]
                ))
            )
            DA_G_path = os.path.join(self.G_dir, DA_G_file_name)

            if os.path.exists(DA_G_path):
                (DA_G, oie_alng_glv_oieseq, oie_in_idx, spoe2nid_dict) = pikleOpen(DA_G_path)
            else:
                DA_G, oie_alng_glv_oieseq, oie_in_idx, spoe2nid_dict = self.CreateWordEntityActsGraph(index, input_pad, itvr_input_pad)
                pickleStore((DA_G, oie_alng_glv_oieseq, oie_in_idx, spoe2nid_dict), DA_G_path)

        DSA_G = None
        se2nid_dict = None
        if self.hps.pred_method in ['test_wdc3', 'acg', 'hsag']:
            DSA_G_file_name = '.'.join(list(filter(None,
                [self.data_set, str(index), 'DSA_G', self.hps.sentaspara]
                ))
            )
            DSA_G_path = os.path.join(self.G_dir, DSA_G_file_name)
            
            if os.path.exists(DSA_G_path):
                (DSA_G, se2nid_dict) = pikleOpen(DSA_G_path)
            else:
                DSA_G, se2nid_dict = self.CreateSentWordEntityActsGraph(index, input_pad, itvr_input_pad, self.abreast_sents[index])
                pickleStore((DSA_G, se2nid_dict), DSA_G_path)

        # structured discorse relation <-> responeses graph, candidate and intelocutor
        M_G_file_name = '.'.join(list(filter(None,
            [self.data_set, str(index), 'M_G']
            ))
        )
        M_G_path = os.path.join(self.G_dir, M_G_file_name)
        if os.path.exists(M_G_path):
            (M_G, ie_count, ir_count) = pikleOpen(M_G_path)
        else:
            M_G, ie_count, ir_count = self.CreateMeetingGraph(input_pad, itvr_input_pad, self.abreast_sents[index], self.relations_sents[index], None)
            pickleStore((M_G, ie_count, ir_count), M_G_path)

        label = torch.argmax(torch.LongTensor(label)[0]).reshape(-1).to(torch.float32)
        speaker_id = item.speaker_id
        utt_id = self.example_list[index].get('id')

        r = {
            'label': label,
            'speaker_id': speaker_id,
            'utt_id': utt_id,
            'G': G,
            'G_c': G_c,
            'itvr_G': itvr_G,
            'M_G': M_G,
            'DA_G': DA_G,
            'DSA_G': DSA_G,
            'G_PDG': G_PDG,
            'ie_count': ie_count,
            'ir_count': ir_count,
            'bert_feature': f,
            'bert_sent_boundary': sb,
            'oie_in_idx': oie_in_idx,
            'spoe2nid_dict': spoe2nid_dict,
            'oie_alng_glv_oieseq': oie_alng_glv_oieseq,
            'oie_alng_bert_oieseq': oie_alng_bert_oieseq,
        }

        if self.hps.pred_method not in ['test_wdc', 'test_wdc2', 'test_wdc3', 'acg', 'hsag']:
            del r["DA_G"]
            
        if self.hps.pred_method not in ['test_wdc3', 'acg', 'hsag']:
            del r["DSA_G"]
            
        if not self.hps.language_use:
            del r["G_PDG"]

        if self.hps.cefr_info != 'graph_init':
            del r["G_c"]


        # oie_alng_glv_bert_oieseq (position_idx -> position_idx but seq under another tokenizer) -> 
        ## -> for glove: get position_id of glove in each sequence -> token_id in glove by input_pad -> glove embed
        ## -> for bert: get position_id of bert in each sequence -> token_id in wordpiece relying on bert_sent_boundary -> bert embed

        return r

    def _iter_ngrams(self, words, n):
        """Iterate over all word n-grams in a list."""
        if len(words) < n:
            yield words

        for i in range(len(words) - n + 1):
            yield words[i:i+n]

    def _calculate_pmi(self, sents, pmi_window_width=2):
        doc_text = ' '.join(sents)
        word_counts = Counter()
        cooccur_counts = defaultdict(Counter)
        pmi = defaultdict(Counter)
        max_pmi = 0.
        for ngram in self._iter_ngrams(doc_text.split(), n=pmi_window_width):
            for i, src_word in enumerate(ngram):
                if src_word not in self.vocab.word_list():    # OOV
                    continue
                word_counts[src_word] += 1
                for j, tgt_word in enumerate(ngram):
                    if i != j and tgt_word != 0:
                        cooccur_counts[src_word][tgt_word] += 1

        log_total_counts = math.log(sum(word_counts.values()))
        for src_word, tgt_word_counts in cooccur_counts.items():
            for tgt_word, counts in tgt_word_counts.items():
                unconstrained_pmi = log_total_counts + math.log(counts)
                unconstrained_pmi -= math.log(word_counts[src_word] *
                                                word_counts[tgt_word])
                if unconstrained_pmi > 0.:
                    pmi[src_word][tgt_word] = unconstrained_pmi
                    max_pmi = max(max_pmi, unconstrained_pmi)
        return {'pmi': pmi, 'max_pmi': max_pmi}

    def get_labels_only(self):
        labels = []
        for index in range(0, self.__len__()):
            item = self.get_example(index)
            label = self.pad_label_m(item.label_matrix)
            labels.append(label)
        return labels

    def __len__(self):
        return self.size

class Example(object):
    """Class representing a train/val/test example for single-document extractive summarization."""

    def __init__(self, article_sents, abstract_sents, vocab, sent_max_len, label, w_cefr=None, sentaspara=None, speaker_id=None, pos=None, dep=None, ged=None):
    # def __init__(self, article_sents, abstract_sents, vocab, sent_max_len, label):
        """ Initializes the Example, performing tokenization and truncation to produce the encoder, decoder and target sequences, which are stored in self.

        :param article_sents: list(strings) for single document or list(list(string)) for multi-document; one per article sentence. each token is separated by a single space.
        :param abstract_sents: list(strings); one per abstract sentence. In each sentence, each token is separated by a single space.
        :param vocab: Vocabulary object
        :param sent_max_len: int, max length of each sentence
        :param label: list, the No of selected sentence, e.g. [1,3,5]
        """

        self.sent_max_len = sent_max_len
        self.sent_word_input = []
        self.enc_sent_len = []
        self.enc_sent_input = []
        self.enc_sent_input_pad = []
        self.enc_sent_pos = []
        self.enc_sent_pos_pad = []
        self.enc_sent_dep = []
        self.enc_sent_dep_pad = []
        self.enc_sent_ged = []
        self.enc_sent_ged_pad = []
        if w_cefr:
            self.enc_sent_wcefr = []
            self.enc_sent_wcefr_pad = []
        if sentaspara == 'sent':
            self.enc_para_input = []
            self.enc_para_input_pad = []

        # Store the original strings
        self.original_article_sents = article_sents
        self.original_article_pos = pos
        self.original_article_dep = dep
        self.original_article_ged = ged
        self.original_abstract = "\n".join(abstract_sents)
        if w_cefr:
            self.original_wcefrs = w_cefr

        # Process the article
        if isinstance(article_sents, list) and isinstance(article_sents[0], list):  # multi document
            self.original_article_sents = []
            for doc in article_sents:
                self.original_article_sents.extend(doc)

        if isinstance(w_cefr, list) and isinstance(w_cefr[0], list):  # multi document
            self.original_wcefrs = []
            for doc in w_cefr:
                self.original_wcefrs.extend(doc)

        for i, sent in enumerate(self.original_article_sents):
            article_words = sent.split()
            self.sent_word_input.append(article_words)
            self.enc_sent_len.append(len(article_words))  # store the length before padding
            self.enc_sent_input.append([vocab.word2id(w.lower()) for w in article_words])  # list of word ids; OOVs are represented by the id for UNK token
            if w_cefr:
                self.enc_sent_wcefr.append([CEFR2INT[w.upper()] for w in w_cefr[i].split()])
            if sentaspara == 'sent':
                self.enc_para_input.extend([vocab.word2id(w.lower()) for w in article_words])
        
        if pos is not None:
            for i, poss in enumerate(self.original_article_pos):
                poss = poss.split()
                self.enc_sent_pos.append([POS2INT[w] for w in poss])
        
        if dep is not None:
            for i, deps in enumerate(self.original_article_dep):
                deps = deps.split()
                self.enc_sent_dep.append([DEP2INT[w] for w in deps])

        if ged is not None:
            for i, geds in enumerate(self.original_article_ged):
                geds = geds.split()
                self.enc_sent_ged.append([GED2INT[w] for w in geds])


        self._pad_encoder_input(vocab.word2id('[PAD]'))
        if w_cefr:
            self._pad_encoder_cefr(CEFR2INT['[PAD]'])
        self._pad_encoder_pos(POS2INT['X'])
        self._pad_encoder_dep(DEP2INT['X'])
        self._pad_encoder_ged(GED2INT['X'])

        # Store the label
        self.label = label
        label_shape = (len(self.original_article_sents), 6)  # [N, 6] due to A1 to C2 (1-6) in CEFR
        label_index = int(label) - 1
        self.label_matrix = np.zeros(label_shape, dtype=int)
        if label:
            self.label_matrix[:, label_index] = 1  # label_matrix[i][j]=1 indicate the i-th sent will be selected in j-th step

        if sentaspara == 'sent':
            self.label_para_matrix = np.zeros((1, 6), dtype=int)
            if label:
                self.label_para_matrix[:, label_index] = 1
            
        self.speaker_id = None    
        if speaker_id:
            self.speaker_id = speaker_id

    def _pad_encoder_input(self, pad_id):
        """
        :param pad_id: int; token pad id
        :return: 
        """
        max_len = self.sent_max_len
        for i in range(len(self.enc_sent_input)):
            article_words = self.enc_sent_input[i].copy()
            if len(article_words) > max_len:
                article_words = article_words[:max_len]
            if len(article_words) < max_len:
                article_words.extend([pad_id] * (max_len - len(article_words)))
            self.enc_sent_input_pad.append(article_words)

        article_words = self.enc_para_input.copy()
        if len(article_words) > max_len:
            article_words = article_words[:max_len]
        if len(article_words) < max_len:
            article_words.extend([pad_id] * (max_len - len(article_words)))
        self.enc_para_input_pad.append(article_words)

    def _pad_encoder_cefr(self, pad_id):
        """
        :param pad_id: int; token pad id
        :return: 
        """
        max_len = self.sent_max_len
        for i in range(len(self.enc_sent_wcefr)):
            article_cefrs = self.enc_sent_wcefr[i].copy()
            if len(article_cefrs) > max_len:
                article_cefrs = article_cefrs[:max_len]
            if len(article_cefrs) < max_len:
                article_cefrs.extend([pad_id] * (max_len - len(article_cefrs)))
            self.enc_sent_wcefr_pad.append(article_cefrs)

    def _pad_encoder_pos(self, pad_id):
        """
        :param pad_id: int; token pad id
        :return: 
        """
        max_len = self.sent_max_len
        for i in range(len(self.enc_sent_pos)):
            article_poss = self.enc_sent_pos[i].copy()
            if len(article_poss) > max_len:
                article_poss = article_poss[:max_len]
            if len(article_poss) < max_len:
                article_poss.extend([pad_id] * (max_len - len(article_poss)))
            self.enc_sent_pos_pad.append(article_poss)

    def _pad_encoder_dep(self, pad_id):
        """
        :param pad_id: int; token pad id
        :return: 
        """
        max_len = self.sent_max_len
        for i in range(len(self.enc_sent_dep)):
            article_deps = self.enc_sent_dep[i].copy()
            if len(article_deps) > max_len:
                article_deps = article_deps[:max_len]
            if len(article_deps) < max_len:
                article_deps.extend([pad_id] * (max_len - len(article_deps)))
            self.enc_sent_dep_pad.append(article_deps)

    def _pad_encoder_ged(self, pad_id):
        """
        :param pad_id: int; token pad id
        :return: 
        """
        max_len = self.sent_max_len
        for i in range(len(self.enc_sent_ged)):
            article_geds = self.enc_sent_ged[i].copy()
            if len(article_geds) > max_len:
                article_geds = article_geds[:max_len]
            if len(article_geds) < max_len:
                article_geds.extend([pad_id] * (max_len - len(article_geds)))
            self.enc_sent_ged_pad.append(article_geds)


class Example2(Example):
    """Class representing a train/val/test example for multi-document extractive summarization."""

    def __init__(self, article_sents, abstract_sents, vocab, sent_max_len, label, w_cefr=None, sentaspara=None):
        """ Initializes the Example, performing tokenization and truncation to produce the encoder, decoder and target sequences, which are stored in self.

        :param article_sents: list(list(string)) for multi-document; one per article sentence. each token is separated by a single space.
        :param abstract_sents: list(strings); one per abstract sentence. In each sentence, each token is separated by a single space.
        :param vocab: Vocabulary object
        :param sent_max_len: int, max length of each sentence
        :param label: list, the No of selected sentence, e.g. [1,3,5]
        """

        super().__init__(article_sents, abstract_sents, vocab, sent_max_len, label)
        cur = 0
        self.original_articles = []
        self.article_len = []
        self.enc_doc_input = []
        for doc in article_sents:
            if len(doc) == 0:
                continue
            docLen = len(doc)
            self.original_articles.append(" ".join(doc))
            self.article_len.append(docLen)
            self.enc_doc_input.append(catDoc(self.enc_sent_input[cur:cur + docLen]))
            cur += docLen


######################################### ExampleSet #########################################

class ExamplePromptSet(torch.utils.data.Dataset):
    """ Constructor: Dataset of example(object) for single document summarization"""

    def __init__(self, data_path, interviewer_data_path, bert_data_path, vocab, doc_max_timesteps, sent_max_len, passage_length, filter_word_path, w2s_path, pmi_window_width, tokenizer, hps):
        """ Initializes the ExampleSet with the path of data
        
        :param data_path: string; the path of data
        :param interviewer_data_path: string; the path of data from interviewer
        :param bert_data_path: string; the path of data from both interviewer and interviewee
        :param vocab: object;
        :param doc_max_timesteps: int; the maximum sentence number of a document, each example should pad sentences to this length
        :param sent_max_len: int; the maximum token number of a sentence, each sentence should pad tokens to this length
        :param filter_word_path: str; file path, the file must contain one word for each line and the tfidf value must go from low to high (the format can refer to script/lowTFIDFWords.py) 
        :param w2s_path: str; file path, each line in the file contain a json format data (which can refer to the format can refer to script/calw2sTFIDF.py)
        :param pmi_window_width: widow with when calculatin NPMI
        :param tokenizer: WordPeice Tokenizer
        :param hps: parameters
        """

        self.hps = hps
        self.vocab = vocab
        self.sent_max_len = sent_max_len
        self.segment_length = passage_length
        self.doc_max_timesteps = doc_max_timesteps

        logger.info("[INFO] Start reading %s", self.__class__.__name__)
        start = time.time()
        self.example_list = readJson(data_path)
        if bert_data_path:
            self.bert_example_list = readJson(bert_data_path)
        logger.info("[INFO] Finish reading %s. Total time is %f, Total size is %d", self.__class__.__name__,
                    time.time() - start, len(self.example_list))
        if interviewer_data_path is not None:
            self.interviewer_example_list = readJson(interviewer_data_path)
            logger.info("[INFO] Finish reading %s. Total time is %f, Total size is %d", self.__class__.__name__,
                        time.time() - start, len(self.interviewer_example_list))
        self.size = len(self.example_list)
        self.data_set = data_path.split('/')[-1].split('.')[0]
        self.doc_pmi_dir = os.path.join(self.hps.cache_dir, 'pmi')
        if not os.path.exists(self.doc_pmi_dir):
            os.makedirs(self.doc_pmi_dir)
        self.G_dir = os.path.join(self.hps.cache_dir, 'G')
        if not os.path.exists(self.G_dir):
            os.makedirs(self.G_dir)
            
        logger.info("[INFO] Loading filter word File %s", filter_word_path)
        tfidf_w = readText(filter_word_path)
        self.filterwords = FILTERWORD
        self.filterids = [vocab.word2id(w.lower()) for w in FILTERWORD]
        self.filterids.append(vocab.word2id("[PAD]"))   # keep "[UNK]" but remove "[PAD]"
        lowtfidf_num = 0
        pattern = r"^[0-9]+$"
        for w in tfidf_w:
            if vocab.word2id(w) != vocab.word2id('[UNK]'):
                self.filterwords.append(w)
                self.filterids.append(vocab.word2id(w))
                # if re.search(pattern, w) == None:  # if w is a number, it will not increase the lowtfidf_num
                    # lowtfidf_num += 1
                lowtfidf_num += 1
            if lowtfidf_num > 5000:
                break

        logger.info("[INFO] Loading word2sent TFIDF file from %s!" % w2s_path)
        self.w2s_tfidf = readJson(w2s_path)
        
        self.pmi_window_width = pmi_window_width
        if pmi_window_width > -1:
            logger.info("[INFO] Use N-PMI!")
        
        self.tokenizer = tokenizer

    def get_example(self, index):
        e = self.example_list[index]
        e["summary"] = e.setdefault("summary", [])
        example = Example(e["text"], e["summary"], self.vocab, self.sent_max_len, e["label"], w_cefr=e["w_cefr"], sentaspara=self.hps.sentaspara, speaker_id=None if not self.hps.eval_speaker_wise else e["speaker_id"])
        return example

    def get_bert_example(self, index):
        e = self.bert_example_list[index]
        t = e.get('text')
        f = []
        if self.hps.sentaspara == 'sent':
            t = [' '.join(t)]
        if self.hps.sentaspara == 'para':
            for p in t:
                ps = [self.tokenizer.cls_token]
                for s in p:
                    for w in s.split():
                        wp = self.tokenizer.tokenize(w)
                        if len(wp) > 1:
                            wp = [self.tokenizer.unk_token]
                        ps.extend(wp)
                    ps.append(self.tokenizer.sep_token)
                f.append(' '.join(ps[1:-1]))
        elif self.hps.sentaspara == 'sent':
            for p in t:
                ps = [self.tokenizer.cls_token]
                for w in p.split():
                    wp = self.tokenizer.tokenize(w)
                    if len(wp) > 1:
                        wp = [self.tokenizer.unk_token]
                    ps.extend(wp)
                ps.append(self.tokenizer.sep_token)
                f.append(' '.join(ps[1:-1]))
        f = [self.tokenizer(f, padding="max_length", truncation=True)]
        return f

    def get_interviewer_example(self, index):
        e = self.interviewer_example_list[index]
        e["summary"] = e.setdefault("summary", [])
        example = Example(e["text"], e["summary"], self.vocab, self.sent_max_len, e["label"], w_cefr=e["w_cefr"], sentaspara=self.hps.sentaspara, speaker_id=None if not self.hps.eval_speaker_wise else e["speaker_id"])
        return example

    def pad_label_m(self, label_matrix):
        label_m = label_matrix[:self.doc_max_timesteps, :self.doc_max_timesteps]
        # N, m = label_m.shape
        # if m < self.doc_max_timesteps:
        #     pad_m = np.zeros((N, self.doc_max_timesteps - m))
        #     return np.hstack([label_m, pad_m])
        return label_m

    def AddWordNode(self, G, inputid):
        wid2nid = {}
        nid2wid = {}
        nid = 0
        for sentid in inputid:
            for wid in sentid:
                if wid not in self.filterids and wid not in wid2nid.keys():
                    wid2nid[wid] = nid
                    nid2wid[nid] = wid
                    nid += 1

        w_nodes = len(nid2wid)

        G.add_nodes(w_nodes)
        G.set_n_initializer(dgl.init.zero_initializer)
        G.ndata["unit"] = torch.zeros(w_nodes)
        G.ndata["id"] = torch.LongTensor(list(nid2wid.values()))
        G.ndata["dtype"] = torch.zeros(w_nodes)

        return wid2nid, nid2wid

    def CreateGraph(self, input_pad, label, w2s_w, w2w_pmi_info):
        """ Create a graph for each document
        
        :param input_pad: list(list); [sentnum, wordnum]
        :param label: list(list); [sentnum, sentnum]
        :param w2s_w: dict(dict) {str: {str: float}}; for each sentence and each word, the tfidf between them
        :param w2w_pmi_info: dict(dict) {str: {str: int}}; for each word and each word, the n-pmi between them
        :return: G: dgl.graph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
            edge:
                word2sent, sent2word:  tffrac=int, dtype=0
                word2word:             tffrac=int, dtype=1
        """
        G = dgl.DGLGraph()
        wid2nid, nid2wid = self.AddWordNode(G, input_pad)
        w_nodes = len(nid2wid)

        N = len(input_pad)
        G.add_nodes(N)
        G.ndata["unit"][w_nodes:] = torch.ones(N)
        G.ndata["dtype"][w_nodes:] = torch.ones(N)
        sentid2nid = [i + w_nodes for i in range(N)]

        G.set_e_initializer(dgl.init.zero_initializer)
        if self.pmi_window_width > -1:
            max_pmi = w2w_pmi_info.get('max_pmi')
            pmi_mat = w2w_pmi_info.get('pmi')
            for i in range(N):
                c = Counter(input_pad[i])
                sent_nid = sentid2nid[i]
                sent_tfw = w2s_w[str(i)]
                for s_wid in c.keys():
                    if s_wid in wid2nid.keys() and self.vocab.id2word(s_wid) in sent_tfw.keys():
                        for t_wid in c.keys():
                            if t_wid in wid2nid.keys() and self.vocab.id2word(t_wid) in sent_tfw.keys():
                                s2t = pmi_mat[self.vocab.id2word(s_wid)][self.vocab.id2word(t_wid)] / max_pmi
                                t2s = pmi_mat[self.vocab.id2word(t_wid)][self.vocab.id2word(s_wid)] / max_pmi
                                s2t = np.round(s2t * 9)
                                t2s = np.round(t2s * 9)
                                G.add_edges(wid2nid[s_wid], wid2nid[t_wid],
                                            data={"tffrac": torch.LongTensor([s2t]), "dtype": torch.Tensor([1])})
                                G.add_edges(wid2nid[t_wid], wid2nid[s_wid],
                                            data={"tffrac": torch.LongTensor([t2s]), "dtype": torch.Tensor([1])})
        for i in range(N):
            c = Counter(input_pad[i])
            sent_nid = sentid2nid[i]
            sent_tfw = w2s_w[str(i)]
            for wid in c.keys():
                if wid in wid2nid.keys() and self.vocab.id2word(wid) in sent_tfw.keys():
                    tfidf = sent_tfw[self.vocab.id2word(wid)]
                    tfidf_box = np.round(tfidf * 9)  # box = 10
                    G.add_edges(wid2nid[wid], sent_nid,
                                data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
                    G.add_edges(sent_nid, wid2nid[wid],
                                data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
            
            # The two lines can be commented out if you use the code for your own training, since HSG does not use sent2sent edges. 
            # However, if you want to use the released checkpoint directly, please leave them here.
            # Otherwise it may cause some parameter corresponding errors due to the version differences.
            G.add_edges(sent_nid, sentid2nid, data={"dtype": torch.ones(N)})
            G.add_edges(sentid2nid, sent_nid, data={"dtype": torch.ones(N)})
        G.nodes[sentid2nid].data["words"] = torch.LongTensor(input_pad)  # [N, seq_len]
        G.nodes[sentid2nid].data["position"] = torch.arange(1, N + 1).view(-1, 1).long()  # [N, 1]
        G.nodes[sentid2nid].data["label"] = torch.LongTensor(label)  # [N, doc_max]

        return G

    def CreateItvrGraph(self, input_pad):
        """ Create a graph for each document
        
        :param input_pad: list(list); [sentnum, wordnum]
        :param label: list(list); [sentnum, sentnum]
        :param w2s_w: dict(dict) {str: {str: float}}; for each sentence and each word, the tfidf between them
        :param w2w_pmi_info: dict(dict) {str: {str: int}}; for each word and each word, the n-pmi between them
        :return: G: dgl.graph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
        """
        G = dgl.DGLGraph()
        wid2nid, nid2wid = self.AddWordNode(G, input_pad)
        w_nodes = len(nid2wid)

        N = len(input_pad)
        G.add_nodes(N)
        G.ndata["unit"][w_nodes:] = torch.ones(N)
        G.ndata["dtype"][w_nodes:] = torch.ones(N)
        sentid2nid = [i + w_nodes for i in range(N)]
        
        G.nodes[sentid2nid].data["words"] = torch.LongTensor(input_pad)  # [N, seq_len]
        G.nodes[sentid2nid].data["position"] = torch.arange(1, N + 1).view(-1, 1).long()  # [N, 1]

        return G
    
    def CreateItvrPGraph(self, input_pad, para_pad):
        """ Create a graph for each document
        
        :param input_pad: list(list); [sentnum, wordnum]
        :param label: list(list); [sentnum, sentnum]
        :param w2s_w: dict(dict) {str: {str: float}}; for each sentence and each word, the tfidf between them
        :param w2w_pmi_info: dict(dict) {str: {str: int}}; for each word and each word, the n-pmi between them
        :return: G: dgl.graph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
        """

        # G = dgl.DGLGraph()
        # wid2nid, nid2wid = self.AddWordNode(G, input_pad)
        # w_nodes = len(nid2wid)

        # N = len(para_pad)
        # G.add_nodes(N)

        # G.ndata["unit"][w_nodes:] = torch.ones(N)
        # G.ndata["dtype"][w_nodes:] = torch.ones(N)
        # paragraphid2nid = [i + w_nodes for i in range(N)]

        # G.nodes[paragraphid2nid].data["words"] = torch.LongTensor(para_pad)  # [N, seq_len]
        # G.nodes[paragraphid2nid].data["position"] = torch.arange(1, N + 1).view(-1, 1).long()  # [N, 1]

        G = dgl.DGLGraph()
       
        # add sentence node 
        num_sentence = len(input_pad)
        G.add_nodes(num_sentence)
        G.ndata["unit"] = torch.ones(num_sentence)
        G.ndata["dtype"] = torch.ones(num_sentence)
        sentid2nid = [i for i in range(num_sentence)]
  
        # add paragraph node
        # we use this function in __getitem__, where index 1 element everytime, if you want to do multiple paragraph here, you need to rewrite this graph building function
        num_paragraph = len(para_pad)
        G.add_nodes(num_paragraph)
        G.ndata["unit"][num_sentence:] = torch.zeros(num_paragraph)
        G.ndata["dtype"][num_sentence:] = torch.zeros(num_paragraph)
        paragraphid2nid = [i + num_sentence for i in range(num_paragraph)]

        # G.set_e_initializer(dgl.init.zero_initializer)

        # for i in range(num_paragraph):
        #     node_paragraph_id = paragraphid2nid[i] #convert paragraph id to node id of pass in graph_wp
        #     for sent_node_id in sentid2nid:
        #         # consider bidirectional way of node from paragraph --> sentence
        #         G.add_edges(node_paragraph_id, sent_node_id, data={"dtype": torch.Tensor([0])})
        #         G.add_edges(sent_node_id, node_paragraph_id, data={"dtype": torch.Tensor([0])})

        G.nodes[sentid2nid].data["words"] = torch.LongTensor(input_pad)  # [N, seq_len]
        G.nodes[sentid2nid].data["position"] = torch.arange(1, num_sentence + 1).view(-1, 1).long()  # [N, 1]
        G.nodes[paragraphid2nid].data["words"] = torch.LongTensor(para_pad)  # [N, seq_len]
        G.nodes[paragraphid2nid].data["position"] = torch.arange(1, num_paragraph + 1).view(-1, 1).long()  # [N, 1]

        return G

    def CreateGraphC2W(self, input_pad, cefr_input_pad, w2s_w):
        G = dgl.DGLGraph()
        wid2nid, nid2wid = self.AddWordNode(G, input_pad)
        w_nodes = len(nid2wid)
        N = 7 # A1, A2, B1, B2, C1, C2, a trash can
        M = len(input_pad)
        
        G.add_nodes(N)
        G.ndata["unit"][w_nodes:] = torch.ones(N)
        G.ndata["dtype"][w_nodes:] = torch.ones(N)
        cefrid2nid = [i + w_nodes for i in range(N)]
        
        G.add_nodes(M)
        G.ndata["unit"][w_nodes+N:] = torch.full((1, M), 2.).squeeze(0)
        G.ndata["dtype"][w_nodes+N:] = torch.full((1, M), 2.).squeeze(0)
        sentid2nid = [i + w_nodes + N for i in range(M)]
        
        for i in range(M):
            w_l = input_pad[i]
            c_l = cefr_input_pad[i]
            sent_tfw = w2s_w[str(i)]
            for cid, wid in zip(c_l, w_l):
                if wid in wid2nid.keys() and self.vocab.id2word(wid) in sent_tfw.keys():
                    n_cid = cid + w_nodes - 1 # index
                    relation = [1]
                    G.add_edges(wid2nid[wid], n_cid,
                                data={"tffrac": torch.LongTensor(relation), "dtype": torch.Tensor([0])})
                    G.add_edges(n_cid, wid2nid[wid],
                                data={"tffrac": torch.LongTensor(relation), "dtype": torch.Tensor([0])})
                    
        G.nodes[sentid2nid].data["label"] = torch.LongTensor(cefr_input_pad)
        return G

    def CreateGraphWP(self, para_pad, w2s_w):
        """ Create a graph for each document
        :param para_pad: list(list); [sentnum, wordnum]
        :param w2s_w: dict(dict) {str: {str: float}}; for each sentence and each word, the tfidf between them
        :return: G: dgl.graph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                paragraph: unit=1, dtype=1, words=tensor, position=int
            edge:
                word2pass, pass2word:  tffrac=int, dtype=0
        """
        G = dgl.DGLGraph()

        #add word node to empty graph
        wid2nid, nid2wid = self.AddWordNode(G, para_pad)
        w_nodes = len(nid2wid)  #number nodes

        num_paragraph = len(para_pad)
        G.add_nodes(num_paragraph)

        G.ndata["unit"][w_nodes:] = torch.ones(num_paragraph)
        G.ndata["dtype"][w_nodes:] = torch.ones(num_paragraph)

        #define sentence node id use word node id
        paragraphid2nid = [i + w_nodes for i in range(num_paragraph)]

        G.set_e_initializer(dgl.init.zero_initializer)

        for i in range(num_paragraph):
            c = Counter(para_pad[i])
            paragraph_id = paragraphid2nid[i] #convert paragraph id to node id of pass in graph_wp
            for wid in c.keys():
                if wid in wid2nid.keys():
                    #only consider edge from word to paragraph
                    G.add_edges(wid2nid[wid], paragraph_id, data={"dtype": torch.Tensor([0])})
                    if self.hps.retain_wp_relation:
                        G.add_edges(paragraph_id, wid2nid[wid], data={"dtype": torch.Tensor([0])})
        return G

    def CreateGraphSP(self, input_pad, para_pad, w2s_w, label_para):

        G = dgl.DGLGraph()
       
        # add sentence node 
        num_sentence = len(input_pad)
        G.add_nodes(num_sentence)
        G.ndata["unit"] = torch.ones(num_sentence)
        G.ndata["dtype"] = torch.ones(num_sentence)
        sentid2nid = [i for i in range(num_sentence)]
  
        # add paragraph node
        # we use this function in __getitem__, where index 1 element everytime, if you want to do multiple paragraph here, you need to rewrite this graph building function
        num_paragraph = len(para_pad)
        G.add_nodes(num_paragraph)
        G.ndata["unit"][num_sentence:] = torch.zeros(num_paragraph)
        G.ndata["dtype"][num_sentence:] = torch.zeros(num_paragraph)
        paragraphid2nid = [i + num_sentence for i in range(num_paragraph)]

        G.set_e_initializer(dgl.init.zero_initializer)

        for i in range(num_paragraph):
            node_paragraph_id = paragraphid2nid[i] #convert paragraph id to node id of pass in graph_wp
            for sent_node_id in sentid2nid:
                # consider bidirectional way of node from paragraph --> sentence
                G.add_edges(node_paragraph_id, sent_node_id, data={"dtype": torch.Tensor([0])})
                G.add_edges(sent_node_id, node_paragraph_id, data={"dtype": torch.Tensor([0])})

        G.nodes[sentid2nid].data["words"] = torch.LongTensor(input_pad)  # [N, seq_len]
        G.nodes[sentid2nid].data["position"] = torch.arange(1, num_sentence + 1).view(-1, 1).long()  # [N, 1]
        G.nodes[paragraphid2nid].data["words"] = torch.LongTensor(para_pad)  # [N, seq_len]
        G.nodes[paragraphid2nid].data["position"] = torch.arange(1, num_paragraph + 1).view(-1, 1).long()  # [N, 1]
        G.nodes[paragraphid2nid].data["label"] = torch.LongTensor(label_para)

        return G

    def get_labels_only(self):
        labels = []
        for index in range(0, self.__len__()):
            item = self.get_example(index)
            label = self.pad_label_m(item.label_matrix)
            labels.append(label)
        return labels

    def __getitem__(self, index):
        """
        :param index: int; the index of the example
        :return 
            G: graph for the example
            index: int; the index of the example in the dataset
        """
        item = self.get_example(index)
        input_pad = item.enc_sent_input_pad[:self.doc_max_timesteps]
        cefr_pad  = item.enc_sent_wcefr_pad[:self.doc_max_timesteps]
        para_pad  = item.enc_para_input_pad

        ### BERT inputs
        if self.tokenizer is not None:
            bert_input_ids = self.get_bert_example(index)
            bert_input_ids = self._merge_input_ids(bert_input_ids)
            for k, v in bert_input_ids.items():
                assert v.shape[0] == len(input_pad) if self.hps.sentaspara == 'para' else v.shape[0] == len(para_pad), 'Maybe try to add up the value of doc_max_timesteps arguments'

        label = self.pad_label_m(item.label_matrix)
        if self.hps.sentaspara == 'sent':
            label_para = self.pad_label_m(item.label_para_matrix)
        w2s_w = self.w2s_tfidf[index]

        ### W2W and W2P nodes
        doc_pmi_file_name = '.'.join(list(filter(None, 
            [self.data_set, str(index), 'pmi', self.hps.sentaspara, ('pmi{}'.format(self.pmi_window_width) if self.pmi_window_width > -1 else ''), ('itvr' if self.hps.interviewer else None),
            ('cefrbd' if self.hps.cefr_word and (self.hps.cefr_info == 'embed_init') else ''), ('fpbd' if self.hps.filled_pauses_word and (self.hps.filled_pauses_info == 'embed_init') else '')
            ]
            ))
        )
        doc_pmi_path = os.path.join(self.doc_pmi_dir, doc_pmi_file_name)
        if os.path.exists(doc_pmi_path):
            w2w_pmi_info = pikleOpen(doc_pmi_path)
        else:
            w2w_pmi_info = self._calculate_pmi(item.original_article_sents, pmi_window_width=self.pmi_window_width)
            pickleStore(w2w_pmi_info, doc_pmi_path)

        ### Create Graph of basic version
        G_file_name = '.'.join(list(filter(None,
            [self.data_set, str(index), 'G', self.hps.sentaspara, ('pmi{}'.format(self.pmi_window_width) if self.pmi_window_width > -1 else ''), ('itvr' if self.hps.interviewer else None),
            ('cefrbd' if self.hps.cefr_word and (self.hps.cefr_info == 'embed_init') else ''), ('fpbd' if self.hps.filled_pauses_word and (self.hps.filled_pauses_info == 'embed_init') else '')
            ]
            ))
        )
        G_path = os.path.join(self.G_dir, G_file_name)
        if os.path.exists(G_path):
            G = pikleOpen(G_path)
        else:
            G = self.CreateGraph(input_pad, label, w2s_w, w2w_pmi_info)
            pickleStore(G, G_path)

        if self.hps.sentaspara == 'sent':
            ### Create graph of word-paragraph
            G_wp = None
            G_wp_file_name = '.'.join(list(filter(None,
                [self.data_set, str(index), 'G_wp', self.hps.sentaspara, ('pmi{}'.format(self.pmi_window_width) if self.pmi_window_width > -1 else ''), ('itvr' if self.hps.interviewer else None),
                ('wpr' if self.hps.retain_wp_relation else ''),
                ('cefrbd' if self.hps.cefr_word and (self.hps.cefr_info == 'embed_init') else ''), ('fpbd' if self.hps.filled_pauses_word and (self.hps.filled_pauses_info == 'embed_init') else '')
                ]
                ))
            )
            G_wp_path = os.path.join(self.G_dir, G_wp_file_name)
            if os.path.exists(G_wp_path):
                G_wp = pikleOpen(G_wp_path)
            else:
                G_wp = self.CreateGraphWP(para_pad, w2s_w)
                pickleStore(G_wp, G_wp_path)

            ### Create graph of sentence-paragraph
            G_sp = None
            G_sp_file_name = '.'.join(list(filter(None,
                [self.data_set, str(index), 'G_sp', self.hps.sentaspara, ('pmi{}'.format(self.pmi_window_width) if self.pmi_window_width > -1 else ''), ('itvr' if self.hps.interviewer else None),
                ('cefrbd' if self.hps.cefr_word and (self.hps.cefr_info == 'embed_init') else ''), ('fpbd' if self.hps.filled_pauses_word and (self.hps.filled_pauses_info == 'embed_init') else '')
                ]
                ))
            )
            G_sp_path = os.path.join(self.G_dir, G_sp_file_name)
            if os.path.exists(G_sp_path):
                G_sp = pikleOpen(G_sp_path)
            else:
                G_sp = self.CreateGraphSP(input_pad, para_pad, w2s_w, label_para)
                pickleStore(G_sp, G_sp_path)

        ### W2CEFR nodes
        G_c = None
        if self.hps.cefr_info == 'graph_init':            
            G_c_file_name = '.'.join(list(filter(None,
                [self.data_set, str(index), 'G_c', self.hps.sentaspara, 
                ('cefrgh' if self.hps.cefr_word and (self.hps.cefr_info == 'graph_init') else '')]
                ))
            )
            G_c_path = os.path.join(self.G_dir, G_c_file_name)
            if os.path.exists(G_c_path):
                G_c = pikleOpen(G_c_path)
            else:
                G_c = self.CreateGraphC2W(input_pad, cefr_pad, w2s_w)
                pickleStore(G_c, G_c_path)

        # interviewer
        itvr_G = None
        if self.hps.interviewer:
            itvr_item = self.get_interviewer_example(index)
            itvr_input_pad = itvr_item.enc_sent_input_pad[:self.doc_max_timesteps]
            assert len(itvr_input_pad) == len(input_pad), 'Problems in interviewers inputs, {} and {}'.format(len(itvr_input_pad), len(input_pad))
            itvr_G_file_name = '.'.join(list(filter(None,
                [self.data_set, str(index), 'itvr_G', self.hps.sentaspara]
                ))
            )
            itvr_G_path = os.path.join(self.G_dir, itvr_G_file_name)
            if os.path.exists(itvr_G_path):
                itvr_G = pikleOpen(itvr_G_path)
            else:
                itvr_G = self.CreateItvrGraph(itvr_input_pad)
                pickleStore(itvr_G, itvr_G_path)

        # interviewer when is sentaspara == sent
        itvr_PG = None
        if self.hps.interviewer and (self.hps.sentaspara == 'sent'):
            itvr_item = self.get_interviewer_example(index)
            itvr_input_pad = itvr_item.enc_sent_input_pad[:self.doc_max_timesteps]
            itvr_para_pad  = itvr_item.enc_para_input_pad[:self.doc_max_timesteps]
            assert len(itvr_input_pad) == len(input_pad), 'Problems in interviewers inputs, {} and {}'.format(len(itvr_input_pad), len(input_pad))
            itvr_PG_file_name = '.'.join(list(filter(None,
                [self.data_set, str(index), 'itvr_PG', self.hps.sentaspara]
                ))
            )
            itvr_PG_path = os.path.join(self.G_dir, itvr_PG_file_name)
            if os.path.exists(itvr_PG_path):
                itvr_PG = pikleOpen(itvr_PG_path)
            else:
                itvr_PG = self.CreateItvrPGraph(itvr_input_pad, itvr_para_pad)
                pickleStore(itvr_PG, itvr_PG_path)

        speaker_id = item.speaker_id

        if self.tokenizer is not None:
            return G, G_c, G_wp, G_sp, speaker_id, bert_input_ids, itvr_G, itvr_PG
        return G, G_c, G_wp, G_sp, speaker_id, None, itvr_G, itvr_PG

    def _merge_input_ids(self, input_ids_list):
        rtn = {}
        for input_ids in input_ids_list:
            for k, v in input_ids.items():
                rtn.setdefault(
                    k,
                    []
                ).append(v)
        rtn = {k:torch.LongTensor(v_list).squeeze(0) for k, v_list in rtn.items()}
        return rtn

    def _iter_ngrams(self, words, n):
        """Iterate over all word n-grams in a list."""
        if len(words) < n:
            yield words

        for i in range(len(words) - n + 1):
            yield words[i:i+n]

    def _calculate_pmi(self, sents, pmi_window_width=2):
        doc_text = ' '.join(sents)
        word_counts = Counter()
        cooccur_counts = defaultdict(Counter)
        pmi = defaultdict(Counter)
        max_pmi = 0.
        for ngram in self._iter_ngrams(doc_text.split(), n=pmi_window_width):
            for i, src_word in enumerate(ngram):
                if src_word not in self.vocab.word_list():    # OOV
                    continue
                word_counts[src_word] += 1
                for j, tgt_word in enumerate(ngram):
                    if i != j and tgt_word != 0:
                        cooccur_counts[src_word][tgt_word] += 1

        log_total_counts = math.log(sum(word_counts.values()))
        for src_word, tgt_word_counts in cooccur_counts.items():
            for tgt_word, counts in tgt_word_counts.items():
                unconstrained_pmi = log_total_counts + math.log(counts)
                unconstrained_pmi -= math.log(word_counts[src_word] *
                                                word_counts[tgt_word])
                if unconstrained_pmi > 0.:
                    pmi[src_word][tgt_word] = unconstrained_pmi
                    max_pmi = max(max_pmi, unconstrained_pmi)
        return {'pmi': pmi, 'max_pmi': max_pmi}

    def __len__(self):
        return self.size


class ExampleSet(torch.utils.data.Dataset):
    """ Constructor: Dataset of example(object) for single document summarization"""

    def __init__(self, data_path, interviewer_data_path, vocab, doc_max_timesteps, sent_max_len, filter_word_path, w2s_path, pmi_window_width, tokenizer, hps):
        """ Initializes the ExampleSet with the path of data
        
        :param data_path: string; the path of data
        :param interviewer_data_path: string; the path of data from interviewer
        :param vocab: object;
        :param doc_max_timesteps: int; the maximum sentence number of a document, each example should pad sentences to this length
        :param sent_max_len: int; the maximum token number of a sentence, each sentence should pad tokens to this length
        :param filter_word_path: str; file path, the file must contain one word for each line and the tfidf value must go from low to high (the format can refer to script/lowTFIDFWords.py) 
        :param w2s_path: str; file path, each line in the file contain a json format data (which can refer to the format can refer to script/calw2sTFIDF.py)
        :param pmi_window_width: widow with when calculatin NPMI
        :param tokenizer: WordPeice Tokenizer
        :param hps: parameters
        """

        self.hps = hps
        self.vocab = vocab
        self.sent_max_len = sent_max_len
        self.doc_max_timesteps = doc_max_timesteps

        logger.info("[INFO] Start reading %s", self.__class__.__name__)
        start = time.time()
        self.example_list = readJson(data_path)
        logger.info("[INFO] Finish reading %s. Total time is %f, Total size is %d", self.__class__.__name__,
                    time.time() - start, len(self.example_list))
        if interviewer_data_path is not None:
            self.interviewer_example_list = readJson(interviewer_data_path)
            logger.info("[INFO] Finish reading %s. Total time is %f, Total size is %d", self.__class__.__name__,
                        time.time() - start, len(self.interviewer_example_list))
        self.size = len(self.example_list)
        self.data_set = data_path.split('/')[-1].split('.')[0]
        self.doc_pmi_dir = os.path.join(self.hps.cache_dir, 'pmi')
        if not os.path.exists(self.doc_pmi_dir):
            os.makedirs(self.doc_pmi_dir)
        self.G_dir = os.path.join(self.hps.cache_dir, 'G')
        if not os.path.exists(self.G_dir):
            os.makedirs(self.G_dir)
            
        logger.info("[INFO] Loading filter word File %s", filter_word_path)
        tfidf_w = readText(filter_word_path)
        self.filterwords = FILTERWORD
        self.filterids = [vocab.word2id(w.lower()) for w in FILTERWORD]
        self.filterids.append(vocab.word2id("[PAD]"))   # keep "[UNK]" but remove "[PAD]"
        lowtfidf_num = 0
        pattern = r"^[0-9]+$"
        for w in tfidf_w:
            if vocab.word2id(w) != vocab.word2id('[UNK]'):
                self.filterwords.append(w)
                self.filterids.append(vocab.word2id(w))
                # if re.search(pattern, w) == None:  # if w is a number, it will not increase the lowtfidf_num
                    # lowtfidf_num += 1
                lowtfidf_num += 1
            if lowtfidf_num > 5000:
                break

        logger.info("[INFO] Loading word2sent TFIDF file from %s!" % w2s_path)
        self.w2s_tfidf = readJson(w2s_path)
        
        self.pmi_window_width = pmi_window_width
        if pmi_window_width > -1:
            logger.info("[INFO] Use N-PMI!")
        
        self.tokenizer = tokenizer

    def get_example(self, index):
        e = self.example_list[index]
        e["summary"] = e.setdefault("summary", [])
        example = Example(e["text"], e["summary"], self.vocab, self.sent_max_len, e["label"], w_cefr=e["w_cefr"], sentaspara=self.hps.sentaspara, speaker_id=None if not self.hps.eval_speaker_wise else e["speaker_id"])
        return example

    def get_bert_example(self, index):
        e = self.example_list[index]
        t = e.get('text')
        f = []
        for p in t:
            s = []
            for w in p.split():
                wp = self.tokenizer.tokenize(w)
                if len(wp) > 1:
                    wp = [self.tokenizer.unk_token]
                s.extend(wp)
            f.append(self.tokenizer(' '.join(s), padding="max_length", truncation=True))
        return f

    def get_interviewer_example(self, index):
        e = self.interviewer_example_list[index]
        e["summary"] = e.setdefault("summary", [])
        example = Example(e["text"], e["summary"], self.vocab, self.sent_max_len, e["label"], w_cefr=e["w_cefr"], sentaspara=self.hps.sentaspara, speaker_id=None if not self.hps.eval_speaker_wise else e["speaker_id"])
        return example

    def pad_label_m(self, label_matrix):
        label_m = label_matrix[:self.doc_max_timesteps, :self.doc_max_timesteps]
        # N, m = label_m.shape
        # if m < self.doc_max_timesteps:
        #     pad_m = np.zeros((N, self.doc_max_timesteps - m))
        #     return np.hstack([label_m, pad_m])
        return label_m

    def AddWordNode(self, G, inputid):
        wid2nid = {}
        nid2wid = {}
        nid = 0
        for sentid in inputid:
            for wid in sentid:
                if wid not in self.filterids and wid not in wid2nid.keys():
                    wid2nid[wid] = nid
                    nid2wid[nid] = wid
                    nid += 1

        w_nodes = len(nid2wid)

        G.add_nodes(w_nodes)
        G.set_n_initializer(dgl.init.zero_initializer)
        G.ndata["unit"] = torch.zeros(w_nodes)
        G.ndata["id"] = torch.LongTensor(list(nid2wid.values()))
        G.ndata["dtype"] = torch.zeros(w_nodes)

        return wid2nid, nid2wid

    def CreateGraph(self, input_pad, label, w2s_w, w2w_pmi_info):
        """ Create a graph for each document
        
        :param input_pad: list(list); [sentnum, wordnum]
        :param label: list(list); [sentnum, sentnum]
        :param w2s_w: dict(dict) {str: {str: float}}; for each sentence and each word, the tfidf between them
        :param w2w_pmi_info: dict(dict) {str: {str: int}}; for each word and each word, the n-pmi between them
        :return: G: dgl.graph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
            edge:
                word2sent, sent2word:  tffrac=int, dtype=0
                word2word:             tffrac=int, dtype=1
        """
        G = dgl.DGLGraph()
        wid2nid, nid2wid = self.AddWordNode(G, input_pad)
        w_nodes = len(nid2wid)

        N = len(input_pad)
        G.add_nodes(N)
        G.ndata["unit"][w_nodes:] = torch.ones(N)
        G.ndata["dtype"][w_nodes:] = torch.ones(N)
        sentid2nid = [i + w_nodes for i in range(N)]

        G.set_e_initializer(dgl.init.zero_initializer)
        if self.pmi_window_width > -1:
            max_pmi = w2w_pmi_info.get('max_pmi')
            pmi_mat = w2w_pmi_info.get('pmi')
            for i in range(N):
                c = Counter(input_pad[i])
                sent_nid = sentid2nid[i]
                sent_tfw = w2s_w[str(i)]
                for s_wid in c.keys():
                    if s_wid in wid2nid.keys() and self.vocab.id2word(s_wid) in sent_tfw.keys():
                        for t_wid in c.keys():
                            if t_wid in wid2nid.keys() and self.vocab.id2word(t_wid) in sent_tfw.keys():
                                s2t = pmi_mat[self.vocab.id2word(s_wid)][self.vocab.id2word(t_wid)] / max_pmi
                                t2s = pmi_mat[self.vocab.id2word(t_wid)][self.vocab.id2word(s_wid)] / max_pmi
                                s2t = np.round(s2t * 9)
                                t2s = np.round(t2s * 9)
                                G.add_edges(wid2nid[s_wid], wid2nid[t_wid],
                                            data={"tffrac": torch.LongTensor([s2t]), "dtype": torch.Tensor([1])})
                                G.add_edges(wid2nid[t_wid], wid2nid[s_wid],
                                            data={"tffrac": torch.LongTensor([t2s]), "dtype": torch.Tensor([1])})
        for i in range(N):
            c = Counter(input_pad[i])
            sent_nid = sentid2nid[i]
            sent_tfw = w2s_w[str(i)]
            for wid in c.keys():
                if wid in wid2nid.keys() and self.vocab.id2word(wid) in sent_tfw.keys():
                    tfidf = sent_tfw[self.vocab.id2word(wid)]
                    tfidf_box = np.round(tfidf * 9)  # box = 10
                    G.add_edges(wid2nid[wid], sent_nid,
                                data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
                    G.add_edges(sent_nid, wid2nid[wid],
                                data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
            
            # The two lines can be commented out if you use the code for your own training, since HSG does not use sent2sent edges. 
            # However, if you want to use the released checkpoint directly, please leave them here.
            # Otherwise it may cause some parameter corresponding errors due to the version differences.
            G.add_edges(sent_nid, sentid2nid, data={"dtype": torch.ones(N)})
            G.add_edges(sentid2nid, sent_nid, data={"dtype": torch.ones(N)})
        G.nodes[sentid2nid].data["words"] = torch.LongTensor(input_pad)  # [N, seq_len]
        G.nodes[sentid2nid].data["position"] = torch.arange(1, N + 1).view(-1, 1).long()  # [N, 1]
        G.nodes[sentid2nid].data["label"] = torch.LongTensor(label)  # [N, doc_max]

        return G

    def CreateItvrGraph(self, input_pad):
        """ Create a graph for each document
        
        :param input_pad: list(list); [sentnum, wordnum]
        :param label: list(list); [sentnum, sentnum]
        :param w2s_w: dict(dict) {str: {str: float}}; for each sentence and each word, the tfidf between them
        :param w2w_pmi_info: dict(dict) {str: {str: int}}; for each word and each word, the n-pmi between them
        :return: G: dgl.graph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
        """
        G = dgl.DGLGraph()
        wid2nid, nid2wid = self.AddWordNode(G, input_pad)
        w_nodes = len(nid2wid)

        N = len(input_pad)
        G.add_nodes(N)
        G.ndata["unit"][w_nodes:] = torch.ones(N)
        G.ndata["dtype"][w_nodes:] = torch.ones(N)
        sentid2nid = [i + w_nodes for i in range(N)]
        
        G.nodes[sentid2nid].data["words"] = torch.LongTensor(input_pad)  # [N, seq_len]
        G.nodes[sentid2nid].data["position"] = torch.arange(1, N + 1).view(-1, 1).long()  # [N, 1]

        return G

    def CreateGraphC2W(self, input_pad, cefr_input_pad, w2s_w):
        G = dgl.DGLGraph()
        wid2nid, nid2wid = self.AddWordNode(G, input_pad)
        w_nodes = len(nid2wid)
        N = 7 # A1, A2, B1, B2, C1, C2, a trash can
        M = len(input_pad)
        
        G.add_nodes(N)
        G.ndata["unit"][w_nodes:] = torch.ones(N)
        G.ndata["dtype"][w_nodes:] = torch.ones(N)
        cefrid2nid = [i + w_nodes for i in range(N)]
        
        for i in range(M):
            w_l = input_pad[i]
            c_l = cefr_input_pad[i]
            sent_tfw = w2s_w[str(i)]
            for cid, wid in zip(c_l.keys(), w_l.keys()):
                if wid in wid2nid.keys() and self.vocab.id2word(wid) in sent_tfw.keys():
                    n_cid = cid + w_nodes - 1 # index
                    relation = np.array(1)
                    G.add_edges(wid2nid[wid], n_cid,
                                data={"tffrac": torch.LongTensor([relation]), "dtype": torch.Tensor([0])})
                    G.add_edges(n_cid, wid2nid[wid],
                                data={"tffrac": torch.LongTensor([relation]), "dtype": torch.Tensor([0])})

        G.nodes[cefrid2nid].data["cefrs"] = torch.LongTensor(cefr_input_pad)  # [N, seq_len]
        return G

    def __getitem__(self, index):
        """
        :param index: int; the index of the example
        :return 
            G: graph for the example
            index: int; the index of the example in the dataset
        """
        item = self.get_example(index)            
        input_pad = item.enc_sent_input_pad[:self.doc_max_timesteps]
        cefr_pad  = item.enc_sent_wcefr_pad[:self.doc_max_timesteps]
        
        ### BERT inputs
        if self.tokenizer is not None:
            bert_input_ids = self.get_bert_example(index)
            bert_input_ids = self._merge_input_ids(bert_input_ids)
            for k, v in bert_input_ids.items():
                assert v.shape[0] == len(input_pad), 'Maybe try to add up the value of doc_max_timesteps arguments'

        label = self.pad_label_m(item.label_matrix)
        w2s_w = self.w2s_tfidf[index]

        ### W2W and W2P nodes
        doc_pmi_file_name = '.'.join(list(filter(None, 
            [self.data_set, str(index), 'pmi', self.hps.sentaspara, ('pmi{}'.format(self.pmi_window_width) if self.pmi_window_width > -1 else ''), ('itvr' if self.hps.interviewer else None),
            ('cefrbd' if self.hps.cefr_word and (self.hps.cefr_info == 'embed_init') else ''), ('fpbd' if self.hps.filled_pauses_word and (self.hps.filled_pauses_info == 'embed_init') else '')
            ]
            ))
        )
        doc_pmi_path = os.path.join(self.doc_pmi_dir, doc_pmi_file_name)
        if os.path.exists(doc_pmi_path):
            w2w_pmi_info = pikleOpen(doc_pmi_path)
        else:
            w2w_pmi_info = self._calculate_pmi(item.original_article_sents, pmi_window_width=self.pmi_window_width)
            pickleStore(w2w_pmi_info, doc_pmi_path)

        ### Create graph of word-paragraph
        G_file_name = '.'.join(list(filter(None,
            [self.data_set, str(index), 'G', self.hps.sentaspara, ('pmi{}'.format(self.pmi_window_width) if self.pmi_window_width > -1 else ''), ('itvr' if self.hps.interviewer else None),
            ('cefrbd' if self.hps.cefr_word and (self.hps.cefr_info == 'embed_init') else ''), ('fpbd' if self.hps.filled_pauses_word and (self.hps.filled_pauses_info == 'embed_init') else '')
            ]
            ))
        )
        G_path = os.path.join(self.G_dir, G_file_name)
        if os.path.exists(G_path):
            G = pikleOpen(G_path)
        else:
            G = self.CreateGraph(input_pad, label, w2s_w, w2w_pmi_info)
            pickleStore(G, G_path)

        ### Create graph of word-paragraph
        G_wp = None
        G_wp_file_name = '.'.join(list(filter(None,
            [self.data_set, str(index), 'G_wp', self.hps.sentaspara, ('pmi{}'.format(self.pmi_window_width) if self.pmi_window_width > -1 else ''), ('itvr' if self.hps.interviewer else None),
            ('cefrbd' if self.hps.cefr_word and (self.hps.cefr_info == 'embed_init') else ''), ('fpbd' if self.hps.filled_pauses_word and (self.hps.filled_pauses_info == 'embed_init') else '')
            ]
            ))
        )
        G_wp_path = os.path.join(self.G_dir, G_wp_file_name)
        if os.path.exists(G_wp_path):
            G_wp = pikleOpen(G_wp_path)
        else:
            G_wp = self.CreateGraphWP(input_pad, w2s_w)
            pickleStore(G_wp, G_wp_path)

        ### W2CEFR nodes
        G_c = None
        if self.hps.cefr_info == 'graph_init':            
            G_c_file_name = '.'.join(list(filter(None,
                [self.data_set, str(index), 'G_c', self.hps.sentaspara,
                ('cefrgh' if self.hps.cefr_word and (self.hps.cefr_info == 'graph_init') else '')]
                ))
            )
            G_c_path = os.path.join(self.G_dir, G_c_file_name)
            if os.path.exists(G_c_path):
                G_c = pikleOpen(G_c_path)
            else:
                G_c = self.CreateGraphC2W(input_pad, cefr_pad, w2s_w)
                pickleStore(G_c, G_c_path)

        # interviewer
        itvr_G = None
        if self.hps.interviewer:
            itvr_item = self.get_interviewer_example(index)
            itvr_input_pad = itvr_item.enc_sent_input_pad[:self.doc_max_timesteps]
            assert len(itvr_input_pad) == len(input_pad), 'Problems in interviewers inputs, {} and {}'.format(len(itvr_input_pad), len(input_pad))
            itvr_G = self.CreateItvrGraph(itvr_input_pad)

        if self.tokenizer is not None:
            return G, G_c, G_wp, G_sp, index, bert_input_ids, itvr_G
        return G, G_c, G_wp, G_sp, index, None, itvr_G

    def _merge_input_ids(self, input_ids_list):
        rtn = {}
        for input_ids in input_ids_list:
            for k, v in input_ids.items():
                rtn.setdefault(
                    k,
                    []
                ).append(v)
        rtn = {k:torch.LongTensor(v_list) for k, v_list in rtn.items()}
        return rtn

    def _iter_ngrams(self, words, n):
        """Iterate over all word n-grams in a list."""
        if len(words) < n:
            yield words

        for i in range(len(words) - n + 1):
            yield words[i:i+n]

    def _calculate_pmi(self, sents, pmi_window_width=2):
        doc_text = ' '.join(sents)
        word_counts = Counter()
        cooccur_counts = defaultdict(Counter)
        pmi = defaultdict(Counter)
        max_pmi = 0.
        for ngram in self._iter_ngrams(doc_text.split(), n=pmi_window_width):
            for i, src_word in enumerate(ngram):
                if src_word not in self.vocab.word_list():    # OOV
                    continue
                word_counts[src_word] += 1
                for j, tgt_word in enumerate(ngram):
                    if i != j and tgt_word != 0:
                        cooccur_counts[src_word][tgt_word] += 1

        log_total_counts = math.log(sum(word_counts.values()))
        for src_word, tgt_word_counts in cooccur_counts.items():
            for tgt_word, counts in tgt_word_counts.items():
                unconstrained_pmi = log_total_counts + math.log(counts)
                unconstrained_pmi -= math.log(word_counts[src_word] *
                                                word_counts[tgt_word])
                if unconstrained_pmi > 0.:
                    pmi[src_word][tgt_word] = unconstrained_pmi
                    max_pmi = max(max_pmi, unconstrained_pmi)
        return {'pmi': pmi, 'max_pmi': max_pmi}

    def __len__(self):
        return self.size


class MultiExampleSet(ExampleSet):
    """ Constructor: Dataset of example(object) for multiple document summarization"""
    def __init__(self, data_path, vocab, doc_max_timesteps, sent_max_len, filter_word_path, w2s_path, w2d_path):
        """ Initializes the ExampleSet with the path of data

        :param data_path: string; the path of data
        :param vocab: object;
        :param doc_max_timesteps: int; the maximum sentence number of a document, each example should pad sentences to this length
        :param sent_max_len: int; the maximum token number of a sentence, each sentence should pad tokens to this length
        :param filter_word_path: str; file path, the file must contain one word for each line and the tfidf value must go from low to high (the format can refer to script/lowTFIDFWords.py) 
        :param w2s_path: str; file path, each line in the file contain a json format data (which can refer to the format can refer to script/calw2sTFIDF.py)
        :param w2d_path: str; file path, each line in the file contain a json format data (which can refer to the format can refer to script/calw2dTFIDF.py)
        """

        super().__init__(data_path, vocab, doc_max_timesteps, sent_max_len, filter_word_path, w2s_path)

        logger.info("[INFO] Loading word2doc TFIDF file from %s!" % w2d_path)
        self.w2d_tfidf = readJson(w2d_path)

    def get_example(self, index):
        e = self.example_list[index]
        e["summary"] = e.setdefault("summary", [])
        example = Example2(e["text"], e["summary"], self.vocab, self.sent_max_len, e["label"])
        return example

    def MapSent2Doc(self, article_len, sentNum):
        sent2doc = {}
        doc2sent = {}
        sentNo = 0
        for i in range(len(article_len)):
            doc2sent[i] = []
            for j in range(article_len[i]):
                sent2doc[sentNo] = i
                doc2sent[i].append(sentNo)
                sentNo += 1
                if sentNo >= sentNum:
                    return sent2doc
        return sent2doc

    def CreateGraph(self, docLen, sent_pad, doc_pad, label, w2s_w, w2d_w):
        """ Create a graph for each document

        :param docLen: list; the length of each document in this example
        :param sent_pad: list(list), [sentnum, wordnum]
        :param doc_pad: list, [document, wordnum]
        :param label: list(list), [sentnum, sentnum]
        :param w2s_w: dict(dict) {str: {str: float}}, for each sentence and each word, the tfidf between them
        :param w2d_w: dict(dict) {str: {str: float}}, for each document and each word, the tfidf between them
        :return: G: dgl.graph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
                document: unit=1, dtype=2
            edge:
                word2sent, sent2word: tffrac=int, dtype=0
                word2doc, doc2word: tffrac=int, dtype=0
                sent2doc: dtype=2
        """
        # add word nodes
        G = dgl.DGLGraph()
        wid2nid, nid2wid = self.AddWordNode(G, sent_pad)
        w_nodes = len(nid2wid)

        # add sent nodes
        N = len(sent_pad)
        G.add_nodes(N)
        G.ndata["unit"][w_nodes:] = torch.ones(N)
        G.ndata["dtype"][w_nodes:] = torch.ones(N)
        sentid2nid = [i + w_nodes for i in range(N)]
        ws_nodes = w_nodes + N

        # add doc nodes
        sent2doc = self.MapSent2Doc(docLen, N)
        article_num = len(set(sent2doc.values()))
        G.add_nodes(article_num)
        G.ndata["unit"][ws_nodes:] = torch.ones(article_num)
        G.ndata["dtype"][ws_nodes:] = torch.ones(article_num) * 2
        docid2nid = [i + ws_nodes for i in range(article_num)]

        # add sent edges
        for i in range(N):
            c = Counter(sent_pad[i])
            sent_nid = sentid2nid[i]
            sent_tfw = w2s_w[str(i)]
            for wid, cnt in c.items():
                if wid in wid2nid.keys() and self.vocab.id2word(wid) in sent_tfw.keys():
                    tfidf = sent_tfw[self.vocab.id2word(wid)]
                    tfidf_box = np.round(tfidf * 9)  # box = 10
                    # w2s s2w
                    G.add_edge(wid2nid[wid], sent_nid,
                               data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
                    G.add_edge(sent_nid, wid2nid[wid],
                               data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
            # s2d
            docid = sent2doc[i]
            docnid = docid2nid[docid]
            G.add_edge(sent_nid, docnid, data={"tffrac": torch.LongTensor([0]), "dtype": torch.Tensor([2])})

        # add doc edges
        for i in range(article_num):
            c = Counter(doc_pad[i])
            doc_nid = docid2nid[i]
            doc_tfw = w2d_w[str(i)]
            for wid, cnt in c.items():
                if wid in wid2nid.keys() and self.vocab.id2word(wid) in doc_tfw.keys():
                    # w2d d2w
                    tfidf = doc_tfw[self.vocab.id2word(wid)]
                    tfidf_box = np.round(tfidf * 9)  # box = 10
                    G.add_edge(wid2nid[wid], doc_nid,
                               data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
                    G.add_edge(doc_nid, wid2nid[wid],
                               data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})

        G.nodes[sentid2nid].data["words"] = torch.LongTensor(sent_pad)  # [N, seq_len]
        G.nodes[sentid2nid].data["position"] = torch.arange(1, N + 1).view(-1, 1).long()  # [N, 1]
        G.nodes[sentid2nid].data["label"] = torch.LongTensor(label)  # [N, doc_max]

        return G

    def __getitem__(self, index):
        """
        :param index: int; the index of the example
        :return 
            G: graph for the example
            index: int; the index of the example in the dataset
        """
        item = self.get_example(index)
        sent_pad = item.enc_sent_input_pad[:self.doc_max_timesteps]
        enc_doc_input = item.enc_doc_input
        article_len = item.article_len
        label = self.pad_label_m(item.label_matrix)

        G = self.CreateGraph(article_len, sent_pad, enc_doc_input, label, self.w2s_tfidf[index], self.w2d_tfidf[index])

        return G, index


class LoadHiExampleSet(torch.utils.data.Dataset):
    def __init__(self, data_root):
        super().__init__()
        self.data_root = data_root
        self.gfiles = [f for f in os.listdir(self.data_root) if f.endswith("graph.bin")]
        logger.info("[INFO] Start loading %s", self.data_root)

    def __getitem__(self, index):
        graph_file = os.path.join(self.data_root, "%d.graph.bin" % index)
        g, label_dict = load_graphs(graph_file)
        # print(graph_file)
        return g[0], index

    def __len__(self):
        return len(self.gfiles)


######################################### Tools #########################################


import dgl


def catDoc(textlist):
    res = []
    for tlist in textlist:
        res.extend(tlist)
    return res


def readJson(fname):
    data = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def readsingleJson(fname):
    with open(fname, 'r') as f:
        return json.load(f)

def readText(fname):
    data = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            data.append(line.strip())
    return data

def readDialogueSents(file_path, sample_cnt=None):
    max_num_contexts = 176
    data_source = []
    with open(file_path) as f:
        i = 0
        group = {
            'context': [],
            'labels': [],
            'length': -1
        }
        for line in f:
            line = line.strip()
            if (i+1)%(max_num_contexts+1) != 0: # text
                group['context'].append(line)
            else:
                parents, length, relation_types = line.split('|')
                length = int(length.strip())
                relation_types = [int(num) for num in relation_types.strip().split()]
                parents = [int(num) for num in parents.strip().split()]
                rows = []
                cols = []
                for child, parent in enumerate(parents):
                    if parent == -1:
                        continue

                    rows.append(parent)
                    cols.append(child)
                    if parent >= child:
                        print(' '.join(map(str, parents)))
                        print(parent, child)
                        print('error')
                        exit()
                group['labels'] = [rows, cols, relation_types]
                group['length'] = length
                data_source.append(group)
                group = {
                    'context': [],
                    'labels': [],
                    'length': -1
                }
            i += 1
            if sample_cnt is not None and len(data_source) >= sample_cnt:
                break
    return data_source


def readDialogueRelations(file_path, relation_index_dict):
    
    relations = []
    with open(file_path) as f:
        for line in f:
            one_file = []
            line = line.strip()
            chunks = line.split('\t')
            for chunk in chunks:
                sent_pre_idx, relation_name, sent_next_idx = chunk.split()
                one_file.append([sent_pre_idx, sent_next_idx, relation_index_dict[relation_name]])
            relations.append(one_file)

    return relations


def indexof(val, src_list, to=None):
    if val in src_list:
        return src_list.index(val)
    else:
        return to


def graph_collate_fn(samples):
    '''
    :param batch: (G, input_pad)
    :return: 
    '''

    # remove None for args.remove_no_tasks
    samples = [s for s in samples if s is not None]

    # graphs, graphs_c, graph_wp, graph_sp, index, input_ids, itvr_input_ids, itvr_graphs, itvr_p_graphs, meeting_graphs, ie_counts, ir_counts, labels = map(list, zip(*samples))
    graphs = [s.get('G') for s in samples]
    graph_len = [len(g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)) for g in graphs]  # sent node of graph
    
    sorted_len, sorted_index = torch.sort(torch.LongTensor(graph_len), dim=0, descending=True)
    samples = [samples[idx] for idx in sorted_index]
    
    items = list(samples[0].keys())
    
    r = {}
    for t in items:
        tt = [s.get(t) for s in samples]
        if 'G' in t:
            tt = dgl.batch(tt)
        r[t] = tt

    return r
