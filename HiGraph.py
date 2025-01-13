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

import numpy as np

import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

import dgl

# from module.GAT import GAT, GAT_ffn
from module.Encoder import sentEncoder, sentLenEncoder
from module.GAT import WSWGAT, WPWGAT, SPSGAT, PWPGAT
from module.Attention import SelfAttention
from module.PositionEmbedding import get_sinusoid_encoding_table
from module.Decoder import Seq2seqDecoder

from transformers import AutoModel

# from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def ruled_min_max(embed, min=0., max=6.):
    a = torch.where(embed > min, embed, torch.full_like(embed, min))
    b = torch.where(a < max, a, torch.full_like(a, max))
    return b

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, hps):
        super().__init__()
        self.dense = nn.Linear(hps.n_feature, hps.hidden_size)
        self.transform_act_fn = nn.GELU()
        self.LayerNorm = nn.LayerNorm(hps.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class PredictionHead(nn.Module):
    '''
    A prediction head for a single objective of the SpeechGraderModel.
    Args:
        hps (Autohps): the hps for the the pre-trained BERT model
        num_labels (int): the number of labels that can be predicted
    Attributes:
        transform (transformers.modeling_bert.BertPredictionHeadTransform): a dense linear layer with gelu activation
            function
        decoder (torch.nn.Linear): a linear layer that makes predictions across the labels
        bias (torch.nn.Parameter): biases per label
    '''
    def __init__(self, hps, num_labels):
        super(PredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(hps)
        self.decoder = nn.Linear(hps.hidden_size, num_labels, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_labels))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class HSentPromptGraph(nn.Module):
    """ without sent2sent and add residual connection """
    def __init__(self, hps, embed):
        """

        :param hps: 
        :param embed: word embedding
        """
        super().__init__()

        self._hps = hps
        self._n_iter = hps.n_iter
        self._embed = embed
        self.embed_size = hps.word_emb_dim
        self.device2 = None
        if hps.save_gpu_mode:
            self.device2 = torch.device("cuda", int(hps.bert_gpu))

        # BERT encoder
        if hps.bert_config is not None:
            self.bert_device = torch.device("cuda", hps.bert_gpu)
            bert = AutoModel.from_config(hps.bert_config)
            # target_modules = ["pooler.dense"]
            # peft_config = LoraConfig(
            #     task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=target_modules
            # )
            # self.bert = get_peft_model(bert, peft_config)
            if hps.bert_mp:
                self.bert_pl_linear = nn.Linear(hps.bert_config.hidden_size*2, hps.bert_config.hidden_size)

        # sent node mean
        if hps.mean_paragraphs == 'mean_residual':
            self.m_para_residual_linear = nn.Linear(hps.hidden_size * 2, hps.hidden_size)

        # sent node feature
        self._init_sn_param()
        self._TFembed = nn.Embedding(10, hps.feat_embed_size)   # box=10
        self.n_feature_proj = nn.Linear(hps.n_feature_size * 2, hps.hidden_size, bias=False)

        if self._hps.han_s:
            # self.sent_in_para_bias = nn.Parameter(torch.zeros(1), requires_grad=True)
            # self.sent_in_para_relu = nn.ReLU(inplace=True)
            self.sent_in_para_weight_attn1 = nn.Linear(hps.n_feature_size * 2, hps.n_feature_size * 2)
            self.sent_in_para_weight_attn2 = nn.Linear(hps.n_feature_size * 2, 1) # sent attention

        # word -> sent
        embed_size = hps.word_emb_dim
        sent_hidden_size = hps.hidden_size * 2 if hps.interviewer else hps.hidden_size
        para_hidden_size = hps.n_feature_size * 2 if hps.interviewer else hps.n_feature_size
        self.word2sent = WSWGAT(in_dim=embed_size,
                                out_dim=sent_hidden_size,
                                num_heads=hps.n_head,
                                attn_drop_out=hps.atten_dropout_prob,
                                ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                ffn_drop_out=hps.ffn_dropout_prob,
                                feat_embed_size=hps.feat_embed_size,
                                layerType="W2S"
                                )

        # sent -> word
        self.sent2word = WSWGAT(in_dim=sent_hidden_size,
                                out_dim=embed_size,
                                num_heads=6,
                                attn_drop_out=hps.atten_dropout_prob,
                                ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                ffn_drop_out=hps.ffn_dropout_prob,
                                feat_embed_size=hps.feat_embed_size,
                                layerType="S2W"
                                )
        
        if self._hps.pmi_window_width > -1:
            self.word2word = WSWGAT(in_dim=embed_size,
                                    out_dim=embed_size,
                                    num_heads=10,
                                    attn_drop_out=hps.atten_dropout_prob,
                                    ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                    ffn_drop_out=hps.ffn_dropout_prob,
                                    feat_embed_size=hps.feat_embed_size,
                                    layerType="W2W"
                                    )

        # cefr -> word
        if self._hps.cefr_info == 'graph_init':
            self.cefr_embed = torch.nn.Embedding(8, hps.word_emb_dim, padding_idx=0)
            torch.nn.init.xavier_normal_(self.cefr_embed.weight)
            self.cefr2word = WSWGAT(in_dim=embed_size,
                                    out_dim=embed_size,
                                    num_heads=10,
                                    attn_drop_out=hps.atten_dropout_prob,
                                    ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                    ffn_drop_out=hps.ffn_dropout_prob,
                                    feat_embed_size=hps.feat_embed_size,
                                    layerType="C2W"
                                    )

        # paragraph --> sent 
        self.paragraph2sent = SPSGAT(in_dim=para_hidden_size,
                                out_dim=sent_hidden_size,
                                num_heads=1,
                                attn_drop_out=hps.atten_dropout_prob,
                                ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                ffn_drop_out=hps.ffn_dropout_prob,
                                feat_embed_size=hps.feat_embed_size,
                                layerType="P2S"
                                )

        # word -> paragraph
        if hps.retain_wp_relation:
            self.word2paragraph = WPWGAT(in_dim=embed_size,
                                    out_dim=para_hidden_size,
                                    num_heads=1,
                                    attn_drop_out=hps.atten_dropout_prob,
                                    ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                    ffn_drop_out=hps.ffn_dropout_prob,
                                    feat_embed_size=hps.feat_embed_size,
                                    layerType="W2P"
                                    )
            self.paragraph2word = PWPGAT(in_dim=para_hidden_size,
                                    out_dim=embed_size,
                                    num_heads=1,
                                    attn_drop_out=hps.atten_dropout_prob,
                                    ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                    ffn_drop_out=hps.ffn_dropout_prob,
                                    feat_embed_size=hps.feat_embed_size,
                                    layerType="P2W"
                                    )

        # sent -> paragraph
        self.sent2paragraph = SPSGAT(in_dim=sent_hidden_size,
                                out_dim=para_hidden_size,
                                num_heads=1,
                                attn_drop_out=hps.atten_dropout_prob,
                                ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                ffn_drop_out=hps.ffn_dropout_prob,
                                feat_embed_size=hps.feat_embed_size,
                                layerType="S2P"
                                )

        self.n_feature = hps.n_feature_size

        # sent dimension
        n_sent_dim = self.n_feature
        if hps.mean_paragraphs == 'mean':
            n_sent_dim = n_sent_dim
        elif hps.mean_paragraphs == 'mean_residual':
            n_sent_dim = n_sent_dim + hps.hidden_size

        if hps.pred_gated_fusion:
            # trainable gated weight
            if hps.bert_config is not None:
                self.bert_gt_w = nn.Linear(hps.bert_config.hidden_size + n_sent_dim if hps.sentaspara == 'para' else hps.bert_config.hidden_size+para_hidden_size, 1)
                self.down_bert = nn.Linear(hps.bert_config.hidden_size, n_sent_dim if hps.sentaspara == 'para' else para_hidden_size)

            self.n_feature = n_sent_dim
            hps.n_feature = n_sent_dim
            self._hps.n_feature = n_sent_dim
        else:
            final_n_dim = n_sent_dim
            if hps.bert_config is not None:
                final_n_dim = final_n_dim + hps.bert_config.hidden_size

            self.n_feature = final_n_dim
            hps.n_feature = final_n_dim
            self._hps.n_feature = final_n_dim

        if hps.final_attention:
            self.final_attn = SelfAttention(hps.bert_config.hidden_size, n_sent_dim, use_dropout=True)

        if hps.baseline:
            self.n_feature = hps.bert_config.hidden_size

        if hps.wcefr:
            self.wwh = nn.Linear(embed_size, 1)

        if hps.use_doc:
            self.doc_layer = nn.Linear(hps.hidden_size, hps.hidden_size)
            self.doc_att_linear = nn.Linear(hps.hidden_size, 1)

        if hps.wcefr:
        #     # for objective, objective_params in config.training_objectives.items():
        #     #     num_predictions, _ = objective_params
            self.ngram_len_enc = sentLenEncoder(self._hps, self._embed)
            b_d = torch.device("cuda", hps.bert_gpu)
            decoder = Seq2seqDecoder(hps.vocab_size, hps.sent_max_len, 300,
                                     sos_id=None, eos_id=None, output_dim=8, # 8 includes `[PAD]`
                                     num_heads=3, num_layers=1, rnn_type='rnn',
                                     dropout_p=hps.recurrent_dropout_prob, embed=self._embed, device=b_d)
            setattr(self, 'w_cefr' + '_decoder', decoder)
            self.proj_mulihead_sent = nn.Linear(n_sent_dim + 300, 300)
        #         # if (self.use_w_cefr_fusiongating) and (objective in ['score', 'score_ppf']):
        #         #     fusion_cefr_freq_linear = nn.Linear(config.hidden_size+6, config.hidden_size)
        #         #     setattr(self, objective + '_fusion_w_cefr_freq', fusion_cefr_freq_linear)

        if hps.head == 'linear':
            if hps.final_attention:
                # self.wh = nn.Linear(n_sent_dim, 6 if hps.problem_type == 'classification' else 1)
                self.pwh = nn.Linear(para_hidden_size, 6 if hps.problem_type == 'classification' else 1)
            else:
                # self.wh = nn.Linear(self.n_feature, 6 if hps.problem_type == 'classification' else 1)
                self.pwh = nn.Linear(para_hidden_size, 6 if hps.problem_type == 'classification' else 1)
        elif hps.head == 'predictionhead':
            # self.wh = PredictionHead(hps, 6 if hps.problem_type == 'classification' else 1)
            self.pwh = PredictionHead(hps, 6 if hps.problem_type == 'classification' else 1)


    def forward(self, graph, graph_c, graph_wp, graph_sp, graph_itvr, graph_p_itvr, bert_input_ids):
        """
        :param graph: [batch_size] * DGLGraph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
            edge:
                word2sent, sent2word:  tffrac=int, type=0
        :param graph_itvr: [batch_size] * DGLGraph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
        :param bert_input_ids: [batch_size, max_positional_length]
        :return: result: [sentnum, 2]
        """

        # return object
        final_return = {'embed': {'before_gat': {}, 'after_gat': {}},
                        'dec_outputs': {},
                        'results': {}}

        # word node init
        word_feature = self.set_wnfeature(graph)    # [wnode, embed_size]

        # CEFR node init
        if self._hps.cefr_info == 'graph_init':
            cefr_feature = self.set_cnfeature(graph_c)

        sent_feature = self.n_feature_proj(self.set_snfeature(graph))    # [wnode, 2 * lstm_hidden_state] -> [snode, n_feature_size]
        
        # interviewer prompt as condition for the responses of interviewee
        if self._hps.interviewer:
            itvr_sent_feature = self.n_feature_proj(self.set_snfeature(graph_itvr))
            sent_feature = torch.cat((sent_feature, itvr_sent_feature), dim=1)

        # the start state
        word_state = word_feature
        sent_state = self.word2sent(graph, word_feature, sent_feature, od=self._hps.device, dd=self.device2, save_gpu_mode=self._hps.save_gpu_mode)

        # paragraph start state
        paragraph_state = self.set_pnfeature(graph_sp)
        if self._hps.interviewer:
            itvr_paragraph_feature = self.set_pnfeature(graph_p_itvr)
            paragraph_state = torch.cat((paragraph_state, itvr_paragraph_feature), dim=1)

        # Final Return
        final_return['embed']['before_gat'] = {'w': word_feature,
                                               's': sent_state, 
                                               'p': paragraph_state}

        # get baseline
        if self._hps.baseline:
            p = self._get_bert_inputs(bert_input_ids)
            # result = self.wh(p)
            result = None
            # Paragraph CEFR
            p_result = self.pwh(paragraph_state)
            w_result = None
            if self._hps.wcefr:
                w_result = self.wwh(word_state)

            # Final Return
            final_return['results'] = {'w': w_result,
                                       's': result,
                                       'p': p_result}
            return final_return

        for i in range(self._n_iter):
            
            if not self._hps.revserse_metapath:
                # paragraph -> word
                if self._hps.retain_wp_relation:
                    word_state_from_paragraph = self.paragraph2word(graph_wp, word_state, paragraph_state, od=self._hps.device, dd=self.device2, save_gpu_mode=self._hps.save_gpu_mode)
                # sent -> word
                word_state_from_sent = self.sent2word(graph, word_state, sent_state, od=self._hps.device, dd=self.device2, save_gpu_mode=self._hps.save_gpu_mode)
                # word -> word
                if self._hps.pmi_window_width > -1:
                    word_state_from_word = self.word2word(graph, word_state, word_state, od=self._hps.device, dd=self.device2, save_gpu_mode=self._hps.save_gpu_mode)
                # cefr -> word
                if self._hps.cefr_info == 'graph_init':
                    word_state_from_cefr = self.cefr2word(graph_c, word_state, cefr_feature, od=self._hps.device, dd=self.device2, save_gpu_mode=self._hps.save_gpu_mode)
                # word information fusing
                word_state = word_state_from_sent
                if self._hps.pmi_window_width > -1:
                    word_state = word_state + word_state_from_word
                if self._hps.cefr_info == 'graph_init':
                    word_state = word_state + word_state_from_cefr
                if self._hps.retain_wp_relation:
                    word_state = word_state + word_state_from_paragraph
                # word -> sent, paragraph -> sent
                sent_state_from_word = self.word2sent(graph, word_state, sent_state, od=self._hps.device, dd=self.device2, save_gpu_mode=self._hps.save_gpu_mode)
                sent_state_from_paragraph = self.paragraph2sent(graph_sp, paragraph_state, sent_state, od=self._hps.device, dd=self.device2, save_gpu_mode=self._hps.save_gpu_mode)
                sent_state = sent_state_from_word + sent_state_from_paragraph
                # word -> paragraph
                if self._hps.retain_wp_relation:
                    paragraph_state_from_word = self.word2paragraph(graph_wp, word_state, paragraph_state, od=self._hps.device, dd=self.device2, save_gpu_mode=self._hps.save_gpu_mode)
                # sent -> paragraph
                paragraph_state_from_sent = self.sent2paragraph(graph_sp, paragraph_state, sent_state, od=self._hps.device, dd=self.device2, save_gpu_mode=self._hps.save_gpu_mode)
                # paragraph information fusing
                if self._hps.retain_wp_relation:
                    paragraph_state = paragraph_state_from_word + paragraph_state_from_sent
                else:
                    paragraph_state = paragraph_state_from_sent

        # update sent_state
        if self._hps.mean_paragraphs == 'mean_residual':
            mean_sent_state = self._mean_snfeature(graph, sent_state, repeat=True)
            sent_state = torch.cat((sent_state, mean_sent_state), dim=1) # add the information of self-mean
        elif self._hps.mean_paragraphs == 'mean':
            sent_state = self._mean_snfeature(graph, sent_state, repeat=True)
        else:
            sent_state = sent_state

        # BERT encoder
        if self._hps.bert_config is not None:
            if self._hps.sentaspara == 'para':
                p = self._get_bert_inputs(bert_input_ids)

                if self._hps.pred_gated_fusion:
                    b_g_w = torch.sigmoid(self.bert_gt_w(torch.cat((sent_state, p), dim=1)))
                    bert_state = b_g_w * self.down_bert(p)
                else:
                    if self._hps.final_attention:
                        b_sent_state = torch.cat((sent_state, p), dim=1)
                    else:
                        sent_state = torch.cat((sent_state, p), dim=1)
            elif self._hps.sentaspara == 'sent':
                p = self._get_bert_inputs(bert_input_ids)

                if self._hps.pred_gated_fusion:
                    b_g_w = torch.sigmoid(self.bert_gt_w(torch.cat((paragraph_state, p), dim=1)))
                    bert_state = b_g_w * self.down_bert(p)
                else:
                    if self._hps.final_attention:
                        b_paragraph_state = torch.cat((paragraph_state, p), dim=1)
                    else:
                        paragraph_state = torch.cat((b_paragraph_state, p), dim=1)


        if self._hps.pred_gated_fusion:
            if self._hps.bert_config is not None:
                if self._hps.sentaspara == 'para':
                    sent_state = sent_state + bert_state
                elif self._hps.sentaspara == 'sent':
                    paragraph_state = paragraph_state + bert_state

        if self._hps.final_attention and not self._hps.pred_gated_fusion:
            sent_state = self.final_attn(sent_state, p)

        # if self._hps.use_doc:    
        #     list_n_sent_node, list_n_passage_node = [], [] 
        #     G_unbatch, G_sp_unbatch = dgl.unbatch(graph), dgl.unbatch(graph_sp)

        #     for g in G_unbatch:
        #         edges = g.edges()
        #         sentence_node = g.filter_nodes(lambda nodes: nodes.data["unit"]==1)
        #         list_n_sent_node.append(len(sentence_node))
        #     list_sent_represent_matrix = torch.split(sent_state, list_n_sent_node, dim=0)

        #     for g_sp in G_sp_unbatch:
        #         edges = g_sp.edges()
        #         passage_node = g_sp.filter_nodes(lambda nodes: nodes.data["unit"]==0)
        #         list_n_passage_node.append(len(passage_node))
        #     list_passage_represent_matrix = torch.split(passage_state, list_n_passage_node, dim=0) # list of elements, each element match to list passage representation of a doc 

        #     sd_state = []
        #     for i in range(len(list_sent_represent_matrix)):
        #         sents = list_sent_represent_matrix[i] # shape (num sent, hidden size)
        #         passages = list_passage_represent_matrix[i]  # (num topic per doc, hidden size)
        #         doc = self.compute_doc(passages) # (hidden size)
        #         doc_repeat = doc.repeat(sents.shape[0], 1) # shape (num sent, hidden size)
        #         sents_doc = torch.cat((sents, doc_repeat), dim=1) # shape (num sent, 2 * hidden size)
        #         sd_state.append(sents_doc)

        #     sd_state = torch.cat(sd_state, dim=0) # (num sent, 2 * hidden size)
        #     result = self.wh(F.relu(self.l1(sd_state))) # shape (snode , 2)
        # else:
        #     result = self.wh(sent_state)
        result = None
        
        # Paragraph CEFR
        p_result = self.pwh(paragraph_state)
        
        # Word CEFR
        w_result = None
        if self._hps.wcefr:
            # w_result = self.wwh(word_state)
            
            # TODO: use sentence embedding to generate a decoding process
            # first is to get the tokens in each of sentence
            snode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
            s_wid = graph.nodes[snode_id].data["words"]
            lens = [torch.count_nonzero(i).cpu().item() for i in s_wid]
            sent_len_feature = self.ngram_len_enc.forward(graph.nodes[snode_id].data["words"])  # [snode, embed_size]
            sent_len_feature = self.proj_mulihead_sent(torch.cat((sent_len_feature, sent_state.unsqueeze(1).repeat([1, sent_len_feature.shape[1], 1])), dim=2))
            sent_len_feature = sent_len_feature[:, :max(lens)] # cut to the max length of sentence to save computation storage
            decoder = getattr(self, 'w_cefr_decoder')
            b_d = torch.device("cuda", self._hps.bert_gpu)
            decoder = decoder.to(b_d)
            s_wid = s_wid.to(b_d)
            sent_len_feature = sent_len_feature.to(b_d)
            decoder_outputs, ret_dict = decoder(inputs=s_wid, decoding_inputs=None, encoder_outputs=sent_len_feature, lens=lens, teacher_forcing_ratio=0)
            decoder_outputs = [t.to(self._hps.device) for t in decoder_outputs]
            final_return['embed']['dec_outputs'] = {'wcefr': decoder_outputs, 'lens': lens}

        # Final Return
        final_return['embed']['after_gat'] = {'w': word_feature,
                                              's': sent_state, 
                                              'p': paragraph_state}
        final_return['results'] = {'w': w_result,
                                   's': result,
                                   'p': p_result}

        return final_return

    def compute_doc(self, passages):
        z_pass = self.doc_layer(passages) #shape (num topic , hidden size)
        #compute attention 
        w = F.leaky_relu(self.doc_att_linear(z_pass)) #shape (num topic , 1)
        att = F.softmax(w, dim=0) #shape (num topic , 1) satisfy condition: sum = 1
        s = att * z_pass #shape (num topic , 1) * (num topic , hidden size ) --> (num topic , hidden size) 
        out = torch.sum(s, dim=0) #shape (hidden size)
        return out       

    def set_pnfeature(self, graph_sp):
        # sentence node feature from graph_sp, get paragraph node embedding from sentence boundary
        snode_id = graph_sp.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        cnn_feature = self._sent_cnn_feature(graph_sp, snode_id)
        features, glen = get_snode_feat(graph_sp, feat="sent_embedding")
        lstm_feature = self._sent_lstm_feature(features, glen)
        sent_node_feature = torch.cat([cnn_feature, lstm_feature], dim=1)  # [n_nodes, n_feature_size * 2]

        glist = dgl.unbatch(graph_sp)
        unbatch_sntensor = []
        for j in range(len(glist)):
            g = glist[j]
            snode_id = g.filter_nodes(lambda nodes: nodes.data['dtype'] == 1)
            num_sents = len(snode_id)
            st_idx = j*num_sents
            cur_sent_state = sent_node_feature[st_idx: st_idx + num_sents]
            unbatch_sntensor.append(cur_sent_state.view(1, cur_sent_state.shape[0], cur_sent_state.shape[-1]))
        unbatch_sntensor = torch.cat([self.compute_paragraph_feature(k) for k in unbatch_sntensor], dim=0)

        # paragraph node featrue from graph_sp, get paragraph node embedding from paragraph boundary
        pnode_id = graph_sp.filter_nodes(lambda nodes: nodes.data["dtype"] == 0)
        cnn_feature = self._sent_cnn_feature(graph_sp, pnode_id)
        features, glen = get_pnode_feat(graph_sp, feat="sent_embedding")
        lstm_feature = self._sent_lstm_feature(features, glen)
        para_node_feature = self.pnode_proj(torch.cat([cnn_feature, lstm_feature], dim=1))
        node_feature = self.snode_pnode_proj(torch.cat((unbatch_sntensor, para_node_feature), dim=1))
        return node_feature

    def compute_paragraph_feature(self, sent_paragraph_tensor):
        #sent paragraph tensor : (num paragraph , max sentence , sent dim )

        if self._hps.han_s:
            w1 = F.relu(self.sent_in_para_weight_attn1(sent_paragraph_tensor.squeeze(0)))
            w2 = F.relu(self.sent_in_para_weight_attn2(w1))
            att = torch.softmax(w2, dim=0)
            extended_att = att.unsqueeze(0)
            sent_paragraph_tensor = sent_paragraph_tensor * extended_att
        output , (_ , _) = self.lstm_paragraph(sent_paragraph_tensor) #output shape (num paragraph , max sentence , 2 * sent hidden dim )
        paragraph_feature = output[: , -1 , :] #shape (num paragraph, 2* sent dim)
        return self.lstm_para_proj(paragraph_feature) #shape (num paragraph , sent dim)

    def _init_sn_param(self):
        self.sent_pos_embed = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(self._hps.doc_max_timesteps + 1, self.embed_size, padding_idx=0),
            freeze=True)
        self.cnn_proj = nn.Linear(self.embed_size, self._hps.n_feature_size)
        self.lstm_hidden_state = self._hps.lstm_hidden_state
        self.lstm = nn.LSTM(self.embed_size, self.lstm_hidden_state, num_layers=self._hps.lstm_layers, dropout=0.1,
                            batch_first=True, bidirectional=self._hps.bidirectional)
        if self._hps.bidirectional:
            self.lstm_proj = nn.Linear(self.lstm_hidden_state * 2, self._hps.n_feature_size)
        else:
            self.lstm_proj = nn.Linear(self.lstm_hidden_state, self._hps.n_feature_size)

        self.ngram_enc = sentEncoder(self._hps, self._embed)

        # lstm for passage 
        self.lstm_paragraph = nn.LSTM(self._hps.n_feature_size * 2, self._hps.hidden_size, num_layers=self._hps.lstm_layers, dropout=0.1, 
                                    batch_first=True, bidirectional=self._hps.bidirectional)
        self.pnode_proj = nn.Linear(self._hps.n_feature_size * 2, self._hps.n_feature_size)
        self.snode_pnode_proj = nn.Linear(self._hps.n_feature_size * 2, self._hps.n_feature_size)
        
        if self._hps.bidirectional:
            self.lstm_para_proj = nn.Linear(
                self._hps.hidden_size*2, self._hps.n_feature_size)
        else:
            self.lstm_para_proj = nn.Linear(
                self._hps.hidden_size , self._hps.n_feature_size)

    def _sent_cnn_feature(self, graph, snode_id):
        ngram_feature = self.ngram_enc.forward(graph.nodes[snode_id].data["words"])  # [snode, embed_size]
        graph.nodes[snode_id].data["sent_embedding"] = ngram_feature
        snode_pos = graph.nodes[snode_id].data["position"].view(-1)  # [n_nodes]
        position_embedding = self.sent_pos_embed(snode_pos)
        cnn_feature = self.cnn_proj(ngram_feature + position_embedding)
        return cnn_feature

    def _sent_lstm_feature(self, features, glen):
        pad_seq = rnn.pad_sequence(features, batch_first=True)
        lstm_input = rnn.pack_padded_sequence(pad_seq, glen, batch_first=True, enforce_sorted=False)
        lstm_output, _ = self.lstm(lstm_input)
        unpacked, unpacked_len = rnn.pad_packed_sequence(lstm_output, batch_first=True)
        lstm_embedding = [unpacked[i][:unpacked_len[i]] for i in range(len(unpacked))]
        a = torch.cat(lstm_embedding, dim=0) # debug
        lstm_feature = self.lstm_proj(a)  # [n_nodes, n_feature_size]
        return lstm_feature

    def set_wnfeature(self, graph):
        wnode_id = graph.filter_nodes(lambda nodes: nodes.data["unit"]==0)
        wsedge_id = graph.filter_edges(lambda edges: edges.data["dtype"] == 0)   # for word to supernode(sent&doc)
        wid = graph.nodes[wnode_id].data["id"]  # [n_wnodes]
        w_embed = self._embed(wid)  # [n_wnodes, D]
        graph.nodes[wnode_id].data["embed"] = w_embed
        etf = graph.edges[wsedge_id].data["tffrac"]
        graph.edges[wsedge_id].data["tfidfembed"] = self._TFembed(etf)
        if self._hps.pmi_window_width > -1:
            wwedge_id = graph.filter_edges(lambda edges: edges.data["dtype"] == 1)   # for word to word
            eww = graph.edges[wwedge_id].data["tffrac"]
            graph.edges[wwedge_id].data["tfidfembed"] = self._TFembed(eww)
        return w_embed
    
    def set_cnfeature(self, graph_c):        
        cnode_id = graph_c.filter_nodes(lambda nodes: nodes.data["unit"]==1)
        csedge_id = graph_c.filter_edges(lambda edges: edges.data["dtype"]==0)   # for cefr to rely node(cefr&word)
        cid = graph_c.nodes[cnode_id].data["id"]  # [n_wnodes]
        c_embed = self.cefr_embed(cid)  # [n_wnodes, D]
        graph_c.nodes[cnode_id].data["embed"] = c_embed
        etf = graph_c.edges[csedge_id].data["tffrac"]
        graph_c.edges[csedge_id].data["tfidfembed"] = torch.ones(etf.shape[-1] ,self._hps.feat_embed_size).to(self._hps.device)
        return c_embed

    def set_snfeature(self, graph):
        # node feature
        snode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        cnn_feature = self._sent_cnn_feature(graph, snode_id)
        features, glen = get_snode_feat(graph, feat="sent_embedding")
        lstm_feature = self._sent_lstm_feature(features, glen)
        node_feature = torch.cat([cnn_feature, lstm_feature], dim=1)  # [n_nodes, n_feature_size * 2]
        return node_feature

    def _mean_snfeature(self, graph, sent_state, repeat=False):
        repeat_cummulate_list = []
        tensors = []
        glist = dgl.unbatch(graph)
        for j in range(len(glist)):
            g = glist[j]
            snode_id = g.filter_nodes(lambda nodes: nodes.data['dtype'] == 1)
            num_sents = len(snode_id)
            st_idx = j*num_sents
            cur_sent_state = sent_state[st_idx: st_idx + num_sents]

            sent_state_r = cur_sent_state.reshape(1, -1, self._hps.hidden_size)
            sent_state_m = torch.mean(sent_state_r, dim=1)
            repeat_cummulate_list.append(num_sents)
            tensors.append(sent_state_m)

        repeat_cummulate_list = torch.tensor(repeat_cummulate_list).to(self._hps.device)
        if repeat:
            return torch.cat(tensors, dim=0).repeat_interleave(repeat_cummulate_list, dim=0)
        return torch.cat(tensors, dim=0)
    
    def _get_bert_inputs(self, bert_input_ids):
        p = []
        for input_ids in bert_input_ids:
            input_ids = {k: v.to(self.bert_device) for k, v in input_ids.items()}
            self.bert = self.bert.to(self.bert_device)
            bert_output = self.bert(input_ids=input_ids.get('input_ids'),
                                    attention_mask=input_ids.get('attention_mask'),
                                    token_type_ids=input_ids.get('token_type_ids'))
            if self._hps.bert_mp:
                a = mean_pooling(bert_output.get('last_hidden_state').to(self._hps.device), attention_mask=input_ids['attention_mask'].to(self._hps.device))
                b = bert_output.get('pooler_output').to(self._hps.device)
                c = self.bert_pl_linear(torch.cat((a, b), dim=1))
                p.append(c)
            else:
                p.append(bert_output.get('pooler_output').to(self._hps.device))
        return torch.cat(p, dim=0)


class HSumPromptGraph(nn.Module):
    """ without sent2sent and add residual connection """
    def __init__(self, hps, embed):
        """

        :param hps: 
        :param embed: word embedding
        """
        super().__init__()

        self._hps = hps
        self._n_iter = hps.n_iter
        self._embed = embed
        self.embed_size = hps.word_emb_dim

        # BERT encoder
        if hps.bert_config is not None:
            self.bert_device = torch.device("cuda", hps.bert_gpu)
            self.bert = AutoModel.from_config(hps.bert_config)
            if hps.bert_mp:
                self.bert_pl_linear = nn.Linear(hps.bert_config.hidden_size*2, hps.bert_config.hidden_size)

        # sent node mean
        if hps.mean_paragraphs == 'mean_residual':
            self.m_para_residual_linear = nn.Linear(hps.hidden_size * 2, hps.hidden_size)

        # sent node feature
        self._init_sn_param()
        self._TFembed = nn.Embedding(10, hps.feat_embed_size)   # box=10
        self.n_feature_proj = nn.Linear(hps.n_feature_size * 2, hps.hidden_size, bias=False)

        # word -> sent
        embed_size = hps.word_emb_dim
        sent_hidden_size = hps.hidden_size*2 if hps.interviewer else hps.hidden_size
        self.word2sent = WSWGAT(in_dim=embed_size,
                                out_dim=sent_hidden_size,
                                num_heads=hps.n_head,
                                attn_drop_out=hps.atten_dropout_prob,
                                ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                ffn_drop_out=hps.ffn_dropout_prob,
                                feat_embed_size=hps.feat_embed_size,
                                layerType="W2S"
                                )

        # sent -> word
        self.sent2word = WSWGAT(in_dim=sent_hidden_size,
                                out_dim=embed_size,
                                num_heads=6,
                                attn_drop_out=hps.atten_dropout_prob,
                                ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                ffn_drop_out=hps.ffn_dropout_prob,
                                feat_embed_size=hps.feat_embed_size,
                                layerType="S2W"
                                )
        
        if self._hps.pmi_window_width > -1:
            self.word2word = WSWGAT(in_dim=embed_size,
                                    out_dim=embed_size,
                                    num_heads=10,
                                    attn_drop_out=hps.atten_dropout_prob,
                                    ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                    ffn_drop_out=hps.ffn_dropout_prob,
                                    feat_embed_size=hps.feat_embed_size,
                                    layerType="W2W"
                                    )

        # cefr -> word
        if self._hps.cefr_info == 'graph_init':
            self.cefr_embed = torch.nn.Embedding(8, hps.word_emb_dim, padding_idx=0)
            torch.nn.init.xavier_normal_(self.cefr_embed.weight)
            self.cefr2word = WSWGAT(in_dim=embed_size,
                                    out_dim=embed_size,
                                    num_heads=10,
                                    attn_drop_out=hps.atten_dropout_prob,
                                    ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                    ffn_drop_out=hps.ffn_dropout_prob,
                                    feat_embed_size=hps.feat_embed_size,
                                    layerType="C2W"
                                    )

        self.n_feature = hps.n_feature_size

        # sent dimension
        n_sent_dim = self.n_feature
        if hps.mean_paragraphs == 'mean':
            n_sent_dim = n_sent_dim
        elif hps.mean_paragraphs == 'mean_residual':
            n_sent_dim = n_sent_dim + hps.hidden_size

        if hps.pred_gated_fusion:
            # trainable gated weight
            if hps.bert_config is not None:
                self.bert_gt_w = nn.Linear(hps.bert_config.hidden_size + n_sent_dim, 1)
                self.down_bert = nn.Linear(hps.bert_config.hidden_size, n_sent_dim)

            self.n_feature = n_sent_dim
            hps.n_feature = n_sent_dim
            self._hps.n_feature = n_sent_dim
        else:
            final_n_dim = n_sent_dim
            if hps.bert_config is not None:
                final_n_dim = final_n_dim + hps.bert_config.hidden_size

            self.n_feature = final_n_dim
            hps.n_feature = final_n_dim
            self._hps.n_feature = final_n_dim

        if hps.final_attention:
            self.final_attn = SelfAttention(hps.bert_config.hidden_size, n_sent_dim, use_dropout=True)

        if hps.baseline:
            self.n_feature = hps.bert_config.hidden_size

        if hps.wcefr:
            self.wwh = nn.Linear(embed_size, 1)

        if hps.head == 'linear':
            if hps.final_attention:
                self.wh = nn.Linear(n_sent_dim, 6 if hps.problem_type == 'classification' else 1)
            else:
                self.wh = nn.Linear(self.n_feature, 6 if hps.problem_type == 'classification' else 1)
        elif hps.head == 'predictionhead':
            self.wh = PredictionHead(hps, 6 if hps.problem_type == 'classification' else 1)

    def forward(self, graph, graph_c, graph_itvr, bert_input_ids):
        """
        :param graph: [batch_size] * DGLGraph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
            edge:
                word2sent, sent2word:  tffrac=int, type=0
        :param graph_itvr: [batch_size] * DGLGraph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
        :param bert_input_ids: [batch_size, max_positional_length]
        :return: result: [sentnum, 2]
        """

        # return object
        final_return = {'embed': {'before_gat': {}, 'after_gat': {}},
                        'results': {}}

        # word node init
        word_feature = self.set_wnfeature(graph)    # [wnode, embed_size]

        # CEFR node init
        if self._hps.cefr_info == 'graph_init':
            cefr_feature = self.set_cnfeature(graph_c)

        sent_feature = self.n_feature_proj(self.set_snfeature(graph))    # [wnode, 2 * lstm_hidden_state] -> [snode, n_feature_size]
        
        # interviewer prompt as condition for the responses of interviewee
        if self._hps.interviewer:
            itvr_sent_feature = self.n_feature_proj(self.set_snfeature(graph_itvr))
            sent_feature = torch.cat((sent_feature, itvr_sent_feature), dim=1)
        
        # the start state
        word_state = word_feature
        sent_state = self.word2sent(graph, word_feature, sent_feature)

        # Final Return
        final_return['embed']['before_gat'] = {'w': word_feature,
                                               's': sent_state}

        # get baseline
        if self._hps.baseline:
            p = self._get_bert_inputs(bert_input_ids)
            result = self.wh(p)
            w_result = None
            if self._hps.wcefr:
                w_result = self.wwh(word_state)

            # Final Return
            final_return['results'] = {'w': w_result,
                                       's': result}
            return final_return

        for i in range(self._n_iter):
            
            if self._hps.pmi_window_width > -1:
                # sent -> word
                word_state_from_sent = self.sent2word(graph, word_state, sent_state)
                # word -> word
                word_state_from_word = self.word2word(graph, word_state, word_state)
                # cefr -> word
                if self._hps.cefr_info == 'graph_init':
                    word_state_from_cefr = self.cefr2word(graph_c, word_state, cefr_feature)
                word_state = word_state_from_sent + word_state_from_word
                if self._hps.cefr_info == 'graph_init':
                    word_state = word_state + word_state_from_cefr
                # word -> sent
                sent_state = self.word2sent(graph, word_state, sent_state)
            else:
                # sent -> word
                word_state = self.sent2word(graph, word_state, sent_state)
                # cefr -> word
                if self._hps.cefr_info == 'graph_init':
                    word_state_from_cefr = self.cefr2word(graph_c, word_state, cefr_feature)
                    word_state = word_state + word_state_from_cefr
                # word -> sent
                sent_state = self.word2sent(graph, word_state, sent_state)

        # update sent_state
        if self._hps.mean_paragraphs == 'mean_residual':
            mean_sent_state = self._mean_snfeature(graph, sent_state, repeat=True)
            sent_state = torch.cat((sent_state, mean_sent_state), dim=1) # add the information of self-mean
        elif self._hps.mean_paragraphs == 'mean':
            sent_state = self._mean_snfeature(graph, sent_state, repeat=True)
        else:
            sent_state = sent_state

        # BERT encoder
        if self._hps.bert_config is not None:
            p = self._get_bert_inputs(bert_input_ids)

            if self._hps.pred_gated_fusion:
                b_g_w = torch.sigmoid(self.bert_gt_w(torch.cat((sent_state, p), dim=1)))
                bert_state = b_g_w * self.down_bert(p)
            else:
                if self._hps.final_attention:
                    b_sent_state = torch.cat((sent_state, p), dim=1)
                else:
                    sent_state = torch.cat((sent_state, p), dim=1)

        if self._hps.pred_gated_fusion:
            if self._hps.bert_config is not None:
                sent_state = sent_state + bert_state

        if self._hps.final_attention and not self._hps.pred_gated_fusion:
            sent_state = self.final_attn(sent_state, p)

        result = self.wh(sent_state)
        w_result = None
        if self._hps.wcefr:
            w_result = self.wwh(word_state)

        # Final Return
        final_return['embed']['after_gat'] = {'w': word_feature,
                                              's': sent_state}
        final_return['results'] = {'w': w_result,
                                   's': result}

        return final_return

    def _init_sn_param(self):
        self.sent_pos_embed = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(self._hps.doc_max_timesteps + 1, self.embed_size, padding_idx=0),
            freeze=True)
        self.cnn_proj = nn.Linear(self.embed_size, self._hps.n_feature_size)
        self.lstm_hidden_state = self._hps.lstm_hidden_state
        self.lstm = nn.LSTM(self.embed_size, self.lstm_hidden_state, num_layers=self._hps.lstm_layers, dropout=0.1,
                            batch_first=True, bidirectional=self._hps.bidirectional)
        if self._hps.bidirectional:
            self.lstm_proj = nn.Linear(self.lstm_hidden_state * 2, self._hps.n_feature_size)
        else:
            self.lstm_proj = nn.Linear(self.lstm_hidden_state, self._hps.n_feature_size)

        self.ngram_enc = sentEncoder(self._hps, self._embed)

    def _sent_cnn_feature(self, graph, snode_id):
        ngram_feature = self.ngram_enc.forward(graph.nodes[snode_id].data["words"])  # [snode, embed_size]
        graph.nodes[snode_id].data["sent_embedding"] = ngram_feature
        snode_pos = graph.nodes[snode_id].data["position"].view(-1)  # [n_nodes]
        position_embedding = self.sent_pos_embed(snode_pos)
        cnn_feature = self.cnn_proj(ngram_feature + position_embedding)
        return cnn_feature

    def _sent_lstm_feature(self, features, glen):
        pad_seq = rnn.pad_sequence(features, batch_first=True)
        lstm_input = rnn.pack_padded_sequence(pad_seq, glen, batch_first=True)
        lstm_output, _ = self.lstm(lstm_input)
        unpacked, unpacked_len = rnn.pad_packed_sequence(lstm_output, batch_first=True)
        lstm_embedding = [unpacked[i][:unpacked_len[i]] for i in range(len(unpacked))]
        lstm_feature = self.lstm_proj(torch.cat(lstm_embedding, dim=0))  # [n_nodes, n_feature_size]
        return lstm_feature

    def set_wnfeature(self, graph):
        wnode_id = graph.filter_nodes(lambda nodes: nodes.data["unit"]==0)
        wsedge_id = graph.filter_edges(lambda edges: edges.data["dtype"] == 0)   # for word to supernode(sent&doc)
        wid = graph.nodes[wnode_id].data["id"]  # [n_wnodes]
        w_embed = self._embed(wid)  # [n_wnodes, D]
        graph.nodes[wnode_id].data["embed"] = w_embed
        etf = graph.edges[wsedge_id].data["tffrac"]
        graph.edges[wsedge_id].data["tfidfembed"] = self._TFembed(etf)
        if self._hps.pmi_window_width > -1:
            wwedge_id = graph.filter_edges(lambda edges: edges.data["dtype"] == 1)   # for word to word
            eww = graph.edges[wwedge_id].data["tffrac"]
            graph.edges[wwedge_id].data["tfidfembed"] = self._TFembed(eww)
        return w_embed
    
    def set_cnfeature(self, graph_c):        
        cnode_id = graph_c.filter_nodes(lambda nodes: nodes.data["unit"]==1)
        csedge_id = graph_c.filter_edges(lambda edges: edges.data["dtype"]==0)   # for cefr to rely node(cefr&word)
        cid = graph_c.nodes[cnode_id].data["id"]  # [n_wnodes]
        c_embed = self.cefr_embed(cid)  # [n_wnodes, D]
        graph_c.nodes[cnode_id].data["embed"] = c_embed
        etf = graph_c.edges[csedge_id].data["tffrac"]
        graph_c.edges[csedge_id].data["tfidfembed"] = torch.ones(etf.shape[-1] ,self._hps.feat_embed_size).to(self._hps.device)
        return c_embed

    def set_snfeature(self, graph):
        # node feature
        snode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        cnn_feature = self._sent_cnn_feature(graph, snode_id)
        features, glen = get_snode_feat(graph, feat="sent_embedding")
        lstm_feature = self._sent_lstm_feature(features, glen)
        node_feature = torch.cat([cnn_feature, lstm_feature], dim=1)  # [n_nodes, n_feature_size * 2]
        return node_feature

    def _mean_snfeature(self, graph, sent_state, repeat=False):
        repeat_cummulate_list = []
        tensors = []
        glist = dgl.unbatch(graph)
        for j in range(len(glist)):
            g = glist[j]
            snode_id = g.filter_nodes(lambda nodes: nodes.data['dtype'] == 1)
            num_sents = len(snode_id)
            st_idx = j*num_sents
            cur_sent_state = sent_state[st_idx: st_idx + num_sents]

            sent_state_r = cur_sent_state.reshape(1, -1, self._hps.hidden_size)
            sent_state_m = torch.mean(sent_state_r, dim=1)
            repeat_cummulate_list.append(num_sents)
            tensors.append(sent_state_m)

        repeat_cummulate_list = torch.tensor(repeat_cummulate_list).to(self._hps.device)
        if repeat:
            return torch.cat(tensors, dim=0).repeat_interleave(repeat_cummulate_list, dim=0)
        return torch.cat(tensors, dim=0)
    
    def _get_bert_inputs(self, bert_input_ids):
        p = []
        for input_ids in bert_input_ids:
            input_ids = {k: v.to(self.bert_device) for k, v in input_ids.items()}
            self.bert = self.bert.to(self.bert_device)
            bert_output = self.bert(input_ids=input_ids.get('input_ids'),
                                    attention_mask=input_ids.get('attention_mask'),
                                    token_type_ids=input_ids.get('token_type_ids'))
            if self._hps.bert_mp:
                a = mean_pooling(bert_output.get('last_hidden_state').to(self._hps.device), attention_mask=input_ids['attention_mask'].to(self._hps.device))
                b = bert_output.get('pooler_output').to(self._hps.device)
                c = self.bert_pl_linear(torch.cat((a, b), dim=1))
                p.append(c)
            else:
                p.append(bert_output.get('pooler_output').to(self._hps.device))
        return torch.cat(p, dim=0)


class HSumGraph(nn.Module):
    """ without sent2sent and add residual connection """
    def __init__(self, hps, embed):
        """

        :param hps: 
        :param embed: word embedding
        """
        super().__init__()

        self._hps = hps
        self._n_iter = hps.n_iter
        self._embed = embed
        self.embed_size = hps.word_emb_dim

        # BERT encoder
        if hps.bert_config is not None:
            self.bert_device = torch.device("cuda", hps.bert_gpu)
            self.bert = AutoModel.from_config(hps.bert_config).to(self.bert_device)

        # sent node mean
        if hps.mean_paragraphs == 'mean_residual':
            self.m_para_residual_linear = nn.Linear(hps.hidden_size * 2, hps.hidden_size)

        # sent node feature
        self._init_sn_param()
        self._TFembed = nn.Embedding(10, hps.feat_embed_size)   # box=10
        self.n_feature_proj = nn.Linear(hps.n_feature_size * 2, hps.hidden_size, bias=False)

        # word -> sent
        embed_size = hps.word_emb_dim
        self.word2sent = WSWGAT(in_dim=embed_size,
                                out_dim=hps.hidden_size,
                                num_heads=hps.n_head,
                                attn_drop_out=hps.atten_dropout_prob,
                                ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                ffn_drop_out=hps.ffn_dropout_prob,
                                feat_embed_size=hps.feat_embed_size,
                                layerType="W2S"
                                )

        # sent -> word
        self.sent2word = WSWGAT(in_dim=hps.hidden_size,
                                out_dim=embed_size,
                                num_heads=6,
                                attn_drop_out=hps.atten_dropout_prob,
                                ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                ffn_drop_out=hps.ffn_dropout_prob,
                                feat_embed_size=hps.feat_embed_size,
                                layerType="S2W"
                                )
        
        if self._hps.pmi_window_width > -1:
            self.word2word = WSWGAT(in_dim=embed_size,
                                    out_dim=embed_size,
                                    num_heads=10,
                                    attn_drop_out=hps.atten_dropout_prob,
                                    ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                    ffn_drop_out=hps.ffn_dropout_prob,
                                    feat_embed_size=hps.feat_embed_size,
                                    layerType="W2W"
                                    )
        # cefr -> word
        if self._hps.cefr_info == 'graph_init':
            self.cefr_embed = torch.nn.Embedding(8, hps.word_emb_dim, padding_idx=0)
            torch.nn.init.xavier_normal_(self.cefr_embed.weight)
            self.cefr2word = WSWGAT(in_dim=embed_size,
                                    out_dim=embed_size,
                                    num_heads=10,
                                    attn_drop_out=hps.atten_dropout_prob,
                                    ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                    ffn_drop_out=hps.ffn_dropout_prob,
                                    feat_embed_size=hps.feat_embed_size,
                                    layerType="C2W"
                                    )

        self.n_feature = hps.hidden_size

        # sent dimension
        n_sent_dim = self.n_feature
        if hps.mean_paragraphs == 'mean':
            n_sent_dim = n_sent_dim
        elif hps.mean_paragraphs == 'mean_residual':
            n_sent_dim = n_sent_dim * 2

        # interviewer information
        if hps.interviewer:
            self.n_feature_similairty = nn.Linear(hps.hidden_size*2, n_sent_dim, bias=True) # down sampling at the same time

        if hps.pred_gated_fusion:
            # trainable gated weight
            if hps.bert_config is not None:
                self.bert_gt_w = nn.Linear(hps.bert_config.hidden_size + n_sent_dim, 1)
                self.down_bert = nn.Linear(hps.bert_config.hidden_size, n_sent_dim)
            if hps.interviewer:
                self.itvr_gt_w = nn.Linear(n_sent_dim * 2, 1)

            self.n_feature = n_sent_dim
            hps.n_feature = n_sent_dim
            self._hps.n_feature = n_sent_dim
        else:
            final_n_dim = n_sent_dim
            if hps.bert_config is not None:
                final_n_dim = final_n_dim + hps.bert_config.hidden_size
            if hps.interviewer:
                final_n_dim = final_n_dim + n_sent_dim

            self.n_feature = final_n_dim
            hps.n_feature = final_n_dim
            self._hps.n_feature = final_n_dim

        if hps.head == 'linear':
            self.wh = nn.Linear(self.n_feature, 6 if hps.problem_type == 'classification' else 1)
        elif hps.head == 'predictionhead':
            self.wh = PredictionHead(hps, 6 if hps.problem_type == 'classification' else 1)

    def forward(self, graph, graph_c, graph_itvr, bert_input_ids):
        """
        :param graph: [batch_size] * DGLGraph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
            edge:
                word2sent, sent2word:  tffrac=int, type=0
        :param graph_itvr: [batch_size] * DGLGraph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
        :param bert_input_ids: [batch_size, max_positional_length]
        :return: result: [sentnum, 2]
        """

        # word node init
        word_feature = self.set_wnfeature(graph)    # [wnode, embed_size]

        sent_feature = self.n_feature_proj(self.set_snfeature(graph))    # [wnode, 2 * lstm_hidden_state] -> [snode, n_feature_size]
        
        # the start state
        word_state = word_feature
        sent_state = self.word2sent(graph, word_feature, sent_feature)

        for i in range(self._n_iter):
            
            if self._hps.pmi_window_width > -1:
                # sent -> word
                word_state_from_sent = self.sent2word(graph, word_state, sent_state)
                # word -> word
                word_state_from_word = self.word2word(graph, word_state, word_state)
                # cefr -> word
                if self._hps.cefr_info == 'graph_init':
                    word_state_from_cefr = self.cefr2word(graph_c, word_state, cefr_feature)
                word_state = word_state_from_sent + word_state_from_word
                if self._hps.cefr_info == 'graph_init':
                    word_state = word_state + word_state_from_cefr
                # word -> sent
                sent_state = self.word2sent(graph, word_state, sent_state)                
            else:
                # sent -> word
                word_state = self.sent2word(graph, word_state, sent_state)
                # cefr -> word
                if self._hps.cefr_info == 'graph_init':
                    word_state_from_cefr = self.cefr2word(graph_c, word_state, cefr_feature)
                    word_state = word_state + word_state_from_cefr
                # word -> sent
                sent_state = self.word2sent(graph, word_state, sent_state)

        # update sent_state
        if self._hps.mean_paragraphs == 'mean_residual':
            mean_sent_state = self._mean_snfeature(graph, sent_state, repeat=True)
            sent_state = torch.cat((sent_state, mean_sent_state), dim=1) # add the information of self-mean
        elif self._hps.mean_paragraphs == 'mean':
            sent_state = self._mean_snfeature(graph, sent_state, repeat=True)
        else:
            sent_state = sent_state

        # interviewer
        if self._hps.interviewer:
            itvr_sent_feature = self.n_feature_proj(self.set_snfeature(graph_itvr))
            itvr_set_snfeature = self.n_feature_similairty(
                torch.cat((itvr_sent_feature, sent_feature), dim=1)
            ) # similarity information via downsampling, and use the embeddings which have not enter GAT
            
            if self._hps.pred_gated_fusion:
                itvr_g_w = torch.sigmoid(self.itvr_gt_w(torch.cat((itvr_set_snfeature, sent_state), dim=1)))
                itvr_state = itvr_g_w * itvr_set_snfeature
            else:
                sent_state = torch.cat((sent_state, itvr_set_snfeature), dim=1)

        # BERT encoder
        if self._hps.bert_config is not None:
            p = self._get_bert_inputs(bert_input_ids)

            if self._hps.pred_gated_fusion:
                b_g_w = torch.sigmoid(self.bert_gt_w(torch.cat((sent_state, p), dim=1)))
                bert_state = b_g_w * self.down_bert(p)
            else:
                sent_state = torch.cat((sent_state, p), dim=1)

        if self._hps.pred_gated_fusion:
            if self._hps.interviewer:
                sent_state = sent_state + itvr_state
            if self._hps.bert_config is not None:
                sent_state = sent_state + bert_state

        result = self.wh(sent_state)

        if self._hps.oe:
            return result, sent_state

        return result

    def _init_sn_param(self):
        self.sent_pos_embed = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(self._hps.doc_max_timesteps + 1, self.embed_size, padding_idx=0),
            freeze=True)
        self.cnn_proj = nn.Linear(self.embed_size, self._hps.n_feature_size)
        self.lstm_hidden_state = self._hps.lstm_hidden_state
        self.lstm = nn.LSTM(self.embed_size, self.lstm_hidden_state, num_layers=self._hps.lstm_layers, dropout=0.1,
                            batch_first=True, bidirectional=self._hps.bidirectional)
        if self._hps.bidirectional:
            self.lstm_proj = nn.Linear(self.lstm_hidden_state * 2, self._hps.n_feature_size)
        else:
            self.lstm_proj = nn.Linear(self.lstm_hidden_state, self._hps.n_feature_size)

        self.ngram_enc = sentEncoder(self._hps, self._embed)

    def _sent_cnn_feature(self, graph, snode_id):
        ngram_feature = self.ngram_enc.forward(graph.nodes[snode_id].data["words"])  # [snode, embed_size]
        graph.nodes[snode_id].data["sent_embedding"] = ngram_feature
        snode_pos = graph.nodes[snode_id].data["position"].view(-1)  # [n_nodes]
        position_embedding = self.sent_pos_embed(snode_pos)
        cnn_feature = self.cnn_proj(ngram_feature + position_embedding)
        return cnn_feature

    def _sent_lstm_feature(self, features, glen):
        pad_seq = rnn.pad_sequence(features, batch_first=True)
        lstm_input = rnn.pack_padded_sequence(pad_seq, glen, batch_first=True)
        lstm_output, _ = self.lstm(lstm_input)
        unpacked, unpacked_len = rnn.pad_packed_sequence(lstm_output, batch_first=True)
        lstm_embedding = [unpacked[i][:unpacked_len[i]] for i in range(len(unpacked))]
        lstm_feature = self.lstm_proj(torch.cat(lstm_embedding, dim=0))  # [n_nodes, n_feature_size]
        return lstm_feature

    def set_wnfeature(self, graph):
        wnode_id = graph.filter_nodes(lambda nodes: nodes.data["unit"]==0)
        wsedge_id = graph.filter_edges(lambda edges: edges.data["dtype"] == 0)   # for word to supernode(sent&doc)
        wid = graph.nodes[wnode_id].data["id"]  # [n_wnodes]
        w_embed = self._embed(wid)  # [n_wnodes, D]
        graph.nodes[wnode_id].data["embed"] = w_embed
        etf = graph.edges[wsedge_id].data["tffrac"]
        graph.edges[wsedge_id].data["tfidfembed"] = self._TFembed(etf)
        if self._hps.pmi_window_width > -1:
            wwedge_id = graph.filter_edges(lambda edges: edges.data["dtype"] == 1)   # for word to word
            eww = graph.edges[wwedge_id].data["tffrac"]
            graph.edges[wwedge_id].data["tfidfembed"] = self._TFembed(eww)
        return w_embed

    def set_cnfeature(self, graph_c):        
        cnode_id = graph_c.filter_nodes(lambda nodes: nodes.data["unit"]==1)
        csedge_id = graph_c.filter_edges(lambda edges: edges.data["dtype"]==0)   # for cefr to rely node(cefr&word)
        cid = graph_c.nodes[cnode_id].data["id"]  # [n_wnodes]
        c_embed = self.cefr_embed(cid)  # [n_wnodes, D]
        graph_c.nodes[cnode_id].data["embed"] = c_embed
        etf = graph_c.edges[csedge_id].data["tffrac"]
        graph_c.edges[csedge_id].data["tfidfembed"] = torch.ones(etf.shape[-1] ,self._hps.feat_embed_size).to(self._hps.device)
        return c_embed

    def set_snfeature(self, graph):
        # node feature
        snode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        cnn_feature = self._sent_cnn_feature(graph, snode_id)
        features, glen = get_snode_feat(graph, feat="sent_embedding")
        lstm_feature = self._sent_lstm_feature(features, glen)
        node_feature = torch.cat([cnn_feature, lstm_feature], dim=1)  # [n_nodes, n_feature_size * 2]
        return node_feature

    def _mean_snfeature(self, graph, sent_state, repeat=False):
        repeat_cummulate_list = []
        tensors = []
        glist = dgl.unbatch(graph)
        for j in range(len(glist)):
            g = glist[j]
            snode_id = g.filter_nodes(lambda nodes: nodes.data['dtype'] == 1)
            num_sents = len(snode_id)
            st_idx = j*num_sents
            cur_sent_state = sent_state[st_idx: st_idx + num_sents]

            sent_state_r = cur_sent_state.reshape(1, -1, self._hps.hidden_size)
            sent_state_m = torch.mean(sent_state_r, dim=1)
            repeat_cummulate_list.append(num_sents)
            tensors.append(sent_state_m)

        repeat_cummulate_list = torch.tensor(repeat_cummulate_list).to(self._hps.device)
        if repeat:
            return torch.cat(tensors, dim=0).repeat_interleave(repeat_cummulate_list, dim=0)
        return torch.cat(tensors, dim=0)
    
    def _get_bert_inputs(self, bert_input_ids):
        p = []
        for input_ids in bert_input_ids:
            input_ids = {k: v.to(self.bert_device) for k, v in input_ids.items()}
            self.bert = self.bert.to(self.bert_device)
            bert_output = self.bert(input_ids=input_ids.get('input_ids'),
                                    attention_mask=input_ids.get('attention_mask'),
                                    token_type_ids=input_ids.get('token_type_ids'))
            p.append(bert_output.get('pooler_output').to(self._hps.device))
        return torch.cat(p, dim=0)
        


class HSumDocGraph(HSumGraph):
    """
        without sent2sent and add residual connection
        add Document Nodes
    """

    def __init__(self, hps, embed):
        super().__init__(hps, embed)
        self.dn_feature_proj = nn.Linear(hps.hidden_size, hps.hidden_size, bias=False)
        self.wh = nn.Linear(self.n_feature * 2, 6)

    def forward(self, graph):
        """
        :param graph: [batch_size] * DGLGraph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
                document: unit=1, dtype=2
            edge:
                word2sent, sent2word: tffrac=int, type=0
                word2doc, doc2word: tffrac=int, type=0
                sent2doc: type=2
        :return: result: [sentnum, 2]
        """

        snode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        dnode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)
        supernode_id = graph.filter_nodes(lambda nodes: nodes.data["unit"] == 1)

        # word node init
        word_feature = self.set_wnfeature(graph)    # [wnode, embed_size]
        sent_feature = self.n_feature_proj(self.set_snfeature(graph))    # [snode, n_feature_size]

        # sent and doc node init
        graph.nodes[snode_id].data["init_feature"] = sent_feature
        doc_feature, snid2dnid = self.set_dnfeature(graph)
        doc_feature = self.dn_feature_proj(doc_feature)
        graph.nodes[dnode_id].data["init_feature"] = doc_feature

        # the start state
        word_state = word_feature
        sent_state = graph.nodes[supernode_id].data["init_feature"]
        sent_state = self.word2sent(graph, word_state, sent_state)

        for i in range(self._n_iter):
            # sent -> word
            word_state = self.sent2word(graph, word_state, sent_state)
            # word -> sent
            sent_state = self.word2sent(graph, word_state, sent_state)

        graph.nodes[supernode_id].data["hidden_state"] = sent_state

        # extract sentence nodes
        s_state_list = []
        for snid in snode_id:
            d_state = graph.nodes[snid2dnid[int(snid)]].data["hidden_state"]
            s_state = graph.nodes[snid].data["hidden_state"]
            s_state = torch.cat([s_state, d_state], dim=-1)
            s_state_list.append(s_state)

        s_state = torch.cat(s_state_list, dim=0)
        result = self.wh(s_state)
        return result


    def set_dnfeature(self, graph):
        """ init doc node by mean pooling on the its sent node (connected by the edges with type=1) """
        dnode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)
        node_feature_list = []
        snid2dnid = {}
        for dnode in dnode_id:
            snodes = [nid for nid in graph.predecessors(dnode) if graph.nodes[nid].data["dtype"]==1]
            doc_feature = graph.nodes[snodes].data["init_feature"].mean(dim=0)
            assert not torch.any(torch.isnan(doc_feature)), "doc_feature_element"
            node_feature_list.append(doc_feature)
            for s in snodes:
                snid2dnid[int(s)] = dnode
        node_feature = torch.stack(node_feature_list)
        return node_feature, snid2dnid


def get_snode_feat(G, feat):
    glist = dgl.unbatch(G)
    feature = []
    glen = []
    for g in glist:
        snode_id = g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        feature.append(g.nodes[snode_id].data[feat])
        glen.append(len(snode_id))
    return feature, glen

def get_pnode_feat(G, feat):
    glist = dgl.unbatch(G)
    feature = []
    glen = []
    for g in glist:
        pnode_id = g.filter_nodes(lambda nodes: nodes.data["dtype"] == 0)
        feature.append(g.nodes[pnode_id].data[feat])
        glen.append(len(pnode_id))
    return feature, glen
