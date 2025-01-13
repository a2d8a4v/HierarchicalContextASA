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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

import dgl

from collections import OrderedDict

# from module.GAT import GAT, GAT_ffn
from module.Encoder import sentEncoder, sentLenEncoder
from module.GAT import WSWGAT, WPWGAT, SPSGAT, PWPGAT, WSGRGAT, SPOEGAT
from module.Attention import SelfAttention
from module.PositionEmbedding import get_sinusoid_encoding_table
from module.Decoder import Seq2seqDecoder
from module.RGAT import RGAT, SWSRGAT, AWGRGAT

# from transformers import AutoModel
from module.bert.modeling_bert import BertModel
from module.roberta.modeling_roberta import RobertaModel
from module.longformer.modeling_longformer import LongformerModel, LongformerSelfAttention
from transformers import LongformerConfig

# from peft import get_peft_model, LoraConfig, TaskType


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


class SimilarityHead(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimilarityHead, self).__init___()
        # initialing nn layers
        hidden_size = input_size
        self.W_for_add = nn.Linear(input_size, hidden_size, bias=False)
        self.W_lat_add = nn.Linear(input_size, hidden_size, bias=False)
        self.W_for_min = nn.Linear(input_size, hidden_size, bias=False)
        self.W_lat_min = nn.Linear(input_size, hidden_size, bias=False)
        self.W_mul = nn.Linear(input_size, hidden_size, bias=False)
        self.W_dot = nn.Linear(input_size, hidden_size, bias=False)
    
    def forward(self, vec1, vec2):
        
        # Add
        vec_for_add = self.W_for_add(vec1)
        vec_lat_add = self.W_lat_add(vec2)
        s_add = F.tanh(vec_for_add+vec_lat_add)

        # Sub
        vec_for_min = self.W_for_min(vec1)
        vec_lat_min = self.W_lat_min(vec2)
        s_min = F.tanh(vec_for_min-vec_lat_min)
        
        # Mul
        s_mul = vec1 * self.W_mul(vec2)
        
        # Dot
        s_dot = F.tanh(self.W_dot(vec1 * vec2))


class WeightedSumLayer(nn.Module):
    def __init__(self, dimension):
        super(WeightedSumLayer, self).__init__()
        # A simple linear layer to compute attention weights
        self.attention_weights_layer = nn.Linear(dimension, 1)

    def forward(self, x):
        # x shape: [batch_size, sequence_length, dimension]

        # Compute attention weights
        # After applying the layer, reshape to [batch_size, sequence_length, 1] for broadcasting
        attention_weights = self.attention_weights_layer(x).softmax(dim=1)

        # Apply attention weights
        weighted_x = x * attention_weights

        # Sum over the sequence length
        sum_x = torch.sum(weighted_x, dim=0)
        sum_x = sum_x.unsqueeze(0)

        return sum_x

class FusionFinalHead(nn.Module):
    def __init__(self, bert_input_dim, graph_input_dim, output_dim, hidden_state_dim, num_heads):
        super(FusionFinalHead, self).__init__()
        self.num_heads = num_heads
        self.bert_input_dim = bert_input_dim
        self.graph_input_dim = graph_input_dim
        self.hidden_state_dim = hidden_state_dim
        self.bert_proj = nn.Linear(bert_input_dim, hidden_state_dim)
        self.graph_proj = nn.Linear(graph_input_dim, hidden_state_dim)
        self.attention_heads = nn.ModuleList([nn.Linear(hidden_state_dim * 2, hidden_state_dim) for _ in range(num_heads)])
        num_combination = 0
        for i in range(num_heads):
            for j in range(i+1, num_heads):
                num_combination += 1
        self.output_dense = nn.Linear(hidden_state_dim * num_heads * num_combination, output_dim)

    def forward(self, tensors_dict):
        tensors = []
        combined_tensors = []
        assert self.num_heads == len(tensors_dict)
        for n, v in tensors_dict.items():
            if n == 'b':
                tensors.append(self.bert_proj(v))
            if n in ['hcg', 'sdg', 'acg']:
                tensors.append(self.graph_proj(v))
        for i in range(self.num_heads):
            for j in range(i+1, self.num_heads):
                # Concatenate pair of tensors
                combined = torch.cat([tensors[i], tensors[j]], dim=1)
                # Apply attention head to each pair
                for head in self.attention_heads:
                    combined_tensors.append(F.relu(head(combined)))

        # Concatenate outputs from all heads
        combined_output = self.output_dense(torch.cat(combined_tensors, dim=1))
        return combined_output

class HierConGraph(nn.Module):
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
            # self.bert = AutoModel.from_config(hps.bert_config)
            self.bert, bin_file_path = automodel(hps, hps.bert_config, hps.bert_roberta_to_long)
            if hps.bert_config.model_type in ['roberta', 'longformer']:
                self.bert.config.type_vocab_size = 2
                self.bert.embeddings.token_type_embeddings = nn.Embedding(2, hps.bert_config.hidden_size) # DEBUG
                self.bert.embeddings.token_type_embeddings.weight.data.normal_(mean=0.0, std=hps.bert_config.initializer_range) # DEBUG
            if hps.bert_roberta_to_long:
                attention_window = [
                    512,
                    512,
                    512,
                    512,
                    512,
                    512,
                    512,
                    512,
                    512,
                    512,
                    512,
                    512
                ]
                # self.bert = change_roberta_to_long_input_model(self.bert, attention_window, hps.sent_max_len)
                self.bert = change_roberta_to_long_input_model(self.bert,
                                                               hps.tokenizer, 
                                                               hps.bert_config, 
                                                               attention_window=attention_window, 
                                                               longformer_max_length=hps.sent_max_len
                                                               )
            self.initial_bert_param(bin_file_path, hps.bert_config.model_type)
            # if hps.bert_train_peft:
            #     peft_config = LoraConfig(
            #         task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
            #     )
            #     self.bert = get_peft_model(self.bert, peft_config)
            
            # debug
            # self.can_convs_bert = nn.Linear(hps.bert_config.hidden_size, hps.hidden_size * 2)
            # self.int_convs_bert = nn.Linear(hps.bert_config.hidden_size, hps.hidden_size * 2)
            self.can_convs_bert = nn.Linear(hps.bert_config.hidden_size + hps.hidden_size * 2, hps.hidden_size * 2)
            self.int_convs_bert = nn.Linear(hps.bert_config.hidden_size + hps.hidden_size * 2, hps.hidden_size * 2)
            self.seg_sent_pooling = WeightedSumLayer(hps.bert_config.hidden_size)

        # sent node feature
        self._init_sn_param()
        self._TFembed = nn.Embedding(10, hps.feat_embed_size)   # box=10
        self.n_feature_proj = nn.Linear(hps.n_feature_size * 2, hps.hidden_size * 2, bias=False)

        # word -> sent
        embed_size = hps.word_emb_dim
        sent_hidden_size = hps.hidden_size * 2
        self.word2sent = WSWGAT(in_dim=embed_size,
                                out_dim=sent_hidden_size,
                                num_heads=hps.n_head,
                                attn_drop_out=hps.atten_dropout_prob,
                                ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                ffn_drop_out=hps.ffn_dropout_prob,
                                feat_embed_size=hps.feat_embed_size,
                                des_embed_size=sent_hidden_size,
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
                                des_embed_size=embed_size,
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
                                    des_embed_size=embed_size,
                                    layerType="W2W"
                                    )
        self.att_pl_hec_global_node = WeightedSumLayer(sent_hidden_size)

        # cefr -> word
        if self._hps.cefr_info == 'graph_init':
            self.cefr_embed = torch.nn.Embedding(8, hps.word_emb_dim, padding_idx=0)
            torch.nn.init.xavier_normal_(self.cefr_embed.weight)
            self.word2cefr = WSWGAT(in_dim=embed_size,
                                    out_dim=embed_size,
                                    num_heads=10,
                                    attn_drop_out=hps.atten_dropout_prob,
                                    ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                    ffn_drop_out=hps.ffn_dropout_prob,
                                    feat_embed_size=hps.feat_embed_size,
                                    des_embed_size=embed_size,
                                    layerType="W2C"
                                    )
            self.cefr2word = WSWGAT(in_dim=embed_size,
                                    out_dim=embed_size,
                                    num_heads=10,
                                    attn_drop_out=hps.atten_dropout_prob,
                                    ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                    ffn_drop_out=hps.ffn_dropout_prob,
                                    feat_embed_size=hps.feat_embed_size,
                                    des_embed_size=embed_size,
                                    layerType="C2W"
                                    )
        self.wordsent2global = WSGRGAT(in_dim=sent_hidden_size,
                                out_dim=sent_hidden_size,
                                num_heads=8,
                                attn_drop_out=hps.atten_dropout_prob,
                                ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                ffn_drop_out=hps.ffn_dropout_prob,
                                feat_embed_size=hps.feat_embed_size,
                                des_embed_size=sent_hidden_size,
                                layerType='WS2G'
                                )
        # global node constructing function
        self.word2sent_proj = nn.Linear(embed_size, sent_hidden_size)

        # Subject, Predicate, Object, Entity
        # TODO: find the best method to do propagate
        if self._hps.pred_method == 'test_wdc':
            self.spoe_embed_size = hps.spo_hidden_size
            self.subject2predicate = SPOEGAT(in_dim=self.spoe_embed_size,
                                        out_dim=self.spoe_embed_size,
                                        num_heads=8,
                                        attn_drop_out=hps.atten_dropout_prob,
                                        ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                        ffn_drop_out=hps.ffn_dropout_prob,
                                        feat_embed_size=hps.feat_embed_size,
                                        des_embed_size=self.spoe_embed_size,
                                        layerType='ES2P'
                                        )
            self.object2predicate = SPOEGAT(in_dim=self.spoe_embed_size,
                                        out_dim=self.spoe_embed_size,
                                        num_heads=8,
                                        attn_drop_out=hps.atten_dropout_prob,
                                        ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                        ffn_drop_out=hps.ffn_dropout_prob,
                                        feat_embed_size=hps.feat_embed_size,
                                        des_embed_size=self.spoe_embed_size,
                                        layerType='EO2P'
                                        )
            self.predicate2subject = SPOEGAT(in_dim=self.spoe_embed_size,
                                        out_dim=self.spoe_embed_size,
                                        num_heads=8,
                                        attn_drop_out=hps.atten_dropout_prob,
                                        ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                        ffn_drop_out=hps.ffn_dropout_prob,
                                        feat_embed_size=hps.feat_embed_size,
                                        des_embed_size=self.spoe_embed_size,
                                        layerType='EP2S'
                                        )
            self.object2subject = SPOEGAT(in_dim=self.spoe_embed_size,
                                        out_dim=self.spoe_embed_size,
                                        num_heads=8,
                                        attn_drop_out=hps.atten_dropout_prob,
                                        ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                        ffn_drop_out=hps.ffn_dropout_prob,
                                        feat_embed_size=hps.feat_embed_size,
                                        des_embed_size=self.spoe_embed_size,
                                        layerType='EO2S'
                                        )
            self.predicate2object = SPOEGAT(in_dim=self.spoe_embed_size,
                                        out_dim=self.spoe_embed_size,
                                        num_heads=8,
                                        attn_drop_out=hps.atten_dropout_prob,
                                        ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                        ffn_drop_out=hps.ffn_dropout_prob,
                                        feat_embed_size=hps.feat_embed_size,
                                        des_embed_size=self.spoe_embed_size,
                                        layerType='EP2O'
                                        )
            self.subject2object = SPOEGAT(in_dim=self.spoe_embed_size,
                                        out_dim=self.spoe_embed_size,
                                        num_heads=8,
                                        attn_drop_out=hps.atten_dropout_prob,
                                        ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                        ffn_drop_out=hps.ffn_dropout_prob,
                                        feat_embed_size=hps.feat_embed_size,
                                        des_embed_size=self.spoe_embed_size,
                                        layerType='ES2O'
                                        )
            self.spo2entity = SPOEGAT(in_dim=self.spoe_embed_size,
                                        out_dim=self.spoe_embed_size,
                                        num_heads=8,
                                        attn_drop_out=hps.atten_dropout_prob,
                                        ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                        ffn_drop_out=hps.ffn_dropout_prob,
                                        feat_embed_size=hps.feat_embed_size,
                                        des_embed_size=self.spoe_embed_size,
                                        layerType='ESPO2E'
                                        )
            # TODO: add glove
            self.att_pl_oie_s_bert = WeightedSumLayer(hps.bert_config.hidden_size)
            self.att_pl_oie_p_bert = WeightedSumLayer(hps.bert_config.hidden_size)
            self.att_pl_oie_o_bert = WeightedSumLayer(hps.bert_config.hidden_size)
            self.cons_bert2oie_s = nn.Linear(hps.bert_config.hidden_size, self.spoe_embed_size)
            self.cons_bert2oie_p = nn.Linear(hps.bert_config.hidden_size, self.spoe_embed_size)
            self.cons_bert2oie_o = nn.Linear(hps.bert_config.hidden_size, self.spoe_embed_size)
            # self.cons_glv2oie_s = nn.Linear(embed_size, self.spoe_embed_size)
            # self.cons_glv2oie_p = nn.Linear(embed_size, self.spoe_embed_size)
            # self.cons_glv2oie_o = nn.Linear(embed_size, self.spoe_embed_size)
            self.cons_woie2sent_size = nn.Linear(sent_hidden_size + hps.spo_hidden_size, sent_hidden_size)
            self.cons_woie2rsent_size = nn.Linear(sent_hidden_size + hps.spo_hidden_size, sent_hidden_size)
        elif self._hps.pred_method in ['test_wdc2', 'test_wdc3', 'acg', 'hsag']:
            self.spoe_embed_size = hps.spo_hidden_size
            self.spo2spo = SPOEGAT(in_dim=self.spoe_embed_size,
                                        out_dim=self.spoe_embed_size,
                                        num_heads=8,
                                        attn_drop_out=hps.atten_dropout_prob,
                                        ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                        ffn_drop_out=hps.ffn_dropout_prob,
                                        feat_embed_size=hps.feat_embed_size,
                                        des_embed_size=self.spoe_embed_size,
                                        layerType='ESPO2SPO'
                                        )
            self.spo2entity = SPOEGAT(in_dim=self.spoe_embed_size,
                                        out_dim=self.spoe_embed_size,
                                        num_heads=8,
                                        attn_drop_out=hps.atten_dropout_prob,
                                        ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                        ffn_drop_out=hps.ffn_dropout_prob,
                                        feat_embed_size=hps.feat_embed_size,
                                        des_embed_size=self.spoe_embed_size,
                                        layerType='ESPO2E'
                                        )
            # TODO: add glove
            self.att_pl_oie_s_bert = WeightedSumLayer(hps.bert_config.hidden_size)
            self.att_pl_oie_p_bert = WeightedSumLayer(hps.bert_config.hidden_size)
            self.att_pl_oie_o_bert = WeightedSumLayer(hps.bert_config.hidden_size)
            self.cons_bert2oie_s = nn.Linear(hps.bert_config.hidden_size, self.spoe_embed_size)
            self.cons_bert2oie_p = nn.Linear(hps.bert_config.hidden_size, self.spoe_embed_size)
            self.cons_bert2oie_o = nn.Linear(hps.bert_config.hidden_size, self.spoe_embed_size)
            self.cons_woie2sent_size = nn.Linear(sent_hidden_size + hps.spo_hidden_size, sent_hidden_size)
            self.cons_woie2rsent_size = nn.Linear(sent_hidden_size + hps.spo_hidden_size, sent_hidden_size)
            
            if self._hps.pred_method in ['test_wdc3', 'acg', 'hsag']:
                self.entity2sent = SPOEGAT(in_dim=self.spoe_embed_size,
                                        out_dim=sent_hidden_size,
                                        num_heads=8,
                                        attn_drop_out=hps.atten_dropout_prob,
                                        ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                        ffn_drop_out=hps.ffn_dropout_prob,
                                        feat_embed_size=hps.feat_embed_size,
                                        des_embed_size=sent_hidden_size,
                                        layerType='ENTITY2SENT'
                                        )
                self.sent2entity = SPOEGAT(in_dim=sent_hidden_size,
                                        out_dim=self.spoe_embed_size,
                                        num_heads=8,
                                        attn_drop_out=hps.atten_dropout_prob,
                                        ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                        ffn_drop_out=hps.ffn_dropout_prob,
                                        feat_embed_size=hps.feat_embed_size,
                                        des_embed_size=self.spoe_embed_size,
                                        layerType='SENT2ENTITY'
                                        )
                self.sententity2global = SPOEGAT(in_dim=sent_hidden_size,
                                        out_dim=sent_hidden_size,
                                        num_heads=8,
                                        attn_drop_out=hps.atten_dropout_prob,
                                        ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                        ffn_drop_out=hps.ffn_dropout_prob,
                                        feat_embed_size=hps.feat_embed_size,
                                        des_embed_size=sent_hidden_size,
                                        layerType='SENTENTITY2GLOBAL'
                                        )
                self.att_pl_ac_global_node = WeightedSumLayer(sent_hidden_size)
                self.cons_entity_sent = nn.Linear(self.spoe_embed_size, sent_hidden_size)

        # # paragraph --> sent 
        # self.paragraph2sent = SPSGAT(in_dim=para_hidden_size,
        #                         out_dim=sent_hidden_size,
        #                         num_heads=1,
        #                         attn_drop_out=hps.atten_dropout_prob,
        #                         ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
        #                         ffn_drop_out=hps.ffn_dropout_prob,
        #                         feat_embed_size=hps.feat_embed_size,
        #                         layerType="P2S"
        #                         )

        # # word -> paragraph
        # if hps.retain_wp_relation:
        #     self.word2paragraph = WPWGAT(in_dim=embed_size,
        #                             out_dim=para_hidden_size,
        #                             num_heads=1,
        #                             attn_drop_out=hps.atten_dropout_prob,
        #                             ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
        #                             ffn_drop_out=hps.ffn_dropout_prob,
        #                             feat_embed_size=hps.feat_embed_size,
        #                             layerType="W2P"
        #                             )
        #     self.paragraph2word = PWPGAT(in_dim=para_hidden_size,
        #                             out_dim=embed_size,
        #                             num_heads=1,
        #                             attn_drop_out=hps.atten_dropout_prob,
        #                             ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
        #                             ffn_drop_out=hps.ffn_dropout_prob,
        #                             feat_embed_size=hps.feat_embed_size,
        #                             layerType="P2W"
        #                             )

        # # sent -> paragraph
        # self.sent2paragraph = SPSGAT(in_dim=sent_hidden_size,
        #                         out_dim=para_hidden_size,
        #                         num_heads=1,
        #                         attn_drop_out=hps.atten_dropout_prob,
        #                         ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
        #                         ffn_drop_out=hps.ffn_dropout_prob,
        #                         feat_embed_size=hps.feat_embed_size,
        #                         layerType="S2P"
        #                         )


        # relation graph convolution network
        self.rel_names = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
        self.rel_names.sort()
        num_bases = 17
        if num_bases < 0 or num_bases > len(self.rel_names):
            self.num_bases = len(self.rel_names)
        else:
            self.num_bases = num_bases

        self.sent2relation = SWSRGAT(in_dim=sent_hidden_size,
                                out_dim=sent_hidden_size,
                                num_heads=8,
                                attn_drop_out=hps.atten_dropout_prob,
                                ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                ffn_drop_out=hps.ffn_dropout_prob,
                                feat_embed_size=hps.feat_embed_size,
                                des_embed_size=sent_hidden_size,
                                layerType='S2R'
                                )
        self.relation2sent = SWSRGAT(in_dim=sent_hidden_size,
                                out_dim=sent_hidden_size,
                                num_heads=8,
                                attn_drop_out=hps.atten_dropout_prob,
                                ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                ffn_drop_out=hps.ffn_dropout_prob,
                                feat_embed_size=hps.feat_embed_size,
                                des_embed_size=sent_hidden_size,
                                layerType='R2S'
                                )
        self.sentrel2global = AWGRGAT(in_dim=sent_hidden_size,
                                out_dim=sent_hidden_size,
                                num_heads=8,
                                attn_drop_out=hps.atten_dropout_prob,
                                ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                ffn_drop_out=hps.ffn_dropout_prob,
                                feat_embed_size=hps.feat_embed_size,
                                des_embed_size=sent_hidden_size,
                                layerType='A2G'
                                )
        self.att_pl_sd_global_node = WeightedSumLayer(sent_hidden_size)
        self.relation_nodes_embeddings = nn.Embedding(num_bases, sent_hidden_size)
        self.relation_nodes_embeddings.weight.data.uniform_(-0.1, 0.1)

        self.pwh = nn.Linear(hps.bert_config.hidden_size if hps.baseline else hps.hidden_size *2,
                             6 if hps.problem_type == 'classification' else 1
                            )

        # Fusion mechanism
        if hps.pred_method in ['all_s', 'test_wdc', 'test_wdc2', 'hsag']:
            self.fuse_muliheadatt = FusionFinalHead(hps.bert_config.hidden_size, sent_hidden_size, hps.hidden_size *2, hps.hidden_size, 3)
            
        if hps.pred_method in ['test_wdc3']:
            self.fuse_muliheadatt = FusionFinalHead(hps.bert_config.hidden_size, sent_hidden_size, hps.hidden_size *2, hps.hidden_size, 4)

        if hps.pred_method in ['sde_s', 'hec_s', 'test_wdc', 'test_wdc2', 'test_wdc3']:
            self.bert_gt_w = nn.Linear(hps.bert_config.hidden_size + sent_hidden_size, 1)
            self.down_bert = nn.Linear(hps.bert_config.hidden_size, sent_hidden_size)

        if hps.pred_method in ['ehg']:
            self.ssss = nn.Linear(hps.hidden_size *4, hps.hidden_size *2)

        if hps.pred_method in ['ehg_ldc', 'hec_ldc', 'sde_ldc', 'all_ldc']:
            self.dd = nn.Linear(sent_hidden_size*5, 1)
            self.down_bert = nn.Linear(hps.bert_config.hidden_size, sent_hidden_size)

        if hps.baseline:
            self.final_dropout = nn.Dropout(p=0.5)

    def initial_bert_param(self, bin_file_path, model_type):
        state_dict = torch.load(bin_file_path, map_location=torch.device('cpu'))
        # if from ForMLM
        if list(state_dict.keys())[-1] == 'lm_head.decoder.bias':
            state_dict = {k.replace('longformer', 'roberta'): v for k, v in state_dict.items() if 'longformer' in k}
        for name, param in self.bert.named_parameters():
            fixed_name = model_type+'.'+name
            if fixed_name in state_dict:
                param = state_dict.get(fixed_name)

    def forward(self, data):
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

        # obtain features        
        graph = data.get('G')
        graph.to(self._hps.device)
        graph_c = data.get('G_c')
        if self._hps.cefr_info == 'graph_init':
            graph_c.to(self._hps.device)
        graph_itvr = data.get('itvr_G')
        graph_itvr.to(self._hps.device)
        M_G = data.get('M_G')
        if self._hps.language_use:
            G_PDG = data.get('G_PDG')
        M_G.to(self._hps.device)
        if self._hps.pred_method in ['test_wdc', 'test_wdc2', 'test_wdc3', 'acg', 'hsag']:
            DA_G = data.get('DA_G')
            DA_G.to(self._hps.device) # out of memory
        if self._hps.pred_method in ['test_wdc3', 'acg', 'hsag']:
            DSA_G = data.get('DSA_G')
            DSA_G.to(self._hps.device) # out of memory
        ie_count = data.get('ie_count')
        ir_count = data.get('ir_count')
        bert_sent_boundary = data.get('bert_sent_boundary')
        oie_in_idx = data.get('oie_in_idx')
        spoe2nid_dict = data.get('spoe2nid_dict')
        oie_alng_glv_oieseq = data.get('oie_alng_glv_oieseq')
        oie_alng_bert_oieseq = data.get('oie_alng_bert_oieseq')
        conversation_embed = self._get_bert_embed(data.get('bert_feature'), specific_key=['input_ids', 'attention_mask', 'token_type_ids'])

        # return object
        final_return = {'embed': {'before_gat': {}, 'after_gat': {}},
                        'dec_outputs': {},
                        'results': {}}

        # get baseline
        if self._hps.baseline:
            result = None

            p_result = self.pwh(self.final_dropout(conversation_embed.get('pooler_output')))
            w_result = None

            # Final Return
            final_return['record'] = {'M_G': None,
                                    'ie': None,
                                    'ir': None}
            final_return['embed']['after_gat'] = {'w': None,
                                                's': None,
                                                'p': conversation_embed.get('pooler_output')}
            final_return['results'] = {'w': w_result,
                                    's': result,
                                    'p': p_result}
            
            return final_return

        # get fine-grained embedding from boundary
        # we utilize `last_hidden_state`, not `pooler_output` in order to obtaining fine-grained contextualized information from short distance context.
        can_int_embed_dict = self._get_pooled_embed(data.get('bert_feature'), bert_sent_boundary, conversation_embed.get('last_hidden_state'))

        # word node init
        word_feature = self.set_wnfeature(graph)    # [wnode, embed_size]

        # CEFR node init
        if self._hps.cefr_info == 'graph_init':
            cefr_feature = self.set_cnfeature(graph_c)

        sent_feature = self.n_feature_proj(self.set_snfeature(graph))    # [wnode, 2 * lstm_hidden_state] -> [snode, n_feature_size]
        
        # interviewer prompt as condition for the responses of interviewee
        r_sent_feature = self.n_feature_proj(self.set_snfeature(graph_itvr))
        
        # pos, dep, ged
        # TODO
        if self._hps.language_use:
            pos_feature = self.set_posfeature(G_PDG)
            dep_feature = self.set_depfeature(G_PDG)
            ged_feature = self.set_gedfeature(G_PDG)

        if self._hps.bert_config is not None:
            can_semantic_embed = can_int_embed_dict['can']
            int_semantic_embed = can_int_embed_dict['int']
            # TODO: concat then project
            # can_semantic_embed = self.can_convs_bert(can_semantic_embed)
            # int_semantic_embed = self.int_convs_bert(int_semantic_embed)
            
            can_semantic_embed = self.can_convs_bert(torch.cat((sent_feature, can_semantic_embed), dim=-1))
            int_semantic_embed = self.int_convs_bert(torch.cat((r_sent_feature, int_semantic_embed), dim=-1))
            sent_feature = sent_feature + can_semantic_embed
            r_sent_feature = r_sent_feature + int_semantic_embed
            # can_semantic_embed = self.can_convs_bert(can_semantic_embed)
            # int_semantic_embed = self.int_convs_bert(int_semantic_embed)
            # sent_feature = can_semantic_embed
            # r_sent_feature = int_semantic_embed
            

        # the start state
        word_state = word_feature
        sent_state = self.word2sent(graph, word_feature, sent_feature)
        r_sent_state = self.word2sent(graph, word_feature, r_sent_feature)

        # fine-grained act modeling within utterances
        if self._hps.pred_method == 'test_wdc':
            DA_G, subject_state, predicate_state, object_state, entity_state, record_spo_length_g_dict, entity2lineid = self.set_spoenfeature(DA_G, sent_state, r_sent_state, ie_count, ir_count, conversation_embed, oie_alng_bert_oieseq, oie_alng_glv_oieseq, bert_sent_boundary, oie_in_idx, spoe2nid_dict)
            for i in range(self._n_iter):
                if predicate_state.shape[0] > 0:
                    predicate_state_from_subject = self.subject2predicate(DA_G, subject_state, predicate_state)
                    predicate_state_from_object = self.object2predicate(DA_G, object_state, predicate_state)
                    predicate_state = predicate_state_from_subject + predicate_state_from_object
                if subject_state.shape[0] > 0:
                    subject_state_from_predicate = self.predicate2subject(DA_G, predicate_state, subject_state)
                    subject_state_from_object = self.object2subject(DA_G, object_state, subject_state)
                    subject_state = subject_state_from_predicate + subject_state_from_object
                if object_state.shape[0] > 0:
                    object_state_from_predicate = self.predicate2object(DA_G, predicate_state, object_state)
                    object_state_from_subject = self.subject2object(DA_G, subject_state, object_state)
                    object_state = object_state_from_predicate + object_state_from_subject
                # reconstruct the spo_state from subject_state, predicate_state, object_state
                js, jp, jo = 0, 0, 0
                spo_state = []
                for j, j_data in record_spo_length_g_dict.items():
                    if subject_state.shape[0] > 0:
                        subject_state_c = subject_state[js:js+j_data['S']]
                    if predicate_state.shape[0] > 0:
                        predicate_state_c = predicate_state[jp:jp+j_data['P']]
                    if object_state.shape[0] > 0:
                        object_state_c = object_state[jo:jo+j_data['O']]
                    js += j_data['S']
                    jp += j_data['P']
                    jo += j_data['O']
                    if subject_state.shape[0] > 0:
                        spo_state.append(subject_state_c)
                    if predicate_state.shape[0] > 0:
                        spo_state.append(predicate_state_c)
                    if object_state.shape[0] > 0:
                        spo_state.append(object_state_c)
                spo_state = torch.cat(spo_state, dim=0)
                entity_state = self.spo2entity(DA_G, spo_state, entity_state)

            enode_split = [j_data['E'] for j, j_data in record_spo_length_g_dict.items()]
            entity2lineid_splited = entity2lineid.split(enode_split)
            entity_state_splited = entity_state.split(enode_split)

            r_sent_state_expand = torch.zeros((r_sent_state.shape[0], self.spoe_embed_size)).to(self._hps.device)
            sent_state_expand = torch.zeros((sent_state.shape[0], self.spoe_embed_size)).to(self._hps.device)
            batched_sent_state_expand = sent_state_expand.split(ie_count)
            batched_r_sent_state_expand = r_sent_state_expand.split(ir_count)
            # update sent_state and r_sent_state
            # we build our task under intelocutor going first and candidate following scheme
            pack_batched_sent_state_expand = []
            pack_batched_r_sent_state_expand = []
            for (k, v, ss_e, rss_e) in zip(entity2lineid_splited, entity_state_splited, batched_sent_state_expand, batched_r_sent_state_expand):
                for i, j in enumerate(k): # k is the target line_id list of conversation 
                    j = j.item()
                    jj = j//2+j%2-1
                    if j%2 == 0:
                        rss_e_c = torch.add(rss_e[jj], v[i])
                        rss_e_updated = rss_e.clone()
                        rss_e_updated[jj] = rss_e_c
                        rss_e = rss_e_updated
                    elif j%2 == 1:
                        ss_e_c = torch.add(ss_e[jj], v[i])
                        ss_e_updated = ss_e.clone()
                        ss_e_updated[jj] = ss_e_c
                        ss_e = ss_e_updated
                pack_batched_sent_state_expand.append(ss_e)
                pack_batched_r_sent_state_expand.append(rss_e)
            sent_state_expand = torch.cat(pack_batched_sent_state_expand, dim=0)
            r_sent_state_expand = torch.cat(pack_batched_r_sent_state_expand, dim=0)
            sent_state_e = torch.cat([sent_state, sent_state_expand], dim=1)
            r_sent_state_e = torch.cat([r_sent_state, r_sent_state_expand], dim=1)
            sent_state = self.cons_woie2sent_size(sent_state_e) + sent_state
            r_sent_state = self.cons_woie2rsent_size(r_sent_state_e) + r_sent_state

        elif self._hps.pred_method == 'test_wdc2':
            DA_G, subject_state, predicate_state, object_state, entity_state, record_spo_length_g_dict, entity2lineid = self.set_spoenfeature(DA_G, sent_state, r_sent_state, ie_count, ir_count, conversation_embed, oie_alng_bert_oieseq, oie_alng_glv_oieseq, bert_sent_boundary, oie_in_idx, spoe2nid_dict)
            # reconstruct the spo_state from subject_state, predicate_state, object_state
            js, jp, jo = 0, 0, 0
            spo_state = []
            for j, j_data in record_spo_length_g_dict.items():
                if subject_state.shape[0] > 0:
                    subject_state_c = subject_state[js:js+j_data['S']]
                if predicate_state.shape[0] > 0:
                    predicate_state_c = predicate_state[jp:jp+j_data['P']]
                if object_state.shape[0] > 0:
                    object_state_c = object_state[jo:jo+j_data['O']]
                js += j_data['S']
                jp += j_data['P']
                jo += j_data['O']
                if subject_state.shape[0] > 0:
                    spo_state.append(subject_state_c)
                if predicate_state.shape[0] > 0:
                    spo_state.append(predicate_state_c)
                if object_state.shape[0] > 0:
                    spo_state.append(object_state_c)
            spo_state = torch.cat(spo_state, dim=0)
            for i in range(self._n_iter):
                spo_state = self.spo2spo(DA_G, spo_state, spo_state)
                entity_state = self.spo2entity(DA_G, spo_state, entity_state)

            enode_split = [j_data['E'] for j, j_data in record_spo_length_g_dict.items()]
            entity2lineid_splited = entity2lineid.split(enode_split)
            entity_state_splited = entity_state.split(enode_split)

            r_sent_state_expand = torch.zeros((r_sent_state.shape[0], self.spoe_embed_size)).to(self._hps.device)
            sent_state_expand = torch.zeros((sent_state.shape[0], self.spoe_embed_size)).to(self._hps.device)
            batched_sent_state_expand = sent_state_expand.split(ie_count)
            batched_r_sent_state_expand = r_sent_state_expand.split(ir_count)
            # update sent_state and r_sent_state
            # we build our task under intelocutor going first and candidate following scheme
            pack_batched_sent_state_expand = []
            pack_batched_r_sent_state_expand = []
            for (k, v, ss_e, rss_e) in zip(entity2lineid_splited, entity_state_splited, batched_sent_state_expand, batched_r_sent_state_expand):
                for i, j in enumerate(k): # k is the target line_id list of conversation 
                    j = j.item()
                    jj = j//2+j%2-1
                    if j%2 == 0:
                        rss_e_c = torch.add(rss_e[jj], v[i])
                        rss_e_updated = rss_e.clone()
                        rss_e_updated[jj] = rss_e_c
                        rss_e = rss_e_updated
                    elif j%2 == 1:
                        ss_e_c = torch.add(ss_e[jj], v[i])
                        ss_e_updated = ss_e.clone()
                        ss_e_updated[jj] = ss_e_c
                        ss_e = ss_e_updated
                pack_batched_sent_state_expand.append(ss_e)
                pack_batched_r_sent_state_expand.append(rss_e)
            sent_state_expand = torch.cat(pack_batched_sent_state_expand, dim=0)
            r_sent_state_expand = torch.cat(pack_batched_r_sent_state_expand, dim=0)
            sent_state_e = torch.cat([sent_state, sent_state_expand], dim=1)
            r_sent_state_e = torch.cat([r_sent_state, r_sent_state_expand], dim=1)
            sent_state = self.cons_woie2sent_size(sent_state_e) + sent_state
            r_sent_state = self.cons_woie2rsent_size(r_sent_state_e) + r_sent_state

        elif self._hps.pred_method in ['test_wdc3', 'acg', 'hsag']:
            DA_G, subject_state, predicate_state, object_state, entity_state, record_spo_length_g_dict, entity2lineid = self.set_spoenfeature(DA_G, sent_state, r_sent_state, ie_count, ir_count, conversation_embed, oie_alng_bert_oieseq, oie_alng_glv_oieseq, bert_sent_boundary, oie_in_idx, spoe2nid_dict)
            # reconstruct the spo_state from subject_state, predicate_state, object_state
            js, jp, jo = 0, 0, 0
            spo_state = []
            for j, j_data in record_spo_length_g_dict.items():
                if subject_state.shape[0] > 0:
                    subject_state_c = subject_state[js:js+j_data['S']]
                if predicate_state.shape[0] > 0:
                    predicate_state_c = predicate_state[jp:jp+j_data['P']]
                if object_state.shape[0] > 0:
                    object_state_c = object_state[jo:jo+j_data['O']]
                js += j_data['S']
                jp += j_data['P']
                jo += j_data['O']
                if subject_state.shape[0] > 0:
                    spo_state.append(subject_state_c)
                if predicate_state.shape[0] > 0:
                    spo_state.append(predicate_state_c)
                if object_state.shape[0] > 0:
                    spo_state.append(object_state_c)
            spo_state = torch.cat(spo_state, dim=0)
            for i in range(self._n_iter):
                spo_state = self.spo2spo(DA_G, spo_state, spo_state)
                entity_state = self.spo2entity(DA_G, spo_state, entity_state)

            enode_split = [j_data['E'] for j, j_data in record_spo_length_g_dict.items()]
            entity_state_splited = entity_state.split(enode_split)

            # update sent_state and r_sent_state
            # we build our task under intelocutor going first and candidate following scheme
            batched_sent_state = sent_state.split(ie_count)
            batched_r_sent_state = r_sent_state.split(ir_count)
            
            # DEBUG
            index_record = []
            combo_sent_state_list = []
            start_index = 0
            for s_state, r_state in zip(batched_sent_state, batched_r_sent_state):
                start_can_index = start_index
                end_can_index = start_can_index + len(s_state) - 1
                start_int_index = end_can_index + 1
                end_int_index = start_int_index + len(r_state) - 1
                start_index = end_int_index + 1
                index_record.append({'c': [start_can_index, end_can_index], 'i': [start_int_index, end_int_index]})
                combo_sent_state_list.append(torch.cat((s_state, r_state), dim=0))
            combo_sent_state = torch.cat(combo_sent_state_list, dim=0)

            # initialize global nodes' embeddings
            combo_global_list = []
            for s_state, r_state in zip(batched_sent_state, batched_r_sent_state):
                g_state = torch.cat((s_state, r_state), dim=0)
                combo_global_list.append(g_state)
            ac_global_state = torch.cat(combo_global_list, dim=0)

            for i in range(self._n_iter):
                # entity -> sentence
                combo_sent_state = self.entity2sent(DSA_G, entity_state, combo_sent_state)
                # sentence -> entity
                entity_state = self.sent2entity(DSA_G, combo_sent_state, entity_state)
                
                # sentence and entity -> global
                combo_sse_state_list = []
                chunk_batched_ss_embed = [combo_sent_state[d['c'][0]: d['i'][-1]+1] for d in index_record]
                for c_ss_state, c_entity_state in zip(chunk_batched_ss_embed, entity_state_splited):
                    c_entity_state = self.cons_entity_sent(c_entity_state)
                    c_state = torch.cat((c_ss_state, c_entity_state), dim=0)
                    combo_sse_state_list.append(c_state)
                tmp_combo_sse_state = torch.cat(combo_sse_state_list, dim=0)
                ac_global_state = self.sententity2global(DSA_G, tmp_combo_sse_state, ac_global_state)

            chunk_batched_ac_global_embed = [ac_global_state[d['c'][0]: d['i'][-1]+1] for d in index_record]
            chunk_batched_ac_global_embed = [self.att_pl_ac_global_node(d) for d in chunk_batched_ac_global_embed]
            ac_global_state = torch.cat(chunk_batched_ac_global_embed, dim=0)

            sent_state = [combo_sent_state[d['c'][0]: d['c'][-1]+1] for d in index_record]
            sent_state = torch.cat(sent_state, dim=0)
            r_sent_state = [combo_sent_state[d['i'][0]: d['i'][-1]+1] for d in index_record]
            r_sent_state = torch.cat(r_sent_state, dim=0)

        ## GET the initialized sent and r_sent features below ##

        # Final Return
        final_return['embed']['before_gat'] = {'w': word_feature,
                                               's': sent_state}

        # get sentence embedding afte GAT
        for i in range(self._n_iter):
            # sent -> word
            word_state_from_sent = self.sent2word(graph, word_state, sent_state)
            # word -> word
            if self._hps.pmi_window_width > -1:
                word_state_from_word = self.word2word(graph, word_state, word_state)
            # cefr -> word & word -> cefr
            if self._hps.cefr_info == 'graph_init':
                cefr_state_from_word = self.word2cefr(graph_c, word_state, cefr_feature)
                word_state_from_cefr = self.cefr2word(graph_c, word_state, cefr_state_from_word)
            # word information fusing
            word_state = word_state_from_sent
            if self._hps.pmi_window_width > -1:
                word_state = word_state + word_state_from_word
            if self._hps.cefr_info == 'graph_init':
                word_state = word_state + word_state_from_cefr
            # word -> sent, paragraph -> sent
            sent_state_from_word = self.word2sent(graph, word_state, sent_state)
            sent_state = sent_state_from_word
            # global node
            # due to different dimension in word nodes and sentence nodes, operating message passing with them seperately
            # initilaize global node state
            glist = dgl.unbatch(graph)
            unbatch_global_tensor = []
            u = 0
            t = 0
            for j in range(len(glist)):
                g = glist[j]
                wnode_id = g.filter_nodes(lambda nodes: nodes.data['dtype'] == 0)
                snode_id = g.filter_nodes(lambda nodes: nodes.data['dtype'] == 1)
                num_words = len(wnode_id)
                num_sents = len(snode_id)
                cur_word_state = word_state[u: u + num_words]
                cur_sent_state = sent_state[t: t + num_sents]
                proj_word_state = self.word2sent_proj(cur_word_state)
                if num_words != proj_word_state.shape[0]:
                    wid = graph.nodes[wnode_id].data["id"]
                unbatch_global_tensor.append(torch.cat((proj_word_state, cur_sent_state), dim=0))
                u += num_words
                t += num_sents

            # hc_global_state = torch.cat([torch.mean(v, dim=0).unsqueeze(0) for v in unbatch_global_tensor], dim=0)
            hc_global_state = torch.cat([self.att_pl_hec_global_node(v) for v in unbatch_global_tensor], dim=0)
            # word & sent -> global
            hc_global_state = self.wordsent2global(graph, torch.cat(unbatch_global_tensor, dim=0), hc_global_state)

        # generate relation nodes' embeddings
        rnode_id = M_G.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)
        rid = M_G.nodes[rnode_id].data['label'].reshape(-1)  # [n_wnodes]
        rel_embed = self.relation_nodes_embeddings(rid)  # (relation num, hidden size)

        batched_sent_state = sent_state.split(ie_count)
        batched_r_sent_state = r_sent_state.split(ir_count)
        batched_rel_embed = torch.split(rel_embed, 17, dim=0)
        index_record = []
        combo_sent_state_list = []
        start_index = 0
        for s_state, r_state in zip(batched_sent_state, batched_r_sent_state):
            start_can_index = start_index
            end_can_index = start_can_index + len(s_state) - 1
            start_int_index = end_can_index + 1
            end_int_index = start_int_index + len(r_state) - 1
            start_index = end_int_index + 1
            index_record.append({'c': [start_can_index, end_can_index], 'i': [start_int_index, end_int_index]})
            combo_sent_state_list.append(torch.cat((s_state, r_state), dim=0))
        combo_sent_state = torch.cat(combo_sent_state_list, dim=0)

        # initialize global nodes' embeddings
        # gnode_id = M_G.filter_nodes(lambda nodes: nodes.data["dtype"] == 0)
        combo_global_list = []
        for s_state, r_state, rel_state in zip(batched_sent_state, batched_r_sent_state, batched_rel_embed):
            ss_state = torch.cat((s_state, r_state), dim=0)
            g_state = torch.cat((ss_state, rel_state), dim=0)
            # combo_global_list.append(torch.mean(g_state, dim=0).unsqueeze(0))
            combo_global_list.append(self.att_pl_sd_global_node(g_state))
        sd_global_state = torch.cat(combo_global_list, dim=0)

        # responses and relation GAT learning
        for i in range(self._n_iter):
            rel_embed = self.sent2relation(M_G, combo_sent_state, rel_embed)
            combo_sent_state = self.relation2sent(M_G, combo_sent_state, rel_embed)
            
            combo_ssr_state_list = []
            chunk_batched_ss_embed = [combo_sent_state[d['c'][0]: d['i'][-1]+1] for d in index_record]
            chunck_batched_rel_embed = torch.split(rel_embed, 17, dim=0)
            for c_ss_state, c_rel_state in zip(chunk_batched_ss_embed, chunck_batched_rel_embed):
                c_state = torch.cat((c_ss_state, c_rel_state), dim=0)
                combo_ssr_state_list.append(c_state)
            tmp_combo_ssr_state = torch.cat(combo_ssr_state_list, dim=0)
            sd_global_state = self.sentrel2global(M_G, tmp_combo_ssr_state, sd_global_state)

        # evaluate from which
        # batched_rel_embed = torch.split(relation_from_sent, 17, dim=0)
        # nodeids_utt_from_candidate = M_G.filter_nodes(lambda nodes: nodes.data['kind'] == 0)
        # nodeids_utt_from_interlocutor = M_G.filter_nodes(lambda nodes: nodes.data['kind'] == 1)
        # split_chunk_can_each_stage_tensor = [combo_sent_state[d['c'][0]: d['c'][-1]+1] for d in index_record]
        # split_chunk_int_each_stage_tensor = [combo_sent_state[d['i'][0]: d['i'][-1]+1] for d in index_record]
        # split_chunk_combo_each_stage_tensor = [combo_sent_state[d['c'][0]: d['i'][-1]+1] for d in index_record]
        # p_results = []
        # for sent_paragraph_tensor, s_state in zip(split_chunk_can_each_stage_tensor, batched_sent_state):
        #     w1 = F.relu(self.sent_in_para_weight_attn1(sent_paragraph_tensor.squeeze(0)+s_state))
        #     w2 = F.relu(self.sent_in_para_weight_attn2(w1))
        #     att = torch.softmax(w2, dim=0)
        #     sent_paragraph_tensor = sent_paragraph_tensor * att
        #     output , (_ , _) = self.lstm_paragraph(sent_paragraph_tensor) #output shape (num paragraph , max sentence , 2 * sent hidden dim )
        #     paragraph_feature = output[-1 , :] #shape (num paragraph, 2* sent dim)
        #     p_results.append(paragraph_feature[None, :]) #shape (num paragraph , 1)
        # paragraph_state = self.lstm_para_proj(torch.cat(p_results, dim=0))

        # candidate_embeds = combo_sent_state[nodeids_utt_from_candidate]
        # interlocutor_embeds = combo_sent_state[nodeids_utt_from_interlocutor]

        # Fusion Mechanism
        # TODO: debug
        if self._hps.pred_method == 'test_wdc':
            bert_input = self._get_bert_input(data.get('bert_feature'), specific_key=['input_ids', 'attention_mask', 'token_type_ids'])
            bert_mp_embed = mean_pooling(conversation_embed.get('last_hidden_state'), attention_mask=bert_input['attention_mask'])
            paragraph_state = self.fuse_muliheadatt({'b': bert_mp_embed, 'hcg': hc_global_state, 'sdg': sd_global_state})
        elif self._hps.pred_method == 'test_wdc2':
            bert_input = self._get_bert_input(data.get('bert_feature'), specific_key=['input_ids', 'attention_mask', 'token_type_ids'])
            bert_mp_embed = mean_pooling(conversation_embed.get('last_hidden_state'), attention_mask=bert_input['attention_mask'])
            paragraph_state = self.fuse_muliheadatt({'b': bert_mp_embed, 'hcg': hc_global_state, 'sdg': sd_global_state})
        elif self._hps.pred_method == 'test_wdc3':
            bert_input = self._get_bert_input(data.get('bert_feature'), specific_key=['input_ids', 'attention_mask', 'token_type_ids'])
            bert_mp_embed = mean_pooling(conversation_embed.get('last_hidden_state'), attention_mask=bert_input['attention_mask'])
            paragraph_state = self.fuse_muliheadatt({'b': bert_mp_embed, 'hcg': hc_global_state, 'sdg': sd_global_state, 'acg': ac_global_state})

        if self._hps.pred_method == 'all_s':
            bert_input = self._get_bert_input(data.get('bert_feature'), specific_key=['input_ids', 'attention_mask', 'token_type_ids'])
            bert_mp_embed = mean_pooling(conversation_embed.get('last_hidden_state'), attention_mask=bert_input['attention_mask'])
            paragraph_state = self.fuse_muliheadatt({'b': bert_mp_embed, 'hcg': hc_global_state, 'sdg': sd_global_state})

        elif self._hps.pred_method == 'sde_s':
            bert_input = self._get_bert_input(data.get('bert_feature'), specific_key=['input_ids', 'attention_mask', 'token_type_ids'])
            bert_mp_embed = mean_pooling(conversation_embed.get('last_hidden_state'), attention_mask=bert_input['attention_mask'])
            s = torch.sigmoid(self.bert_gt_w(torch.cat((sd_global_state, bert_mp_embed), dim=1)))
            bert_state = s * self.down_bert(bert_mp_embed)
            paragraph_state = sd_global_state + bert_state
        elif self._hps.pred_method == 'hec_s':
            bert_input = self._get_bert_input(data.get('bert_feature'), specific_key=['input_ids', 'attention_mask', 'token_type_ids'])
            bert_mp_embed = mean_pooling(conversation_embed.get('last_hidden_state'), attention_mask=bert_input['attention_mask'])
            s = torch.sigmoid(self.bert_gt_w(torch.cat((hc_global_state, bert_mp_embed), dim=1)))
            bert_state = s * self.down_bert(bert_mp_embed)
            paragraph_state = hc_global_state + bert_state
        elif self._hps.pred_method == 'sde':
            paragraph_state = sd_global_state
        elif self._hps.pred_method == 'hec':
            paragraph_state = hc_global_state
        elif self._hps.pred_method == 'acg':
            paragraph_state = ac_global_state
            # paragraph_state = hc_global_state + sd_global_state
        elif self._hps.pred_method == 'ehg':
            paragraph_state = self.ssss(torch.cat([hc_global_state, sd_global_state], dim=1))
            # paragraph_state = hc_global_state + sd_global_state
        elif self._hps.pred_method == 'hsag':
            paragraph_state = self.fuse_muliheadatt({'hcg': hc_global_state, 'sdg': sd_global_state, 'acg': ac_global_state})
        elif self._hps.pred_method == 'ehg_ldc':
            a = hc_global_state
            b = sd_global_state
            c = hc_global_state + sd_global_state
            d = torch.abs(hc_global_state - sd_global_state)
            e = hc_global_state * sd_global_state
            paragraph_state = self.dd(torch.cat([a, b, c, d, e], dim=1))
        elif self._hps.pred_method == 'hec_ldc':
            a = hc_global_state
            b = self.down_bert(bert_state)
            c = hc_global_state + bert_state
            d = torch.abs(hc_global_state - bert_state)
            e = hc_global_state * bert_state
            paragraph_state = self.dd(torch.cat([a, b, c, d, e], dim=1))
        elif self._hps.pred_method == 'sde_ldc':
            a = sd_global_state
            b = self.down_bert(bert_state)
            c = sd_global_state + bert_state
            d = torch.abs(sd_global_state - bert_state)
            e = sd_global_state * bert_state
            paragraph_state = self.dd(torch.cat([a, b, c, d, e], dim=1))
        elif self._hps.pred_method == 'all_ldc':
            a = hc_global_state
            b = self.down_bert(bert_state)
            c = hc_global_state + bert_state
            d = torch.abs(hc_global_state - bert_state)
            e = hc_global_state * bert_state
            paragraph_state = self.dd(torch.cat([a, b, c, d, e], dim=1)) + sd_global_state
            

        result = None
        w_result = None
        
        # Paragraph CEFR
        p_result = self.pwh(paragraph_state)
        
        # Final Return
        final_return['record'] = {'M_G': index_record,
                                  'ie': ie_count,
                                  'ir': ir_count}
        final_return['embed']['after_gat'] = {'w': word_feature,
                                              's': sent_state,
                                              'p': paragraph_state}
        final_return['results'] = {'w': w_result,
                                   's': result,
                                   'p': p_result}

        return final_return

    # def compute_doc(self, passages):
    #     z_pass = self.doc_layer(passages) #shape (num topic , hidden size)
    #     #compute attention 
    #     w = F.leaky_relu(self.doc_att_linear(z_pass)) #shape (num topic , 1)
    #     att = F.softmax(w, dim=0) #shape (num topic , 1) satisfy condition: sum = 1
    #     s = att * z_pass #shape (num topic , 1) * (num topic , hidden size ) --> (num topic , hidden size) 
    #     out = torch.sum(s, dim=0) #shape (hidden size)
    #     return out       

    # def set_pnfeature(self, graph_sp):
    #     # sentence node feature from graph_sp, get paragraph node embedding from sentence boundary
    #     snode_id = graph_sp.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
    #     cnn_feature = self._sent_cnn_feature(graph_sp, snode_id)
    #     features, glen = get_snode_feat(graph_sp, feat="sent_embedding")
    #     lstm_feature = self._sent_lstm_feature(features, glen)
    #     sent_node_feature = torch.cat([cnn_feature, lstm_feature], dim=1)  # [n_nodes, n_feature_size * 2]

    #     glist = dgl.unbatch(graph_sp)
    #     unbatch_sntensor = []
    #     for j in range(len(glist)):
    #         g = glist[j]
    #         snode_id = g.filter_nodes(lambda nodes: nodes.data['dtype'] == 1)
    #         num_sents = len(snode_id)
    #         st_idx = j*num_sents
    #         cur_sent_state = sent_node_feature[st_idx: st_idx + num_sents]
    #         unbatch_sntensor.append(cur_sent_state.view(1, cur_sent_state.shape[0], cur_sent_state.shape[-1]))
    #     unbatch_sntensor = torch.cat([self.compute_paragraph_feature(k) for k in unbatch_sntensor], dim=0)

    #     # paragraph node featrue from graph_sp, get paragraph node embedding from paragraph boundary
    #     pnode_id = graph_sp.filter_nodes(lambda nodes: nodes.data["dtype"] == 0)
    #     cnn_feature = self._sent_cnn_feature(graph_sp, pnode_id)
    #     features, glen = get_pnode_feat(graph_sp, feat="sent_embedding")
    #     lstm_feature = self._sent_lstm_feature(features, glen)
    #     para_node_feature = self.pnode_proj(torch.cat([cnn_feature, lstm_feature], dim=1))
    #     node_feature = self.snode_pnode_proj(torch.cat((unbatch_sntensor, para_node_feature), dim=1))
    #     return node_feature

    # def compute_paragraph_feature(self, sent_paragraph_tensor):
    #     #sent paragraph tensor : (num paragraph , max sentence , sent dim )

    #     if self._hps.han_s:
    #         w1 = F.relu(self.sent_in_para_weight_attn1(sent_paragraph_tensor.squeeze(0)))
    #         w2 = F.relu(self.sent_in_para_weight_attn2(w1))
    #         att = torch.softmax(w2, dim=0)
    #         extended_att = att.unsqueeze(0)
    #         sent_paragraph_tensor = sent_paragraph_tensor * extended_att
    #     output , (_ , _) = self.lstm_paragraph(sent_paragraph_tensor) #output shape (num paragraph , max sentence , 2 * sent hidden dim )
    #     paragraph_feature = output[: , -1 , :] #shape (num paragraph, 2* sent dim)
    #     return self.lstm_para_proj(paragraph_feature) #shape (num paragraph , sent dim)

    def _init_sn_param(self):
        self.sent_pos_embed = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(self._hps.doc_max_timesteps + 1, self.embed_size, padding_idx=0),
            freeze=True)
        self.cnn_proj = nn.Linear(self.embed_size, self._hps.n_feature_size)
        if self._hps.language_use:
            self.cnn_pos_proj = nn.Linear(self.embed_size, self._hps.n_feature_size)
            self.cnn_dep_proj = nn.Linear(self.embed_size, self._hps.n_feature_size)
            self.cnn_ged_proj = nn.Linear(self.embed_size, self._hps.n_feature_size)
        self.lstm_hidden_state = self._hps.lstm_hidden_state
        self.lstm = nn.LSTM(self.embed_size, self.lstm_hidden_state, num_layers=self._hps.lstm_layers, dropout=0.1,
                            batch_first=True, bidirectional=self._hps.bidirectional)
        if self._hps.language_use:
            self.pos_lstm = nn.LSTM(self.embed_size, self.lstm_hidden_state, num_layers=self._hps.lstm_layers, dropout=0.1,
                                batch_first=True, bidirectional=self._hps.bidirectional)
            self.dep_lstm = nn.LSTM(self.embed_size, self.lstm_hidden_state, num_layers=self._hps.lstm_layers, dropout=0.1,
                                batch_first=True, bidirectional=self._hps.bidirectional)
            self.ged_lstm = nn.LSTM(self.embed_size, self.lstm_hidden_state, num_layers=self._hps.lstm_layers, dropout=0.1,
                                batch_first=True, bidirectional=self._hps.bidirectional)
        if self._hps.bidirectional:
            self.lstm_proj = nn.Linear(self.lstm_hidden_state * 2, self._hps.n_feature_size)
            if self._hps.language_use:
                self.lstm_pos_proj = nn.Linear(self.lstm_hidden_state * 2, self._hps.n_feature_size)
                self.lstm_dep_proj = nn.Linear(self.lstm_hidden_state * 2, self._hps.n_feature_size)
                self.lstm_ged_proj = nn.Linear(self.lstm_hidden_state * 2, self._hps.n_feature_size)
        else:
            self.lstm_proj = nn.Linear(self.lstm_hidden_state, self._hps.n_feature_size)
            if self._hps.language_use:
                self.lstm_pos_proj = nn.Linear(self.lstm_hidden_state, self._hps.n_feature_size)
                self.lstm_dep_proj = nn.Linear(self.lstm_hidden_state, self._hps.n_feature_size)
                self.lstm_ged_proj = nn.Linear(self.lstm_hidden_state, self._hps.n_feature_size)

        self.ngram_enc = sentEncoder(self._hps, self._embed)
        if self._hps.language_use:
            self.pos_enc = posEncoder(self._hps)
            self.dep_enc = depEncoder(self._hps)
            self.ged_enc = gedEncoder(self._hps)

        # lstm for passage 
        sent_hidden_size = self._hps.hidden_size * 2
        # if self._hps.bert_config is not None:
        #     sent_hidden_size = sent_hidden_size + self._hps.bert_config.hidden_size
        self.lstm_paragraph = nn.LSTM(sent_hidden_size, self._hps.hidden_size * 2, num_layers=self._hps.lstm_layers, dropout=0.1, 
                                    batch_first=True, bidirectional=self._hps.bidirectional)
        self.pnode_proj = nn.Linear(self._hps.n_feature_size * 2, self._hps.n_feature_size)
        self.snode_pnode_proj = nn.Linear(self._hps.n_feature_size * 2, self._hps.n_feature_size)
        
        if self._hps.bidirectional:
            self.lstm_para_proj = nn.Linear(
                self._hps.hidden_size*4, self._hps.hidden_size*2)
        else:
            self.lstm_para_proj = nn.Linear(
                self._hps.hidden_size*4 , self._hps.hidden_size*2)

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
    
    def set_posfeature(self, graph):
        # node feature
        snode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        
        ngram_feature = self.pos_enc.forward(graph.nodes[snode_id].data["pos"])  # [snode, embed_size]
        graph.nodes[snode_id].data["pos_embedding"] = ngram_feature
        snode_pos = graph.nodes[snode_id].data["position"].view(-1)  # [n_nodes]
        position_embedding = self.sent_pos_embed(snode_pos)
        cnn_feature = self.cnn_pos_proj(ngram_feature + position_embedding)
        
        features, glen = get_snode_feat(graph, feat="pos_embedding")
        
        pad_seq = rnn.pad_sequence(features, batch_first=True)
        lstm_input = rnn.pack_padded_sequence(pad_seq, glen, batch_first=True)
        lstm_output, _ = self.pos_lstm(lstm_input)
        unpacked, unpacked_len = rnn.pad_packed_sequence(lstm_output, batch_first=True)
        lstm_embedding = [unpacked[i][:unpacked_len[i]] for i in range(len(unpacked))]
        a = torch.cat(lstm_embedding, dim=0) # debug
        lstm_feature = self.lstm_pos_proj(a)  # [n_nodes, n_feature_size]
        
        node_feature = torch.cat([cnn_feature, lstm_feature], dim=1)  # [n_nodes, n_feature_size * 2]
        return node_feature
    
    def set_depfeature(self, graph):
        # node feature
        snode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        
        ngram_feature = self.dep_enc.forward(graph.nodes[snode_id].data["dep"])  # [snode, embed_size]
        graph.nodes[snode_id].data["dep_embedding"] = ngram_feature
        snode_pos = graph.nodes[snode_id].data["position"].view(-1)  # [n_nodes]
        position_embedding = self.sent_dep_embed(snode_pos)
        cnn_feature = self.cnn_pos_proj(ngram_feature + position_embedding)
        
        features, glen = get_snode_feat(graph, feat="dep_embedding")
        
        pad_seq = rnn.pad_sequence(features, batch_first=True)
        lstm_input = rnn.pack_padded_sequence(pad_seq, glen, batch_first=True)
        lstm_output, _ = self.dep_lstm(lstm_input)
        unpacked, unpacked_len = rnn.pad_packed_sequence(lstm_output, batch_first=True)
        lstm_embedding = [unpacked[i][:unpacked_len[i]] for i in range(len(unpacked))]
        a = torch.cat(lstm_embedding, dim=0) # debug
        lstm_feature = self.lstm_dep_proj(a)  # [n_nodes, n_feature_size]
        
        node_feature = torch.cat([cnn_feature, lstm_feature], dim=1)  # [n_nodes, n_feature_size * 2]
        return node_feature

    def set_gedfeature(self, graph):
        # node feature
        snode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        
        ngram_feature = self.ged_enc.forward(graph.nodes[snode_id].data["ged"])  # [snode, embed_size]
        graph.nodes[snode_id].data["ged_embedding"] = ngram_feature
        snode_pos = graph.nodes[snode_id].data["position"].view(-1)  # [n_nodes]
        position_embedding = self.sent_pos_embed(snode_pos)
        cnn_feature = self.cnn_ged_proj(ngram_feature + position_embedding)
        
        features, glen = get_snode_feat(graph, feat="ged_embedding")
        
        pad_seq = rnn.pad_sequence(features, batch_first=True)
        lstm_input = rnn.pack_padded_sequence(pad_seq, glen, batch_first=True)
        lstm_output, _ = self.ged_lstm(lstm_input)
        unpacked, unpacked_len = rnn.pad_packed_sequence(lstm_output, batch_first=True)
        lstm_embedding = [unpacked[i][:unpacked_len[i]] for i in range(len(unpacked))]
        a = torch.cat(lstm_embedding, dim=0) # debug
        lstm_feature = self.lstm_ged_proj(a)  # [n_nodes, n_feature_size]
        
        node_feature = torch.cat([cnn_feature, lstm_feature], dim=1)  # [n_nodes, n_feature_size * 2]
        return node_feature

    def set_spoenfeature(self, graph, sent_state, r_sent_state, ie_count, ir_count, conversation_embed, oie_alng_bert_oieseq, oie_alng_glv_oieseq, bert_sent_boundary, oie_in_idx, spoe2nid_dict):

        ## BERT
        complete_bert_embed = conversation_embed.get('last_hidden_state')

        subject_predicate_object_entity_tensor = []
        subject_predicate_object_tensor = []
        entity_tensor = []
        batch_graph = []
        record_spo_length_g_dict = {}

        sent_state = sent_state.split(ie_count)
        r_sent_state = r_sent_state.split(ir_count)

        for i, (g, bsb, oabo, oago, oii) in enumerate(zip(dgl.unbatch(graph), bert_sent_boundary, oie_alng_bert_oieseq, oie_alng_glv_oieseq, oie_in_idx)):

            # node ids
            snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 0) # subject nodes
            pnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1) # predicate nodes
            onode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 2) # object nodes
            enode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 3) # entity nodes
            sponode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] != 3) # subject, predicate, object nodes
            spoenode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] <= 3) # subject, predicate, object, entity nodes
            S = len(snode_id)
            P = len(pnode_id)
            O = len(onode_id)
            E = len(enode_id)
            record_spo_length_g_dict[i] = {'S': S, 'P': P, 'O': O, 'E': E}

            # edge ids, where the embedding of edges is set to ones
            spoedge_id = g.filter_edges(lambda edges: edges.data["dtype"] == 1) # for word to supernode(sent&doc)
            g.edges[spoedge_id].data["tfidfembed"] = torch.ones(spoedge_id.shape[-1], self._hps.feat_embed_size).to(self._hps.device)

            # first we build a full embed table, then insert the node embed from filtered node ids
            spo_embed = torch.rand(sponode_id.shape[-1], self.spoe_embed_size).to(self._hps.device)
            spoe_embed = torch.rand(spoenode_id.shape[-1], self.spoe_embed_size).to(self._hps.device)
            entity_embed = torch.rand(enode_id.shape[-1], self.spoe_embed_size).to(self._hps.device)

            # run in sentence-wise of the processed conversation content
            for line_id, oies in oii.items():

                # 
                token_boundary = bsb[int(line_id)]
                oie_bert_seq_align = oabo[line_id]['s_c2res_bert']
                oie_glv_seq_align = oago[line_id]['s_c2res_glove']

                # the segmental sent embed of BERT
                o_sent_tokens_embed = complete_bert_embed[i][list(range(token_boundary[0],token_boundary[-1]+1))]

                # # the selected sent embed of GloVe
                # j = int(line_id)
                # jj = j//2+j%2-1
                # o_sent_glv_embed = sent_state[i][jj] if j%2 == 0 else r_sent_state[i][jj]
                # print('o_sent_glv_embed: ', o_sent_glv_embed.shape)
                # input()

                # filter out the Subject, Predicate, and Object embed from the above segmental sent embed
                for io, oie in enumerate(oies):

                    # BERT part
                    # obtain the tokens in specific range as an sentence
                    bert_arg0 = list(filter(None, [oie_bert_seq_align[i] for i in oie.get('ARG0', [])]))
                    bert_v = list(filter(None, [oie_bert_seq_align[i] for i in oie.get('V', [])]))
                    bert_arg1 = list(filter(None, [oie_bert_seq_align[i] for i in oie.get('ARG1', [])]))
                    bert_arg0 = o_sent_tokens_embed[list(set(sum(bert_arg0, [])))] if bert_arg0 else None
                    bert_v = o_sent_tokens_embed[list(set(sum(bert_v, [])))] if bert_v else None
                    bert_arg1 = o_sent_tokens_embed[list(set(sum(bert_arg1, [])))] if bert_arg1 else None

                    # # GloVe part
                    # # obtain the tokens in specific range as an sentence
                    # glv_arg0 = list(filter(None, [oie_glv_seq_align[i] for i in oie.get('ARG0', [])]))
                    # glv_v = list(filter(None, [oie_glv_seq_align[i] for i in oie.get('V', [])]))
                    # glv_arg1 = list(filter(None, [oie_glv_seq_align[i] for i in oie.get('ARG1', [])]))
                    # glv_arg0 = o_sent_glv_embed[list(set(sum(glv_arg0, [])))] if glv_arg0 else None
                    # glv_v = o_sent_glv_embed[list(set(sum(glv_v, [])))] if glv_v else None
                    # glv_arg1 = o_sent_glv_embed[list(set(sum(glv_arg1, [])))] if glv_arg1 else None
                    
                    # if glv_arg0:
                    #     print('glv_arg0: ', glv_arg0.shape)
                    #     input()
                    # if glv_v:
                    #     print('glv_v: ', glv_v.shape)
                    #     input()
                    # if glv_arg1:
                    #     print('glv_arg1: ', glv_arg1.shape)
                    #     input()

                    # TODO:attention pooling or BLSTM w/ CNN
                    if bert_arg0 is not None:
                        if bert_arg0.shape[0] > 1:
                            # bert_arg0 = torch.mean(bert_arg0, dim=0).unsqueeze(0)
                            bert_arg0 = self.att_pl_oie_s_bert(bert_arg0)
                    if bert_v is not None:
                        if bert_v.shape[0] > 1:
                            # bert_v = torch.mean(bert_v, dim=0).unsqueeze(0)
                            bert_v = self.att_pl_oie_p_bert(bert_v)
                    if bert_arg1 is not None:
                        if bert_arg1.shape[0] > 1:
                            # bert_arg1 = torch.mean(bert_arg1, dim=0).unsqueeze(0)
                            bert_arg1 = self.att_pl_oie_o_bert(bert_arg1)

                    # # TODO:attention pooling or BLSTM w/ CNN
                    # if glv_arg0 is not None:
                    #     if glv_arg0.shape[0] > 1:
                    #         glv_arg0 = torch.mean(glv_arg0, dim=0).unsqueeze(0)
                    # if glv_v is not None:
                    #     if glv_v.shape[0] > 1:
                    #         glv_v = torch.mean(glv_v, dim=0).unsqueeze(0)
                    # if glv_arg1 is not None:
                    #     if glv_arg1.shape[0] > 1:
                    #         print('glv_arg1: ', glv_arg1.shape)
                    #         input()
                    #         glv_arg1 = torch.mean(glv_arg1, dim=0).unsqueeze(0)

                    # if 
                    if bert_arg0 is not None:
                        bert_arg0 = self.cons_bert2oie_s(bert_arg0)
                    if bert_v is not None:
                        bert_v = self.cons_bert2oie_p(bert_v)
                    if bert_arg1 is not None:
                        bert_arg1 = self.cons_bert2oie_o(bert_arg1)

                    # # if
                    # if glv_arg0 is not None:
                    #     print('glv_arg0: ', glv_arg0.shape)
                    #     input()
                    #     glv_arg0 = self.cons_glv2oie_s(glv_arg0)
                    # if glv_v is not None:
                    #     print('glv_v: ', glv_v.shape)
                    #     input()
                    #     glv_v = self.cons_glv2oie_p(glv_v)
                    # if glv_arg1 is not None:
                    #     print('glv_arg1: ', glv_arg1.shape)
                    #     input()
                    #     glv_arg1 = self.cons_glv2oie_o(glv_arg1)

                    # if tensor is None:
                    if bert_arg0 is None:
                        bert_arg0 = torch.zeros((1, self.spoe_embed_size)).to(self._hps.device)
                    if bert_v is None:
                        bert_v = torch.zeros((1, self.spoe_embed_size)).to(self._hps.device)
                    if bert_arg1 is None:
                        bert_arg1 = torch.zeros((1, self.spoe_embed_size)).to(self._hps.device)

                    # # if tensor is None:
                    # if glv_arg0 is None:
                    #     glv_arg0 = torch.zeros((1, self.spoe_embed_size)).to(self._hps.device)
                    # if glv_v is None:
                    #     glv_v = torch.zeros((1, self.spoe_embed_size)).to(self._hps.device)
                    # if glv_arg1 is None:
                    #     glv_arg1 = torch.zeros((1, self.spoe_embed_size)).to(self._hps.device)

                    # use the prepared index dictionary map back to the index in graph
                    # update subject, predicate, and object nodes at the same time
                    oie_item_id = '{}_{}'.format(line_id, io)
                    po_arg0 = spoe2nid_dict[i][oie_item_id]['ARG0']
                    po_v = spoe2nid_dict[i][oie_item_id]['V']
                    po_arg1 = spoe2nid_dict[i][oie_item_id]['ARG1']
                    po_entity = spoe2nid_dict[i][oie_item_id]['ENTITY']

                    if po_arg0:
                        spo_embed[po_arg0] = bert_arg0
                        spoe_embed[po_arg0] = bert_arg0
                    if po_v:
                        spo_embed[po_v] = bert_v
                        spoe_embed[po_v] = bert_v
                    if po_arg1:
                        spo_embed[po_arg1] = bert_arg1
                        spoe_embed[po_arg1] = bert_arg1
                    entity_embed[po_entity-(S+P+O)] = bert_arg0 + bert_v + bert_arg1
                    spoe_embed[po_entity] = bert_arg0 + bert_v + bert_arg1

                # target: fill the embed back to the corresponding node index in graph

                ## one sentence has one averaged entity embed finally.

            subject_predicate_object_entity_tensor.append(spo_embed)
            subject_predicate_object_entity_tensor.append(entity_embed)
            subject_predicate_object_tensor.append(spo_embed)
            # subject_tensor.append()
            # predicate_tensor.append()
            # object_tensor.append()
            entity_tensor.append(entity_embed)

            g.nodes[sponode_id].data["embed"] = spo_embed
            g.nodes[enode_id].data["embed"] = entity_embed
            batch_graph.append(g)

            # pooled_sent_embed = torch.mean(complete_bert_embed[list(range(token_boundary[0],token_boundary[-1]+1))], dim=0).reshape(1, -1)

        # wid = graph.nodes[sponode_id].data["id"]  # [n_wnodes]
        # w_embed = self._embed(wid)  # [n_wnodes, D]
        # graph.nodes[wnode_id].data["embed"] = w_embed
        # etf = graph.edges[wsedge_id].data["tffrac"]
        # graph.edges[wsedge_id].data["tfidfembed"] = self._TFembed(etf)
        # if self._hps.pmi_window_width > -1:
        #     wwedge_id = graph.filter_edges(lambda edges: edges.data["dtype"] == 1)   # for word to word
        #     eww = graph.edges[wwedge_id].data["tffrac"]
        #     graph.edges[wwedge_id].data["tfidfembed"] = self._TFembed(eww)
        # return w_embed

        graph = dgl.batch(batch_graph)

        snode_id = graph.filter_nodes(lambda nodes: nodes.data["unit"] == 0) # subject nodes
        pnode_id = graph.filter_nodes(lambda nodes: nodes.data["unit"] == 1) # predicate nodes
        onode_id = graph.filter_nodes(lambda nodes: nodes.data["unit"] == 2) # object nodes
        enode_id = graph.filter_nodes(lambda nodes: nodes.data["unit"] == 3) # entity nodes

        subject_state = graph.nodes[snode_id].data["embed"]
        predicate_state = graph.nodes[pnode_id].data["embed"]
        object_state = graph.nodes[onode_id].data["embed"]
        entity_state = graph.nodes[enode_id].data["embed"]
        entity2lineid = graph.nodes[enode_id].data["position"]

        return graph, subject_state, predicate_state, object_state, entity_state, record_spo_length_g_dict, entity2lineid

    def _get_bert_embed(self, token_inputs, specific_key=None):
        r = {}
        for pack_input in token_inputs:
            if specific_key:
                pack_input = {k: v.to(self.bert_device) for k, v in pack_input.items() if k in specific_key}
            else:
                pack_input = {k: v.to(self.bert_device) for k, v in pack_input.items()}
            self.bert = self.bert.to(self.bert_device)

            # freeze
            if not self._hps.bert_train:
                for name, param in self.bert.named_parameters():
                    # 'embeddings.LayerNorm.weight', 'embeddings.LayerNorm.bias'
                    if name in [
                        'pooler.dense.weight', 'pooler.dense.bias',
                    ]:
                        continue
                    param.requires_grad = False

            bert_output = self.bert(input_ids=pack_input.get('input_ids'),
                                    attention_mask=pack_input.get('attention_mask'),
                                    token_type_ids=pack_input.get('token_type_ids'))
            bert_output = {k: v.to(self._hps.device) for k, v in bert_output.items() if k != 'hidden_states'}
            for k, v in bert_output.items():
                r.setdefault(k, []).append(v)
        r = {k: torch.cat(v, dim=0).to(self._hps.device) for k, v in r.items()}
        return r

    def _get_bert_input(self, token_inputs, specific_key=None):
        r = {}
        for pack_input in token_inputs:
            if specific_key:
                pack_input = {k: v.to(self._hps.device) for k, v in pack_input.items() if k in specific_key}
            else:
                pack_input = {k: v.to(self._hps.device) for k, v in pack_input.items()}
            for k, v in pack_input.items():
                r.setdefault(k, []).append(v)
        r = {k: torch.cat(v, dim=0) for k, v in r.items()}
        return r

    def _get_pooled_embed(self, token_inputs, bert_sent_boundary, conversation_embed):
        r = {'can': [], 'int': []}
        for (pack_input, sent_token_boundary, bert_embed) in zip(token_inputs, bert_sent_boundary, conversation_embed):
            can_int_spk_list = pack_input.get('spks_list').reshape(-1).tolist()
            for (spk, token_boundary) in zip(can_int_spk_list, sent_token_boundary):
                pooled_sent_embed = self.seg_sent_pooling(bert_embed[list(range(token_boundary[0],token_boundary[-1]+1))])
                # pooled_sent_embed = torch.mean(bert_embed[list(range(token_boundary[0],token_boundary[-1]+1))], dim=0).reshape(1, -1)
                r['int' if spk == 0 else 'can'].append(pooled_sent_embed)
        r = {i: torch.cat(j, dim=0) for i, j in r.items()}
        return r


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

def automodel(hps, config, long_strenthened_from_roberta):
    if config.model_type == 'bert':
        return BertModel(config), '/share/nas167/a2y3a1N0n2Yann/BertModel/basic_pretrained_model/bert-base-uncased/pytorch_model.bin'
    elif config.model_type == 'roberta':
        if hps.bert_train_finetune:
            return RobertaModel(config), '/share/nas167/a2y3a1N0n2Yann/BertModel/basic_pretrained_model/nictjle-roberta-base/pytorch_model.bin'
        return RobertaModel(config), '/share/nas167/a2y3a1N0n2Yann/BertModel/basic_pretrained_model/roberta-base/pytorch_model.bin'
    elif config.model_type == 'longformer':
        if long_strenthened_from_roberta:
            if hps.bert_train_finetune:
                return RobertaModel(config), '/share/nas167/a2y3a1N0n2Yann/BertModel/basic_pretrained_model/nictjle-roberta-base/pytorch_model.bin'
            return RobertaModel(config), '/share/nas167/a2y3a1N0n2Yann/BertModel/basic_pretrained_model/roberta-base/pytorch_model.bin'
        else:
            return LongformerModel(config), '/share/nas167/a2y3a1N0n2Yann/BertModel/basic_pretrained_model/longformer-base-4096/pytorch_model.bin'

# See: https://github.com/allenai/longformer/blob/master/scripts/convert_model_to_long.ipynb
# See: https://github.com/LennartKeller/roberta2longformer/blob/main/roberta2longformer.py
# We only want to change the length of input for roberta extending to our needs, not really retrain a longformer model
# def change_roberta_to_long_input_model(model, attention_window, max_pos):
#     # model = RobertaForMaskedLM.from_pretrained('roberta-base')
#     config = model.config

#     # extend position embeddings
#     current_max_pos, embed_size = model.embeddings.position_embeddings.weight.shape
#     max_pos += 2  # NOTE: RoBERTa has positions 0,1 reserved, so embedding size is max position + 2
#     config.max_position_embeddings = max_pos
#     assert max_pos > current_max_pos
#     # allocate a larger position embedding matrix
#     new_pos_embed = model.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)
#     # copy position embeddings over and over to initialize the new position embeddings
#     k = 2
#     step = current_max_pos - 2
#     while k < max_pos - 1:
#         print('a: ', new_pos_embed[k:(k + step)].shape)
#         print('b: ', model.embeddings.position_embeddings.weight[2:].shape)
#         input()
#         new_pos_embed[k:(k + step)] = model.embeddings.position_embeddings.weight[2:]
#         k += step
#     model.embeddings.position_embeddings.weight.data = new_pos_embed
#     model.embeddings.position_ids.data = torch.tensor([i for i in range(max_pos)]).reshape(1, max_pos)

#     # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
#     config.attention_window = [attention_window] * config.num_hidden_layers
#     for i, layer in enumerate(model.encoder.layer):
#         longformer_self_attn = LongformerSelfAttention(config, layer_id=i)
#         longformer_self_attn.query = layer.attention.self.query
#         longformer_self_attn.key = layer.attention.self.key
#         longformer_self_attn.value = layer.attention.self.value

#         longformer_self_attn.query_global = copy.deepcopy(layer.attention.self.query)
#         longformer_self_attn.key_global = copy.deepcopy(layer.attention.self.key)
#         longformer_self_attn.value_global = copy.deepcopy(layer.attention.self.value)

#         layer.attention.self = longformer_self_attn

#     return model

def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string

def change_roberta_to_long_input_model(
    roberta_model,
    roberta_tokenizer,
    bert_config,
    attention_window: int = 512,
    longformer_max_length: int = 4096,
):

    ##################################
    # Create new longformer instance #
    ##################################
    bert_config.max_position_embeddings = longformer_max_length + 2
    bert_config.attention_window = attention_window
    # longformer_config = LongformerConfig(
    #     max_position_embeddings=longformer_max_length + 2,
    #     attention_window=attention_window,
    # )
    longformer_config = bert_config
    longformer_model = LongformerModel(longformer_config)

    ###############################
    # Create longformer tokenizer #
    ###############################

    # Longformer tokenizers are Roberta tokenizers.
    # But to follow the conventions
    # and to avoid confusion we create a
    # longformer tokenizer class with the state of
    # the original tokenizer.
    # with TemporaryDirectory() as temp_dir:
    #     roberta_tokenizer.save_pretrained(temp_dir)
    #     longformer_tokenizer = LongformerTokenizerFast.from_pretrained(temp_dir)
    longformer_tokenizer = roberta_tokenizer
    longformer_tokenizer.model_max_length = longformer_max_length
    longformer_tokenizer.init_kwargs["model_max_length"] = longformer_max_length

    ######################
    # Copy model weights #
    ######################

    # We only copy the encoder weights and resize the embeddings.
    # Pooler weights are kept untouched.

    # ---------#
    # Encoder  #
    # ---------#
    roberta_parameters = roberta_model.encoder.state_dict()
    longformer_parameters = longformer_model.encoder.state_dict()

    # Load all compatible keys directly and obtain missing keys to handle later
    errors = longformer_model.encoder.load_state_dict(roberta_parameters, strict=False)
    assert not errors.unexpected_keys, "Found unexpected keys"
    missing_keys = errors.missing_keys

    # We expect, the keys to be the weights of the global attention modules and
    # reuse roberta's normal attention weights for those modules.
    for longformer_key in missing_keys:
        # Resolve layer properties
        (
            prefix,
            layer_idx,
            layer_class,
            layer_type,
            target,
            params,
        ) = longformer_key.split(".")
        assert layer_class == "attention" or target.endswith(
            "global"
        ), f"Unexcpected parameters {longformer_key}."
        # Copy the normal weights attention weights to the global attention layers too
        roberta_target_key = ".".join(
            [
                prefix,
                layer_idx,
                layer_class,
                layer_type,
                remove_suffix(target, "_global"),
                params,
            ]
        )
        # target.removesuffix("_global"),
        roberta_weights = roberta_parameters[roberta_target_key]
        longformer_parameters[longformer_key] = roberta_weights

    # Update the state of the longformer model
    longformer_model.encoder.load_state_dict(longformer_parameters, strict=True)

    # ------------#
    # Embeddings  #
    # ------------#
    # There are two types of embeddings:

    # 1. Token embeddings
    # We can simply copy the token embeddings.

    # We have to resize the token embeddings upfront, to make load_state_dict work.
    longformer_model.resize_token_embeddings(len(roberta_tokenizer))

    roberta_embeddings_parameters = roberta_model.embeddings.state_dict()
    embedding_parameters2copy = []

    for key, item in roberta_embeddings_parameters.items():
        if not "position" in key and not "token_type_embeddings" in key:
            embedding_parameters2copy.append((key, item))

    # 2. Positional embeddings
    # The positional embeddings are repeatedly copied over
    # to longformer to match the new max_seq_length

    roberta_pos_embs = roberta_model.embeddings.state_dict()[
        "position_embeddings.weight"
    ][:-2]
    roberta_pos_embs_extra = roberta_model.embeddings.state_dict()[
        "position_embeddings.weight"
    ][-2:]

    assert (
        roberta_pos_embs.size(0) < longformer_max_length
    ), "Longformer sequence length has to be longer than roberta original sequence length"

    # Figure out how many time we need to copy the original embeddings
    n_copies = round(longformer_max_length / roberta_pos_embs.size(0))

    # Copy the embeddings and handle the last missing ones.
    longformer_pos_embs = roberta_pos_embs.repeat((n_copies, 1))

    n_pos_embs_left = longformer_max_length - longformer_pos_embs.size(0) # 1300 - 1536 = -236

    if n_pos_embs_left < 0:
        longformer_pos_embs = longformer_pos_embs[:n_pos_embs_left] # 1536 - 236 = 1300
    else:
        longformer_pos_embs = torch.cat(
            [longformer_pos_embs, roberta_pos_embs[:n_pos_embs_left]], dim=0
        )

    # Add the last extra embeddings.
    longformer_pos_embs = torch.cat(
        [longformer_pos_embs, roberta_pos_embs_extra], dim=0
    )

    embedding_parameters2copy.append(
        ("position_embeddings.weight", longformer_pos_embs)
    )

    # Load the embedding weights into the longformer model
    embedding_parameters2copy = OrderedDict(embedding_parameters2copy)
    longformer_model.embeddings.load_state_dict(embedding_parameters2copy, strict=False)

    # return longformer_model, longformer_tokenizer
    return longformer_model


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def ruled_min_max(embed, min=0., max=6.):
    a = torch.where(embed > min, embed, torch.full_like(embed, min))
    b = torch.where(a < max, a, torch.full_like(a, max))
    return b

