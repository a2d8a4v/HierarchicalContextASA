#!/usr/bin/python
# -*- coding: utf-8 -*-

# __author__="Jiun-Ting Li"

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
import torch.nn as nn
import torch.nn.functional as F

from module.GATStackLayer import MultiHeadLayer
from module.GATLayer import PositionwiseFeedForward
from module.RGATLayer import HeteroRGATLayer, SRRGATLayer, RSRGATLayer, AGRGATLayer

class RGAT(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, attn_drop_out, ffn_inner_hidden_size, ffn_drop_out, feat_embed_size):
        super().__init__()
        self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, layer=HeteroRGATLayer)

        self.ffn = PositionwiseFeedForward(out_dim, ffn_inner_hidden_size, ffn_drop_out)

    def forward(self, g, h, eh):

        embed, rel_embed = h, eh  #from word to passage 

        h = F.relu(self.layer(g, embed, rel_embed))
        h = h + embed
        h = self.ffn(h.unsqueeze(0)).squeeze(0)

        return h


class SWSRGAT(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, attn_drop_out, ffn_inner_hidden_size, ffn_drop_out, feat_embed_size, des_embed_size, layerType):
        super().__init__()
        self.layerType = layerType
        if layerType == "S2R":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, des_embed_size, layer=SRRGATLayer)
        if layerType == "R2S":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, des_embed_size, layer=RSRGATLayer)

        self.ffn = PositionwiseFeedForward(out_dim, ffn_inner_hidden_size, ffn_drop_out)

    def forward(self, g, s, r):

        if self.layerType == "S2R":
            origin, neighbor = r, s  #from sentence to relation
        elif self.layerType == "R2S":
            origin, neighbor = s, r  #from sentence to relation 

        h = F.relu(self.layer(g, neighbor, origin))
        h = h + origin
        h = self.ffn(h.unsqueeze(0)).squeeze(0)

        return h

class AWGRGAT(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, attn_drop_out, ffn_inner_hidden_size, ffn_drop_out, feat_embed_size, des_embed_size, layerType):
        super().__init__()
        self.layerType = layerType
        if layerType == "A2G":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, des_embed_size, layer=AGRGATLayer)

        self.ffn = PositionwiseFeedForward(out_dim, ffn_inner_hidden_size, ffn_drop_out)

    def forward(self, g, a, i):

        if self.layerType == "A2G":
            origin, neighbor = i, a  #from sentence to relation

        h = F.relu(self.layer(g, neighbor, origin))
        h = h + origin
        h = self.ffn(h.unsqueeze(0)).squeeze(0)

        return h
