#!/usr/bin/python
# -*- coding: utf-8 -*-

# __author__="Danqing Wang, Jiun-Ting Li"

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

from module.GATStackLayer import MultiHeadSGATLayer, MultiHeadLayer
from module.GATLayer import (
    PositionwiseFeedForward,
    WSGATLayer,
    SWGATLayer,
    WGATLayer,
    CWGATLayer,
    WCGATLayer,
    WPGATLayer,
    PWGATLayer,
    PSGATLayer,
    SPGATLayer,
    WSGGATLayer,
    ESOGATLayer,
    EOSGATLayer,
    EOPGATLayer,
    EPOGATLayer,
    EPSGATLayer,
    ESPGATLayer,
    ESPOSPOGATLayer,
    ESPOEGATLayer,
    SENTENTITYGATLayer,
    ENTITYSENTGATLayer,
    SENTENTITYGLOBALGATLayer
)

def get_dgl_graph_device(graph):
    """
    Get the device used in a DGL graph.

    Parameters:
    graph (DGLGraph): The DGL graph for which to get the device.

    Returns:
    str or None: The device name if found, None if the graph has no node or edge data.
    """
    # Check if the graph has any node or edge features
    if graph.ndata or graph.edata:
        # Get the first tensor in the node or edge data (assumes all tensors are on the same device)
        first_tensor = next(iter(graph.ndata.values() if graph.ndata else graph.edata.values()))
        return first_tensor.device
    else:
        return None

######################################### SubModule #########################################
class WSWGAT(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, attn_drop_out, ffn_inner_hidden_size, ffn_drop_out, feat_embed_size, des_embed_size, layerType):
        super().__init__()
        self.layerType = layerType
        if layerType == "W2S":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, des_embed_size, layer=WSGATLayer)
        elif layerType == "S2W":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, des_embed_size, layer=SWGATLayer)
        elif layerType == "S2S":
            self.layer = MultiHeadSGATLayer(in_dim, int(out_dim / num_heads), num_heads, des_embed_size, attn_drop_out)
        elif layerType == "W2W":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, des_embed_size, layer=WGATLayer)
        elif layerType == "C2W":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, des_embed_size, layer=CWGATLayer)
        elif layerType == "W2C":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, des_embed_size, layer=WCGATLayer)
        else:
            raise NotImplementedError("GAT Layer has not been implemented!")

        self.ffn = PositionwiseFeedForward(out_dim, ffn_inner_hidden_size, ffn_drop_out)

    def forward(self, g, w, s, od=None, dd=None, save_gpu_mode=False):
        if od and dd and save_gpu_mode:
            assert od != dd
            assert get_dgl_graph_device(g) == od
            assert w.device == od
            assert s.device == od
            # for module in self.modules():
            #     for param in module.parameters():
            #         assert param.device == od
            g.to(dd)
            w = w.to(dd)
            s = s.to(dd)
            for module in self.modules():
                module.to(dd)
  
        if self.layerType == "W2S":
            origin, neighbor = s, w
        elif self.layerType == "S2W":
            origin, neighbor = w, s
        elif self.layerType == "S2S":
            assert torch.equal(w, s)
            origin, neighbor = w, s
        elif self.layerType == "W2W":
            assert torch.equal(w, s)
            origin, neighbor = w, s
        elif self.layerType == "C2W":
            origin, neighbor = w, s
        elif self.layerType == "W2C":
            origin, neighbor = s, w
        else:
            origin, neighbor = None, None

        h = F.relu(self.layer(g, neighbor, origin))
        h = h + origin
        h = self.ffn(h.unsqueeze(0)).squeeze(0)
        
        if od and dd and save_gpu_mode:
            g.to(od)
            w = w.to(od)
            s = s.to(od)
            # for module in self.modules():
            #     module.to(od)
            h = h.to(od)
            
        return h


class WPWGAT(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, attn_drop_out, ffn_inner_hidden_size, ffn_drop_out, feat_embed_size, des_embed_size, layerType):
        super().__init__()
        self.layerType = layerType
        if layerType == "W2P":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, des_embed_size, layer=WPGATLayer)

        self.ffn = PositionwiseFeedForward(out_dim, ffn_inner_hidden_size, ffn_drop_out)

    def forward(self, g, w, p, od=None, dd=None, save_gpu_mode=False):
        if od and dd and save_gpu_mode:
            assert od != dd
            assert get_dgl_graph_device(g) == od
            assert w.device == od
            assert p.device == od
            # for module in self.modules():
            #     for param in module.parameters():
            #         assert param.device == od
            g.to(dd)
            w = w.to(dd)
            p = p.to(dd)
            for module in self.modules():
                module.to(dd)
        
        if self.layerType == "W2P":
            origin, neighbor = p, w  #from word to passage 

        h = F.relu(self.layer(g, neighbor, origin))
        h = h + origin
        h = self.ffn(h.unsqueeze(0)).squeeze(0)
        
        if od and dd and save_gpu_mode:
            g.to(od)
            w = w.to(od)
            p = p.to(od)
            # for module in self.modules():
            #     module.to(od)
            h = h.to(od)
        return h


class PWPGAT(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, attn_drop_out, ffn_inner_hidden_size, ffn_drop_out, feat_embed_size, des_embed_size, layerType):
        super().__init__()
        self.layerType = layerType
        if layerType == "P2W":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, des_embed_size, layer=PWGATLayer)

        self.ffn = PositionwiseFeedForward(out_dim, ffn_inner_hidden_size, ffn_drop_out)

    def forward(self, g, w, p, od=None, dd=None, save_gpu_mode=False):
        if od and dd and save_gpu_mode:
            assert od != dd
            assert get_dgl_graph_device(g) == od
            assert w.device == od
            assert p.device == od
            # for module in self.modules():
            #     for param in module.parameters():
            #         assert param.device == od
            g.to(dd)
            w = w.to(dd)
            p = p.to(dd)
            for module in self.modules():
                module.to(dd)
        
        if self.layerType == "P2W":
            origin, neighbor = w, p  #from word to passage 

        h = F.relu(self.layer(g, neighbor, origin))
        h = h + origin
        h = self.ffn(h.unsqueeze(0)).squeeze(0)
        
        if od and dd and save_gpu_mode:
            g.to(od)
            w = w.to(od)
            p = p.to(od)
            # for module in self.modules():
            #     module.to(od)
            h = h.to(od)
        return h


class SPSGAT(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, attn_drop_out, ffn_inner_hidden_size, ffn_drop_out, feat_embed_size, des_embed_size, layerType):
        super().__init__()
        self.layerType = layerType
        if layerType == "P2S":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, des_embed_size, layer=PSGATLayer)
        elif layerType == "S2P":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, des_embed_size, layer=SPGATLayer)

        self.ffn = PositionwiseFeedForward(out_dim, ffn_inner_hidden_size, ffn_drop_out)

    def forward(self, g, p, s, od=None, dd=None, save_gpu_mode=False):
        if od and dd and save_gpu_mode:
            assert od != dd
            assert get_dgl_graph_device(g) == od
            assert s.device == od
            assert p.device == od
            # for module in self.modules():
            #     for param in module.parameters():
            #         assert param.device == od
            g.to(dd)
            s = s.to(dd)
            p = p.to(dd)
            for module in self.modules():
                module.to(dd)
        
        if self.layerType == "P2S":
            origin, neighbor = s, p  #from passage to sentence
        elif self.layerType == "S2P":
            origin, neighbor = p, s  #from passage to sentence

        h = F.relu(self.layer(g, neighbor, origin))
        h = h + origin
        h = self.ffn(h.unsqueeze(0)).squeeze(0)
        
        if od and dd and save_gpu_mode:
            g.to(od)
            s = s.to(od)
            p = p.to(od)
            # for module in self.modules():
            #     module.to(od)
            h = h.to(od)
        
        return h
    


class WSGRGAT(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, attn_drop_out, ffn_inner_hidden_size, ffn_drop_out, feat_embed_size, des_embed_size, layerType):
        super().__init__()
        self.layerType = layerType
        if layerType == "WS2G":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, des_embed_size, layer=WSGGATLayer)

        self.ffn = PositionwiseFeedForward(out_dim, ffn_inner_hidden_size, ffn_drop_out)

    def forward(self, g, a, i):

        if self.layerType == "WS2G":
            origin, neighbor = i, a  #from sentence to relation

        h = F.relu(self.layer(g, neighbor, origin))
        h = h + origin
        h = self.ffn(h.unsqueeze(0)).squeeze(0)

        return h


class SPOEGAT(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, attn_drop_out, ffn_inner_hidden_size, ffn_drop_out, feat_embed_size, des_embed_size, layerType):
        super().__init__()
        self.layerType = layerType
        if layerType == "ES2P":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, des_embed_size, layer=ESPGATLayer)
        elif layerType == "EO2P":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, des_embed_size, layer=EOPGATLayer)
        elif layerType == "EP2S":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, des_embed_size, layer=EPSGATLayer)
        elif layerType == "EO2S":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, des_embed_size, layer=EOSGATLayer)
        elif layerType == "EP2O":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, des_embed_size, layer=EPOGATLayer)
        elif layerType == "ES2O":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, des_embed_size, layer=ESOGATLayer)
        elif layerType == "ESPO2E":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, des_embed_size, layer=ESPOEGATLayer)
        elif layerType == "ESPO2SPO":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, des_embed_size, layer=ESPOSPOGATLayer)
        elif layerType == "SENT2ENTITY":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, des_embed_size, layer=SENTENTITYGATLayer)
        elif layerType == "ENTITY2SENT":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, des_embed_size, layer=ENTITYSENTGATLayer)
        elif layerType == "SENTENTITY2GLOBAL":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, des_embed_size, layer=SENTENTITYGLOBALGATLayer)

        self.ffn = PositionwiseFeedForward(out_dim, ffn_inner_hidden_size, ffn_drop_out)

    def forward(self, g, a, i):

        if self.layerType == "ES2P":
            origin, neighbor = i, a  #from subject to predicate
        elif self.layerType == "EO2P":
            origin, neighbor = i, a  #from object to predicate
        elif self.layerType == "EP2S":
            origin, neighbor = i, a  #from predicate to subject
        elif self.layerType == "EO2S":
            origin, neighbor = i, a  #from object to subject
        elif self.layerType == "EP2O":
            origin, neighbor = i, a  #from predicate to object
        elif self.layerType == "ES2O":
            origin, neighbor = i, a  #from subject to object
        elif self.layerType == "ESPO2E":
            origin, neighbor = i, a  #from subject to object
        elif self.layerType == "ESPO2SPO":
            origin, neighbor = i, a  #from subject to object
        elif self.layerType == "SENT2ENTITY":
            origin, neighbor = i, a  #from subject to object
        elif self.layerType == "ENTITY2SENT":
            origin, neighbor = i, a  #from subject to object
        elif self.layerType == "SENTENTITY2GLOBAL":
            origin, neighbor = i, a  #from subject to object

        h = F.relu(self.layer(g, neighbor, origin))
        h = h + origin
        h = self.ffn(h.unsqueeze(0)).squeeze(0)

        return h
