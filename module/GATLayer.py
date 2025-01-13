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
# Unless required by applicable law or agreed to in writing,GC software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

######################################### SubLayer #########################################
class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        assert not torch.any(torch.isnan(x)), "FFN input"
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        assert not torch.any(torch.isnan(output)), "FFN output"
        return output


######################################### HierLayer #########################################

class SGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, des_embed_size, weight=0):
        super(SGATLayer, self).__init__()
        self.weight = weight
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)  # [edge_num, 2 * out_dim]
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h, o):
        snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        sedge_id = g.filter_edges(lambda edges: edges.data["dtype"] == 0)
        g.filter_edges(lambda edges: (edges.src["unit"] == 0) & (edges.dst["unit"] == 1))
        z = self.fc(h)
        g.nodes[snode_id].data['z'] = z
        g.apply_edges(self.edge_attention, edges=sedge_id)
        g.pull(snode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        g.edata.pop('e')
        h = g.ndata.pop('sh')
        return h[snode_id]


class WSGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size, des_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.fs = nn.Linear(des_embed_size, out_dim, bias=False)

        self.feat_fc = nn.Linear(feat_embed_size, out_dim, bias=False)
        self.attn_fc = nn.Linear(3 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        dfeat = self.feat_fc(edges.data["tfidfembed"])                  # [edge_num, out_dim]
        z2 = torch.cat([edges.src['z'], edges.dst['z'], dfeat], dim=1)  # [edge_num, 3 * out_dim]
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h, o):
        wnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        wsedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 0) & (edges.dst["unit"] == 1))
        z = self.fc(h)
        i = self.fs(o)
        g.nodes[wnode_id].data['z'] = z
        g.nodes[snode_id].data['z'] = i
        g.apply_edges(self.edge_attention, edges=wsedge_id)
        g.pull(snode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        g.edata.pop('e')
        h = g.ndata.pop('sh')
        return h[snode_id]



class SWGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size, des_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.fs = nn.Linear(des_embed_size, out_dim, bias=False)

        self.feat_fc = nn.Linear(feat_embed_size, out_dim)
        self.attn_fc = nn.Linear(3 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        dfeat = self.feat_fc(edges.data["tfidfembed"])  # [edge_num, out_dim]
        z2 = torch.cat([edges.src['z'], edges.dst['z'], dfeat], dim=1)  # [edge_num, 3 * out_dim]
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h, o):
        wnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        swedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 1) & (edges.dst["unit"] == 0))
        z = self.fc(h)
        i = self.fs(o)
        g.nodes[snode_id].data['z'] = z
        g.nodes[wnode_id].data['z'] = i
        g.apply_edges(self.edge_attention, edges=swedge_id)
        g.pull(wnode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        g.edata.pop('e')
        h = g.ndata.pop('sh')
        return h[wnode_id]


class WGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size, des_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)

        self.feat_fc = nn.Linear(feat_embed_size, out_dim)
        self.attn_fc = nn.Linear(3 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        dfeat = self.feat_fc(edges.data["tfidfembed"])  # [edge_num, out_dim]
        z2 = torch.cat([edges.src['z'], edges.dst['z'], dfeat], dim=1)  # [edge_num, 3 * out_dim]
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h, o):
        wnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        wwedge_id = g.filter_edges(lambda edges: edges.data["dtype"] == 1)
        z = self.fc(h)
        g.nodes[wnode_id].data['z'] = z
        g.apply_edges(self.edge_attention, edges=wwedge_id)
        g.pull(wnode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        g.edata.pop('e')
        h = g.ndata.pop('sh')
        return h[wnode_id]


class CWGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size, des_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.fs = nn.Linear(des_embed_size, out_dim, bias=False)

        self.feat_fc = nn.Linear(feat_embed_size, out_dim)
        self.attn_fc = nn.Linear(3 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        dfeat = self.feat_fc(edges.data["tfidfembed"])  # [edge_num, out_dim]
        z2 = torch.cat([edges.src['z'], edges.dst['z'], dfeat], dim=1)  # [edge_num, 3 * out_dim]
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h, o):
        wnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        cnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        cwedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 1) & (edges.dst["unit"] == 0))
        z = self.fc(h)
        i = self.fs(o)
        g.nodes[cnode_id].data['z'] = z
        g.nodes[wnode_id].data['z'] = i
        g.apply_edges(self.edge_attention, edges=cwedge_id)
        g.pull(wnode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        g.edata.pop('e')
        h = g.ndata.pop('sh')
        return h[wnode_id]

class WCGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size, des_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.fs = nn.Linear(des_embed_size, out_dim, bias=False)

        self.feat_fc = nn.Linear(feat_embed_size, out_dim)
        self.attn_fc = nn.Linear(3 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        dfeat = self.feat_fc(edges.data["tfidfembed"])  # [edge_num, out_dim]
        z2 = torch.cat([edges.src['z'], edges.dst['z'], dfeat], dim=1)  # [edge_num, 3 * out_dim]
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h, o):
        wnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        cnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        wcedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 0) & (edges.dst["unit"] == 1))
        z = self.fc(h)
        i = self.fs(o)
        g.nodes[wnode_id].data['z'] = z
        g.nodes[cnode_id].data['z'] = i
        g.apply_edges(self.edge_attention, edges=wcedge_id)
        g.pull(cnode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        g.edata.pop('e')
        h = g.ndata.pop('sh')
        return h[cnode_id]

class WPGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size, des_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.fs = nn.Linear(des_embed_size, out_dim, bias=False)

        self.feat_fc = nn.Linear(feat_embed_size, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']] , dim=1)  # [edge_num, 2 * out_dim]
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h, o):
        #h is neighboor, in that case, they are hidden state of words.
        wnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        pnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        wpedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 0) & (edges.dst["unit"] == 1))
        z = self.fc(h) #shape (num word , passage dim = sent dim )
        i = self.fs(o)
        g.nodes[wnode_id].data['z'] = z
        g.nodes[pnode_id].data['z'] = i
        g.apply_edges(self.edge_attention, edges=wpedge_id)
        g.pull(pnode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        g.edata.pop('e')
        h = g.ndata.pop('sh')
        return h[pnode_id]
    
class PWGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size, des_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.fs = nn.Linear(des_embed_size, out_dim, bias=False)

        self.feat_fc = nn.Linear(feat_embed_size, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']] , dim=1)  # [edge_num, 2 * out_dim]
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h, o):
        #h is neighboor, in that case, they are hidden state of words.
        wnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        pnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        pwedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 1) & (edges.dst["unit"] == 0))
        z = self.fc(h) #shape (num word , passage dim = sent dim )
        i = self.fs(o)
        g.nodes[pnode_id].data['z'] = z
        g.nodes[wnode_id].data['z'] = i
        g.apply_edges(self.edge_attention, edges=pwedge_id)
        g.pull(wnode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        g.edata.pop('e')
        h = g.ndata.pop('sh')
        return h[wnode_id]

class PSGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size, des_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.fs = nn.Linear(des_embed_size, out_dim, bias=False)

        self.feat_fc = nn.Linear(feat_embed_size, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)  # [edge_num, 2 * out_dim]
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h, o):
        #in that case, h is hidden state from passage to sentence
        pnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        psedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 0) & (edges.dst["unit"] == 1))

        z = self.fc(h) # shape (num passage , sent dim = passage dim). Specially, many sentence only receice information one passage
        i = self.fs(o)
        g.nodes[pnode_id].data['z'] = z
        g.nodes[snode_id].data['z'] = i
        g.apply_edges(self.edge_attention, edges=psedge_id)
        g.pull(snode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        g.edata.pop('e')
        h = g.ndata.pop('sh')
        return h[snode_id]

class SPGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size, des_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.fs = nn.Linear(des_embed_size, out_dim, bias=False)

        self.feat_fc = nn.Linear(feat_embed_size, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)  # [edge_num, 2 * out_dim]
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h, o):
        #in that case, h is hidden state from passage to sentence
        pnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        spedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 1) & (edges.dst["unit"] == 0))

        z = self.fc(h) # shape (num passage , sent dim = passage dim). Specially, many sentence only receice information one passage
        i = self.fs(o)
        g.nodes[snode_id].data['z'] = z
        g.nodes[pnode_id].data['z'] = i
        g.apply_edges(self.edge_attention, edges=spedge_id)
        g.pull(pnode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        g.edata.pop('e')
        h = g.ndata.pop('sh')
        return h[pnode_id]

class WSGGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size, des_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.fs = nn.Linear(des_embed_size, out_dim, bias=False)

        self.feat_fc = nn.Linear(feat_embed_size, out_dim)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        # dfeat = self.feat_fc(edges.data["tfidfembed"])  # [edge_num, out_dim]
        # z2 = torch.cat([edges.src['z'], edges.dst['z'], dfeat], dim=1)  # [edge_num, 3 * out_dim]
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)  # [edge_num, 3 * out_dim]
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h, o):
        gnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 2)
        anode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] != 2)
        agedge_id = g.filter_edges(lambda edges: (edges.src["unit"] != 2) & (edges.dst["unit"] == 2))
        z = self.fc(h)
        i = self.fs(o)
        g.nodes[anode_id].data['z'] = z
        g.nodes[gnode_id].data['z'] = i
        g.apply_edges(self.edge_attention, edges=agedge_id)
        g.pull(gnode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        g.edata.pop('e')
        h = g.ndata.pop('sh')
        return h[gnode_id]



# SPOE

class ESPGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size, des_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.fs = nn.Linear(des_embed_size, out_dim, bias=False)

        self.feat_fc = nn.Linear(feat_embed_size, out_dim)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        # dfeat = self.feat_fc(edges.data["tfidfembed"])  # [edge_num, out_dim]
        # z2 = torch.cat([edges.src['z'], edges.dst['z'], dfeat], dim=1)  # [edge_num, 3 * out_dim]
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)  # [edge_num, 3 * out_dim]
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h, o):
        pnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        spedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 0) & (edges.dst["unit"] == 1))
        z = self.fc(h)
        i = self.fs(o)
        g.nodes[snode_id].data['z'] = z
        g.nodes[pnode_id].data['z'] = i
        g.apply_edges(self.edge_attention, edges=spedge_id)
        g.pull(pnode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        g.edata.pop('e')
        h = g.ndata.pop('sh')
        return h[pnode_id]
    
    
class EPSGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size, des_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.fs = nn.Linear(des_embed_size, out_dim, bias=False)

        self.feat_fc = nn.Linear(feat_embed_size, out_dim)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        # dfeat = self.feat_fc(edges.data["tfidfembed"])  # [edge_num, out_dim]
        # z2 = torch.cat([edges.src['z'], edges.dst['z'], dfeat], dim=1)  # [edge_num, 3 * out_dim]
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)  # [edge_num, 3 * out_dim]
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h, o):
        snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        pnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        psedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 1) & (edges.dst["unit"] == 0))
        z = self.fc(h)
        i = self.fs(o)
        g.nodes[pnode_id].data['z'] = z
        g.nodes[snode_id].data['z'] = i
        g.apply_edges(self.edge_attention, edges=psedge_id)
        g.pull(snode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        g.edata.pop('e')
        h = g.ndata.pop('sh')
        return h[snode_id]

class EPOGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size, des_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.fs = nn.Linear(des_embed_size, out_dim, bias=False)

        self.feat_fc = nn.Linear(feat_embed_size, out_dim)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        # dfeat = self.feat_fc(edges.data["tfidfembed"])  # [edge_num, out_dim]
        # z2 = torch.cat([edges.src['z'], edges.dst['z'], dfeat], dim=1)  # [edge_num, 3 * out_dim]
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)  # [edge_num, 3 * out_dim]
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h, o):
        onode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 2)
        pnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        poedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 1) & (edges.dst["unit"] == 2))
        z = self.fc(h)
        i = self.fs(o)
        g.nodes[pnode_id].data['z'] = z
        g.nodes[onode_id].data['z'] = i
        g.apply_edges(self.edge_attention, edges=poedge_id)
        g.pull(onode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        g.edata.pop('e')
        h = g.ndata.pop('sh')
        return h[onode_id]
    
class EOPGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size, des_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.fs = nn.Linear(des_embed_size, out_dim, bias=False)

        self.feat_fc = nn.Linear(feat_embed_size, out_dim)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        # dfeat = self.feat_fc(edges.data["tfidfembed"])  # [edge_num, out_dim]
        # z2 = torch.cat([edges.src['z'], edges.dst['z'], dfeat], dim=1)  # [edge_num, 3 * out_dim]
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)  # [edge_num, 3 * out_dim]
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h, o):
        pnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        onode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 2)
        opedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 2) & (edges.dst["unit"] == 1))
        z = self.fc(h)
        i = self.fs(o)
        g.nodes[onode_id].data['z'] = z
        g.nodes[pnode_id].data['z'] = i
        g.apply_edges(self.edge_attention, edges=opedge_id)
        g.pull(pnode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        g.edata.pop('e')
        h = g.ndata.pop('sh')
        return h[pnode_id]
    
class EOSGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size, des_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.fs = nn.Linear(des_embed_size, out_dim, bias=False)

        self.feat_fc = nn.Linear(feat_embed_size, out_dim)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        # dfeat = self.feat_fc(edges.data["tfidfembed"])  # [edge_num, out_dim]
        # z2 = torch.cat([edges.src['z'], edges.dst['z'], dfeat], dim=1)  # [edge_num, 3 * out_dim]
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)  # [edge_num, 3 * out_dim]
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h, o):
        snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        onode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 2)
        osedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 2) & (edges.dst["unit"] == 0))
        z = self.fc(h)
        i = self.fs(o)
        g.nodes[onode_id].data['z'] = z
        g.nodes[snode_id].data['z'] = i
        g.apply_edges(self.edge_attention, edges=osedge_id)
        g.pull(snode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        g.edata.pop('e')
        h = g.ndata.pop('sh')
        return h[snode_id]
    
class ESOGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size, des_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.fs = nn.Linear(des_embed_size, out_dim, bias=False)

        self.feat_fc = nn.Linear(feat_embed_size, out_dim)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        # dfeat = self.feat_fc(edges.data["tfidfembed"])  # [edge_num, out_dim]
        # z2 = torch.cat([edges.src['z'], edges.dst['z'], dfeat], dim=1)  # [edge_num, 3 * out_dim]
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)  # [edge_num, 3 * out_dim]
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h, o):
        onode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 2)
        snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        soedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 0) & (edges.dst["unit"] == 2))
        z = self.fc(h)
        i = self.fs(o)
        g.nodes[snode_id].data['z'] = z
        g.nodes[onode_id].data['z'] = i
        g.apply_edges(self.edge_attention, edges=soedge_id)
        g.pull(onode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        g.edata.pop('e')
        h = g.ndata.pop('sh')
        return h[onode_id]
    
    
class ESPOEGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size, des_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.fs = nn.Linear(des_embed_size, out_dim, bias=False)

        self.feat_fc = nn.Linear(feat_embed_size, out_dim)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        # dfeat = self.feat_fc(edges.data["tfidfembed"])  # [edge_num, out_dim]
        # z2 = torch.cat([edges.src['z'], edges.dst['z'], dfeat], dim=1)  # [edge_num, 3 * out_dim]
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)  # [edge_num, 3 * out_dim]
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h, o):
        enode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 3)
        sponode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] != 3)
        soedge_id = g.filter_edges(lambda edges: (edges.src["unit"] != 3) & (edges.dst["unit"] == 3))
        z = self.fc(h)
        i = self.fs(o)
        g.nodes[sponode_id].data['z'] = z
        g.nodes[enode_id].data['z'] = i
        g.apply_edges(self.edge_attention, edges=soedge_id)
        g.pull(enode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        g.edata.pop('e')
        h = g.ndata.pop('sh')
        return h[enode_id]

class ESPOSPOGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size, des_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.fs = nn.Linear(des_embed_size, out_dim, bias=False)

        self.feat_fc = nn.Linear(feat_embed_size, out_dim)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        # dfeat = self.feat_fc(edges.data["tfidfembed"])  # [edge_num, out_dim]
        # z2 = torch.cat([edges.src['z'], edges.dst['z'], dfeat], dim=1)  # [edge_num, 3 * out_dim]
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)  # [edge_num, 3 * out_dim]
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h, o):
        spo2node_id = g.filter_nodes(lambda nodes: nodes.data["unit"] != 3)
        sponode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] != 3)
        soedge_id = g.filter_edges(lambda edges: (edges.src["unit"] != 3) & (edges.dst["unit"] != 3))
        z = self.fc(h)
        i = self.fs(o)
        g.nodes[sponode_id].data['z'] = z
        g.nodes[spo2node_id].data['z'] = i
        g.apply_edges(self.edge_attention, edges=soedge_id)
        g.pull(spo2node_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        g.edata.pop('e')
        h = g.ndata.pop('sh')
        return h[spo2node_id]
    
class SENTENTITYGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size, des_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.in_dim = in_dim
        self.fs = nn.Linear(des_embed_size, out_dim, bias=False)

        self.feat_fc = nn.Linear(feat_embed_size, out_dim)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        # dfeat = self.feat_fc(edges.data["tfidfembed"])  # [edge_num, out_dim]
        # z2 = torch.cat([edges.src['z'], edges.dst['z'], dfeat], dim=1)  # [edge_num, 3 * out_dim]
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)  # [edge_num, 3 * out_dim]
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h, o):
        sent_node_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        entity_node_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        sententityedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 0) & (edges.dst["unit"] == 1))
        z = self.fc(h)
        i = self.fs(o)
        g.nodes[sent_node_id].data['z'] = z
        g.nodes[entity_node_id].data['z'] = i
        g.apply_edges(self.edge_attention, edges=sententityedge_id)
        g.pull(entity_node_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        g.edata.pop('e')
        h = g.ndata.pop('sh')
        return h[entity_node_id]

class ENTITYSENTGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size, des_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.in_dim = in_dim
        self.fs = nn.Linear(des_embed_size, out_dim, bias=False)

        self.feat_fc = nn.Linear(feat_embed_size, out_dim)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        # dfeat = self.feat_fc(edges.data["tfidfembed"])  # [edge_num, out_dim]
        # z2 = torch.cat([edges.src['z'], edges.dst['z'], dfeat], dim=1)  # [edge_num, 3 * out_dim]
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)  # [edge_num, 3 * out_dim]
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h, o):
        sent_node_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        entity_node_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        entitysentedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 1) & (edges.dst["unit"] == 0))
        z = self.fc(h)
        i = self.fs(o)
        g.nodes[entity_node_id].data['z'] = z
        g.nodes[sent_node_id].data['z'] = i
        g.apply_edges(self.edge_attention, edges=entitysentedge_id)
        g.pull(sent_node_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        g.edata.pop('e')
        h = g.ndata.pop('sh')
        return h[sent_node_id]

class SENTENTITYGLOBALGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size, des_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.fs = nn.Linear(des_embed_size, out_dim, bias=False)

        self.feat_fc = nn.Linear(feat_embed_size, out_dim)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        # dfeat = self.feat_fc(edges.data["tfidfembed"])  # [edge_num, out_dim]
        # z2 = torch.cat([edges.src['z'], edges.dst['z'], dfeat], dim=1)  # [edge_num, 3 * out_dim]
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)  # [edge_num, 3 * out_dim]
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h, o):
        sententity_node_id = g.filter_nodes(lambda nodes: nodes.data["unit"] != 2)
        global_node_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 2)
        sententityedge_id = g.filter_edges(lambda edges: (edges.src["unit"] != 2) & (edges.dst["unit"] == 2))
        z = self.fc(h)
        i = self.fs(o)
        g.nodes[sententity_node_id].data['z'] = z
        g.nodes[global_node_id].data['z'] = i
        g.apply_edges(self.edge_attention, edges=sententityedge_id)
        g.pull(global_node_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        g.edata.pop('e')
        h = g.ndata.pop('sh')
        return h[global_node_id]