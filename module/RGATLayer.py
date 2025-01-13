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
# Unless required by applicable law or agreed to in writing,GC software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeteroRGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.fy = nn.Linear(in_dim, out_dim, bias=False)

        self.feat_fc = nn.Linear(feat_embed_size, out_dim, bias=False)
        self.attn_fc = nn.Linear(3 * out_dim, 1, bias=False)
        self.attn_r1 = nn.Linear(feat_embed_size, feat_embed_size, bias=True)
        self.attn_r2 = nn.Linear(feat_embed_size, 1, bias=True)
        self.linear  = nn.Linear(2 * out_dim, out_dim, bias=True)

        self.relation_edges_embeddings = nn.Embedding(6, feat_embed_size)
        self.relation_edges_embeddings.weight.data.uniform_(-0.1, 0.1)

    def edge_attention(self, edges):
        # dfeat = self.feat_fc(edges.data[])
        # z2 = torch.cat([edges.src['z'], dfeat, edges.dst['z']], dim=1)  # [edge_num, 2 * out_dim]
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)  # [edge_num, 2 * out_dim]
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}
    
    def relation_attention(self, edges):
        r = edges.data['y']
        g = F.sigmoid(self.attn_r2(F.leaky_relu(self.attn_r1(r))))
        return {'e': g}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def rel_message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def rel_reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def node_func(self, g, src_n_id, des_n_id, edge_id, h):

        # h: src
        z = self.fc(h) # shape (num passage , sent dim = passage dim). Specially, many sentence only receice information one passage
        g.nodes[src_n_id].data['z'] = z
        g.apply_edges(self.edge_attention, edges=edge_id)
        g.pull(des_n_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        g.edata.pop('e')
        nh = g.ndata.pop('sh') #[des_n_id] # size: [number of all nodes, out_dim]

        # fh: all (src+des)
        fh = torch.zeros_like(nh).to(h.get_device())
        fh[des_n_id] = nh[des_n_id]
        fh[src_n_id] = h

        return fh # updated nodes embeddings with other not influenced

    def edge_func(self, g, src_n_id, des_n_id, edge_id, type_id, h):
        eh = self.relation_edges_embeddings(torch.LongTensor([type_id]*len(edge_id)).to(h.get_device()))
        z = self.fc(h)
        y = self.fy(eh)
        g.nodes[src_n_id].data['z'] = z
        g.edges[edge_id].data['y'] = y
        g.apply_edges(self.relation_attention, edges=edge_id)
        g.pull(des_n_id, self.rel_message_func, self.rel_reduce_func)
        g.ndata.pop('z')
        g.edata.pop('e')
        nh = g.ndata.pop('sh') #[des_n_id] # size: [number of all nodes, out_dim]

        # fh: all (src+des)
        fh = torch.zeros_like(nh).to(h.get_device())
        fh[des_n_id] = nh[des_n_id]
        fh[src_n_id] = h

        return fh # updated nodes embeddings with other not influenced

    def forward(self, g, h, eh):

        # # A. obtain all kinds of relation edges
        # # in that case, h is hidden state from passage to sentence
        # snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1) # sentence nodes
        # rnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 2) # relation nodes
        # ssedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 1) & (edges.dst["unit"] == 1) & (edges.data["dtype"] == 0))
        # rredge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 2) & (edges.dst["unit"] == 2) & (edges.data["dtype"] == 1))
        # didedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 1) & (edges.dst["unit"] == 2) & (edges.data["dtype"] == 2))
        # dodedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 2) & (edges.dst["unit"] == 1) & (edges.data["dtype"] == 3))
        # rodedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 2) & (edges.dst["unit"] == 1) & (edges.data["dtype"] == 4))
        # ridedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 1) & (edges.dst["unit"] == 2) & (edges.data["dtype"] == 5))

        # edge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 1) & (edges.dst["unit"] == 2))
        # r_edge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 2) & (edges.dst["unit"] == 1))
        # ss_edge_id = g.filter_edges(lambda edges: (edges.data["dtype"] == 0))
        # rr_edge_id = g.filter_edges(lambda edges: (edges.data["dtype"] == 1))

        # # B. build up data
        # data = {
        #     'ssedge_id': {'e': ssedge_id, 'n': [snode_id, snode_id], 't': 0},
        #     'rredge_id': {'e': rredge_id, 'n': [rnode_id, rnode_id], 't': 1},
        #     'didedge_id': {'e': didedge_id, 'n': [snode_id, rnode_id], 't': 2},
        #     'dodedge_id': {'e': dodedge_id, 'n': [rnode_id, snode_id], 't': 3},
        #     'rodedge_id': {'e': rodedge_id, 'n': [rnode_id, snode_id], 't': 4},
        #     'ridedge_id': {'e': ridedge_id, 'n': [snode_id, rnode_id], 't': 5},
        # }

        # data = {
        #     'ss_edge_id': {'e': ss_edge_id, 'n': [snode_id, snode_id], 't': 0},
        #     'rr_edge_id': {'e': rr_edge_id, 'n': [rnode_id, rnode_id], 't': 1},
        #     'edge_id': {'e': edge_id, 'n': [snode_id, rnode_id], 't': 2},
        #     'r_edge_id': {'e': r_edge_id, 'n': [rnode_id, snode_id], 't': 3},
        # }

        # A. obtain all kinds of relation edges
        # in that case, h is hidden state from passage to sentence
        # snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1) # sentence nodes
        rnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 2) # relation nodes
        h_outs = []
        for red_id in rnode_id:
            specific_rel_edge_id = g.filter_edges(lambda edges: (edges.data["des_unit"] == 2) & (edges.data["relation"] == red_id))
            src_n_id = g.edges[specific_rel_edge_id].src
            des_n_id = g.edges[specific_rel_edge_id].dst
            f_h = h[specific_rel_edge_id]
            h_out = self.node_func(g, src_n_id, des_n_id, specific_rel_edge_id, f_h)
            # rnode_id = g.filter_nodes(lambda nodes: (nodes.data["unit"] == 2)) # relation nodes
            h_outs.append(h_out)
        h = torch.mean(torch.stack(h_outs))  # [n_nodes, hidden_size]

        # # C. sum up the relation loop
        # h_outs = []
        # for _, en in data.items():
        #     # # relation edges' embedding attention mechanism
        #     # h_rel = self.edge_func(g, en['n'][0], en['n'][1], en['e'], en['t'], h)
        #     # each relation loop has its aggregation from the connected nodes
        #     h_att = self.node_func(g, en['n'][0], en['n'][1], en['e'], h)
        #     h_out = F.leaky_relu(self.linear(torch.cat((h_att, h_rel), dim=1)))
        #     h_outs.append(h_out)
        # h = torch.mean(torch.stack(h_outs))  # [n_nodes, hidden_size]
            
        return h # return the total h, containing the embeddings of sentence nodes and relation nodes



class SRRGATLayer(nn.Module):
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
        snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        rnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 2)
        sredge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 1) & (edges.dst["unit"] == 2))
        z = self.fc(h)
        i = self.fs(o)
        g.nodes[snode_id].data['z'] = z
        g.nodes[rnode_id].data['z'] = i
        g.apply_edges(self.edge_attention, edges=sredge_id)
        g.pull(rnode_id, self.message_func, self.reduce_func)
        # g.update_all(self.message_func, self.reduce_func)
        g.ndata.pop('z')
        g.edata.pop('e')
        h = g.ndata.pop('sh')
        return h[rnode_id]


class RSRGATLayer(nn.Module):
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
        snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        rnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 2)
        rsedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 2) & (edges.dst["unit"] == 1))
        z = self.fc(h)
        i = self.fs(o)
        g.nodes[rnode_id].data['z'] = z
        g.nodes[snode_id].data['z'] = i
        g.apply_edges(self.edge_attention, edges=rsedge_id)
        g.pull(snode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        g.edata.pop('e')
        h = g.ndata.pop('sh')
        return h[snode_id]


class AGRGATLayer(nn.Module):
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
        gnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        anode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] != 0)
        agedge_id = g.filter_edges(lambda edges: (edges.src["unit"] != 0) & (edges.dst["unit"] == 0))
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
