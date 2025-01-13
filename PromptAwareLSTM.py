import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

from tools.utils import POS2INT, DEP2INT, GED2INT
from module.PositionEmbedding import get_sinusoid_encoding_table

def position_encoding(sentence_size, embedding_dim):
    encoding = np.ones((embedding_dim, sentence_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_dim + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (embedding_dim+1)/2) * (j - (sentence_size+1)/2)
    encoding = 1 + 4 * encoding / embedding_dim / sentence_size
    # Make position encoding of time words identity to avoid modifying them
    encoding[:, -1] = 1.0
    return np.transpose(encoding)

class PromptAwareLSTM(nn.Module):
    def __init__(self, hps, embed):
        super(PromptAwareLSTM, self).__init__()
        self._hps = hps
        self._embed = embed
        self.embed_size = hps.word_emb_dim
        lstm_dim = hps.lstm_hidden_state
        hidden_dim = hps.feat_embed_size
        language_use_blstm_size = len(POS2INT)+len(DEP2INT)+len(GED2INT)

        self.embedding = embed
        self.prompt_encoder = nn.LSTM(self.embed_size, lstm_dim // 2, bidirectional=True)
        self.to_down_sample1 = nn.Linear(self.embed_size, self.embed_size // 2)
        self.to_down_sample2 = nn.Linear(lstm_dim * 2, self.embed_size // 2)
        self.blstm = nn.LSTM(self.embed_size, lstm_dim // 2, bidirectional=True)
        self.project_for_blstm = nn.Linear(lstm_dim, lstm_dim)
        self.attention = nn.Linear(lstm_dim, 1)

        if hps.memn2n:
            self.memn2n = MemN2N(hps, embed)

        if hps.language_use:
            self.language_use_blstm = nn.LSTM(hidden_dim, lstm_dim // 2, bidirectional=True)
            self.to_down_sample4 = nn.Linear(language_use_blstm_size, hidden_dim)
            self.language_use_attention = nn.Linear(lstm_dim, 1)
            self.project_for_language_use_blstm = nn.Linear(lstm_dim, lstm_dim)

        self.fcd = nn.Dropout(p=hps.ffn_dropout_prob)
        self.fc = nn.Linear(lstm_dim * 2 if hps.language_use else lstm_dim, 1)
        if hps.language_use:
            self.list_lstm = nn.LSTM(lstm_dim*2, lstm_dim*2, bidirectional=True)
            self.to_down_sample3 = nn.Linear(lstm_dim*2, lstm_dim)
        else:
            self.list_lstm = nn.LSTM(lstm_dim, lstm_dim, bidirectional=True)
            self.to_down_sample3 = nn.Linear(lstm_dim, lstm_dim // 2)

    def forward(self, data):
        
        # obtain features        
        graph = data.get('G')
        graph.to(self._hps.device)
        graph_c = None
        if self._hps.cefr_info == 'graph_init':
            graph_c = data.get('G_c')
            graph_c.to(self._hps.device)
        graph_itvr = data.get('itvr_G')
        graph_itvr.to(self._hps.device)
        M_G = data.get('M_G')
        if self._hps.language_use:
            G_PDG = data.get('G_PDG')
        M_G.to(self._hps.device)
        if self._hps.pred_method == 'test_wdc':
            DA_G = data.get('DA_G')
            DA_G.to(self._hps.device) # out of memory
        ie_count = data.get('ie_count')
        ir_count = data.get('ir_count')
        bert_sent_boundary = data.get('bert_sent_boundary')
        oie_in_idx = data.get('oie_in_idx')
        spoe2nid_dict = data.get('spoe2nid_dict')
        oie_alng_glv_oieseq = data.get('oie_alng_glv_oieseq')
        oie_alng_bert_oieseq = data.get('oie_alng_bert_oieseq')

        # Encode prompts
        embedded_prompts = self.set_sent_feature(graph_itvr) # w^{p}_{t} convey to word embed
        # embedded_prompts = embedded_prompts[:, 512, :]# debug length
        _, (hidden, _) = self.prompt_encoder(embedded_prompts) # e^{p}_{t}
        prompt_vector = torch.cat((hidden[:,0,:], hidden[:,-1,:]), dim = 1).view(1, -1) # v_p

        # Process responses
        embedded_responses = self.set_sent_feature(graph)
        # embedded_responses = embedded_responses[:, 512, :]# debug length
        prompt_vector = prompt_vector.unsqueeze(1).expand(embedded_responses.shape[0], embedded_responses.shape[1], -1)
        embedded_responses = self.to_down_sample1(embedded_responses)
        prompt_vector = self.to_down_sample2(prompt_vector)

        if self._hps.memn2n:
            for responses in self.get_sent_wids(graph).split(ie_count):
                for i, _ in enumerate(responses):
                    # self.memn2n(responses[:i+1], self.get_sent_wids(graph_itvr)[i])
                    self.memn2n(responses[:6], self.get_sent_wids(graph_itvr)[i]) # debug


        blstm_input = torch.cat((embedded_responses, prompt_vector), dim=2)
        blstm_output, _ = self.blstm(blstm_input)
        
        # Apply attention for context
        attention_weights = F.softmax(self.attention(blstm_output).squeeze(2), dim=1)
        weighted_output = torch.mul(blstm_output, attention_weights.unsqueeze(2).expand_as(blstm_output))
        representation = weighted_output.sum(dim=1)
        representation = self.project_for_blstm(representation)

        if self._hps.language_use:
            # Process langugae use
            embedded_language_use_responses = self.set_language_use_feature(G_PDG)
            embedded_language_use_responses = self.to_down_sample4(embedded_language_use_responses)
            language_use_blstm_output, _ = self.language_use_blstm(embedded_language_use_responses)

            # Apply attention for context
            language_use_attention_weights = F.softmax(self.language_use_attention(language_use_blstm_output).squeeze(2), dim=1)
            language_use_weighted_output = torch.mul(language_use_blstm_output, language_use_attention_weights.unsqueeze(2).expand_as(language_use_blstm_output))
            language_use_representation = language_use_weighted_output.sum(dim=1)
            language_use_representation = self.project_for_language_use_blstm(language_use_representation)
        
            representation = torch.cat((representation, language_use_representation), dim=-1)

        representation = representation.split(ie_count)
        final_representation = []
        for r in representation:
            r = r.unsqueeze(0)
            _, (r, _)  = self.list_lstm(r)
            r = self.to_down_sample3(r)
            r = torch.cat((r[0,0,:], r[1,-1,:]), dim = 0).view(1, -1)
            final_representation.append(r)
        final_representation = torch.cat(final_representation, dim=0)
        result = self.fc(self.fcd(final_representation))

        # return object
        final_return = {'embed': {'before_gat': {}, 'after_gat': {}},
                        'dec_outputs': {},
                        'results': {}}

        # Final Return
        final_return['record'] = {'M_G': None,
                                'ie': None,
                                'ir': None}
        final_return['embed']['after_gat'] = {'w': None,
                                            's': None,
                                            'p': None}
        final_return['results'] = {'w': None,
                                's': None,
                                'p': result}
        
        return final_return
        

    def get_sent_wids(self, graph):
        snode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        return graph.nodes[snode_id].data["words"]

    def set_sent_feature(self, graph):
        snode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        embedded_sents = self.embedding(graph.nodes[snode_id].data["words"])
        return embedded_sents
    
    def set_language_use_feature(self, graph):
        snode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        pos = self.one_hot_encode(graph.nodes[snode_id].data["pos"], len(POS2INT))
        dep = self.one_hot_encode(graph.nodes[snode_id].data["dep"], len(DEP2INT))
        ged = self.one_hot_encode(graph.nodes[snode_id].data["ged"], len(GED2INT))
        return torch.cat([pos, dep, ged], dim=-1)

    def one_hot_encode(self, sequence, vocab_size):
        # Initialize a matrix of zeros with the shape (len(sequence), vocab_size)
        sequence = sequence.to(self._hps.device)
        one_hot_matrix = torch.zeros((len(sequence), sequence.shape[-1], vocab_size)).to(self._hps.device)

        # Perform one-hot encoding
        one_hot_matrix = torch.nn.functional.one_hot(sequence, num_classes=vocab_size).float()

        return one_hot_matrix

# https://github.com/KingAndQueen/memn2n
# https://github.com/zshihang/MemN2N/blob/master/model.py
# https://github.com/nmhkahn/MemN2N-pytorch/blob/master/memn2n/model.py
class MemN2N(nn.Module):
    def __init__(self, hps, embed):
        super(MemN2N, self).__init__()

        self._hps = hps
        self.embedding_dim = hps.word_emb_dim
        self.sent_max_len = hps.sent_max_len
        self.max_hops = 20

        self.C = nn.ModuleList([])
        for _ in range(0, self.max_hops+1):
            C = nn.Embedding(hps.vocab_size, self.embedding_dim, padding_idx=0)
            C.weight.data.normal_(0, 0.1)
            self.C.append(C)

        self.softmax = nn.Softmax()
        self.encoding = Variable(torch.FloatTensor(
            position_encoding(self.sent_max_len, self.embedding_dim)), requires_grad=False)

        # self.encoding = nn.Embedding.from_pretrained(
        #     get_sinusoid_encoding_table(self.sent_max_len + 1, self.embedding_dim, padding_idx=0), freeze=True)

    def forward(self, story, query):
        """
        query: [1, num_tokens] -> a reponse or utterance
        story: [num_sentences, num_tokens]
        """
        story_size = story.shape

        u = list()
        query_embed = self.C[0](query) # [num_tokens, token_embed_dim]

        # weired way to perform reduce_dot
        encoding = self.encoding.expand_as(query_embed).to(self._hps.device) # [num_tokens, token_embed_dim]
        u.append(torch.sum(query_embed*encoding, dim=0).unsqueeze(0)) # u[-1]: [1, token_embed_dim]

        # inside memory, the tokens in sentence sequence are retained
        for hop in range(0, self.max_hops):
            embed_A = self.C[hop](story.view(story.shape[0], -1)) # [num_sentences, num_tokens, token_embed_dim]
            embed_A = embed_A.view(story_size+(embed_A.shape[-1],)) # [num_sentences, num_tokens, token_embed_dim]
            encoding = self.encoding.unsqueeze(0).expand_as(embed_A).to(self._hps.device) # [num_sentences, num_tokens, token_embed_dim]
            
            m_A = torch.sum(embed_A*encoding, dim=1) # [num_sentences, token_embed_dim]
            u_temp = u[-1].expand_as(m_A) # [num_sentences, token_embed_dim]
            prob   = self.softmax(torch.sum(m_A*u_temp, dim=1)) # [num_sentences]

            embed_C = self.C[hop+1](story.view(story.shape[0], -1)) # [num_sentences, num_tokens, token_embed_dim]
            embed_C = embed_C.view(story_size+(embed_C.shape[-1],)) # [num_sentences, num_tokens, token_embed_dim]
            
            m_C     = torch.sum(embed_C*encoding, 1) # [num_sentences, token_embed_dim]
            prob = prob.unsqueeze(1).expand_as(m_C) # [num_sentences, token_embed_dim]
            o_k  = torch.sum(m_C*prob, 0).unsqueeze(0) # [num_sentences]

            u_k = u[-1] + o_k
            u.append(u_k)

        print('u[-1]: ', u[-1].shape)
        print('t: ', (self.C[self.max_hops].weight.transpose(0, 1)).shape)
        input()

        a_hat = u[-1]@self.C[self.max_hops].weight.transpose(0, 1)

        return a_hat, self.softmax(a_hat)