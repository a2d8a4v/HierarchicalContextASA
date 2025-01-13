import torch
import torch.nn as nn

class MemN2N(nn.Module):
    def __init__(self, hps, embed):
        super(MemN2N, self).__init__()

        self.embedding_dim = hps.n_feature_size
        self.sent_max_len = hps.sent_max_len
        self.max_hops = 20

        self.C = nn.ModuleList([])
        for _ in range(0, self.max_hops+1):
            C = nn.Embedding(len(embed), self.embedding_dim, padding_idx=0)
            C.weight.data.normal_(0, 0.1)
            self.C.append(C)

        self.softmax = nn.Softmax()
        # self.encoding = Variable(torch.FloatTensor(
        #     position_encoding(sentence_size, embedding_dim)), requires_grad=False)

        self.encoding = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(self.sent_max_len + 1, self.embedding_dim, padding_idx=0), freeze=True)


    def forward(self, story, query):
        """
        query: [batch_size, sent_dim] -> a reponse or utterance
        story: [batch_size, num_sentences, sent_dim]
        """
        story_size = story.shape

        u = list()
        query_embed = self.C[0](query)
        # weired way to perform reduce_dot
        encoding = self.encoding.unsqueeze(0).expand_as(query_embed)
        u.append(torch.sum(query_embed*encoding, dim=1))

        for hop in range(0, self.max_hops):
            embed_A = self.C[hop](story.view(story.shape[0], -1))
            embed_A = embed_A.view(story_size+(embed_A.shape[-1],))

            encoding = self.encoding.unsqueeze(0).unsqueeze(1).expand_as(embed_A)
            m_A = torch.sum(embed_A*encoding, 2)
       
            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
            prob   = self.softmax(torch.sum(m_A*u_temp, 2))
        
            embed_C = self.C[hop+1](story.view(story.shape[0], -1))
            embed_C = embed_C.view(story_size+(embed_C.shape[-1],))
            m_C     = torch.sum(embed_C*encoding, 2)
       
            prob = prob.unsqueeze(2).expand_as(m_C)
            o_k  = torch.sum(m_C*prob, 1)
       
            u_k = u[-1] + o_k
            u.append(u_k)
       
        a_hat = u[-1]@self.C[self.max_hops].weight.transpose(0, 1)
        return a_hat, self.softmax(a_hat)