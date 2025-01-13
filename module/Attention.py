import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim, use_dropout=True):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(0.5 if use_dropout else 0.)

    def attention(self, a, b):
        a = a.unsqueeze(1)
        b = b.unsqueeze(1).permute(0, 2, 1)
        weights = torch.bmm(b, a)
        weights = F.softmax(weights, dim=1)
        return torch.bmm(a, torch.transpose(weights, 1, 2)).squeeze(1)

    def forward(self, a, b):
        attn_output = self.attention(a, b)
        return a + self.fc(self.dropout(attn_output))
