import torch
import torch.nn as nn
import math
from torch.autograd import Variable


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=300, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        #x = x + Variable(self.pe[:, :seq_len], requires_grad=False).cuda(1)    ## Variable is used for wrapping tensors
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False)
        return self.dropout(x)
