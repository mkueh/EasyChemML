import torch.nn as nn

from EasyChemML.Model.impl_Pytorch.Models.BERT.BasicLayers.FeedForward import FeedForward
from EasyChemML.Model.impl_Pytorch.Models.BERT.BasicLayers.MultiHeadAttention import MultiHeadAttention
from EasyChemML.Model.impl_Pytorch.Models.BERT.BasicLayers.Norm import Norm


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


