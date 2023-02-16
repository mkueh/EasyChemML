import copy

import torch.nn as nn

from EasyChemML.Model.impl_Pytorch.Models.BERT.BasicLayers.Norm import Norm
from EasyChemML.Model.impl_Pytorch.Models.BERT.Embedder import Embedder
from EasyChemML.Model.impl_Pytorch.Models.BERT.Layer.EncoderLayer import EncoderLayer
from EasyChemML.Model.impl_Pytorch.Models.BERT.PositionalEncoder import PositionalEncoder


class Encoder(nn.Module):
    def get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

    def __init__(self, vocab_size, d_model, N, heads, dropout, max_seq_len):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout, max_seq_len=max_seq_len)
        self.layers = self.get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask):
        x = self.embed.forward(src)
        x = self.pe.forward(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm.forward(x)
