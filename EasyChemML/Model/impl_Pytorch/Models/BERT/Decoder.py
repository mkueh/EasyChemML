import copy

import torch.nn as nn

from EasyChemML.Model.impl_Pytorch.Models.BERT.Embedder import Embedder
from EasyChemML.Model.impl_Pytorch.Models.BERT.BasicLayers.Norm import Norm
from EasyChemML.Model.impl_Pytorch.Models.BERT.Layer.DecoderLayer import DecoderLayer
from EasyChemML.Model.impl_Pytorch.Models.BERT.PositionalEncoder import PositionalEncoder


class Decoder(nn.Module):
    def get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

    def __init__(self, vocab_size, d_model, N, heads, dropout, max_seq_len):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout, max_seq_len=max_seq_len)
        self.layers = self.get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)