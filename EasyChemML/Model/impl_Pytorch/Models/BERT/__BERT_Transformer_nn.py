import torch.nn as nn

from EasyChemML.Model.impl_Pytorch.Models.BERT.Decoder import Decoder
from EasyChemML.Model.impl_Pytorch.Models.BERT.Encoder import Encoder


class BERT_Transformer_nn(nn.Module):
    encoder: Encoder
    decoder: Decoder

    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout, max_seq_len):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout, max_seq_len)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout, max_seq_len)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder.forward(src, src_mask)
        d_output = self.decoder.forward(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)

        return output
