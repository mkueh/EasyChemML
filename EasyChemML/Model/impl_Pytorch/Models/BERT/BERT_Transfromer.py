from typing import Any, Dict, OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch import device
from torch.optim import Optimizer
from torch.autograd import Variable

from EasyChemML.Model.impl_Pytorch.Models.AbstractPytorchModel import AbstractPytorchModel
from EasyChemML.Model.impl_Pytorch.Models.BERT.__BERT_Transformer_nn import BERT_Transformer_nn
from EasyChemML.Model.impl_Pytorch.Models.BERT.Decoder import Decoder
from EasyChemML.Model.impl_Pytorch.Models.BERT.Encoder import Encoder


class BERT_Transformer(AbstractPytorchModel):
    __pytorch_nn_model = None
    __pytorch_device: device = None

    __max_seq_len: int = 300

    def __init__(self, src_vocab, trg_vocab, N, heads, d_model, dropout, max_seq_len):
        self.__pytorch_nn_model = BERT_Transformer_nn(src_vocab=src_vocab, trg_vocab=trg_vocab, N=N, heads=heads, d_model=d_model,
                                                      dropout=dropout, max_seq_len=max_seq_len)
        self.__max_seq_len = max_seq_len

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> Any:
        self.__pytorch_nn_model.train()

        X = X[:, 0:self.__max_seq_len]

        trg_input = y[:, :-1]
        src_mask, trg_mask = self.__create_masks(X, trg_input, self.__pytorch_device)
        preds = self.__pytorch_nn_model(X, trg_input, src_mask, trg_mask)
        return preds

    def __create_masks(self, input_seq, trg_input, pytorch_device):
        # input_pad = EN_TEXT.vocab.stoi['<pad>']
        # creates mask with 0s wherever there is padding in the input
        input_msk = (input_seq != 0).unsqueeze(-2)

        target_seq = trg_input
        # target_pad = FR_TEXT.vocab.stoi['<pad>']
        target_msk = (target_seq != 0).unsqueeze(1)
        size = target_seq.size(1)  # get seq_len for matrix
        nopeak_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
        nopeak_mask = Variable(torch.from_numpy(nopeak_mask) == 0).to(pytorch_device)

        target_msk = target_msk & nopeak_mask

        return input_msk, target_msk

    def parameters(self):
        return self.__pytorch_nn_model.parameters()

    def set_pytorchDevice(self, pytorch_device: device):
        if not self.__pytorch_device is None:
            raise Exception('It is not possible to override device when it is set once')

        self.__pytorch_device = pytorch_device
        self.__pytorch_nn_model.to(self.__pytorch_device)

        if self.__pytorch_device is None:
            raise Exception('No device for pytorch is selected')

    def get_state_dict(self):
        return self.__pytorch_nn_model.state_dict()

    def set_state_dict(self, state_dict: OrderedDict[str,torch.Tensor]):
        self.__pytorch_nn_model.load_state_dict(state_dict)
