from typing import Type, Dict, Union

import numpy as np, torch
import torch.nn.functional

from torch import device
from torch.optim import Optimizer
from torch.autograd import Variable

from EasyChemML.Model.impl_Pytorch.Models.AbstractPytorchModel import AbstractPytorchModel


class Pytorch_Model:
    __model: Union[AbstractPytorchModel, torch.nn.Module]
    __pytorch_device: device
    __optimiser: Optimizer

    def __init__(self, model: Union[AbstractPytorchModel, torch.nn.Module], optimiser: Type[Optimizer],
                 optimiser_kwargs: Dict, torchDevice_str: str = 'cuda:0'):
        self.__model = model
        self.__pytorch_device = torch.device(torchDevice_str)
        self.__model.set_pytorchDevice(self.__pytorch_device)
        self.__optimiser = optimiser(params=self.__model.parameters(), **optimiser_kwargs)

    def __init_internal_paramters(self):
        for p in self.__model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def __fit_CalcLoss(self, X: np.ndarray, y: np.ndarray) -> float:
        X: torch.Tensor = torch.LongTensor(X).to(self.__pytorch_device)
        y: torch.Tensor = torch.LongTensor(y).to(self.__pytorch_device)

        preds = self.__model.fit(X, y)
        del X

        self.__optimiser.zero_grad()
        targets = y[:, 1:].contiguous().view(-1)
        del y

        loss = torch.nn.functional.cross_entropy(preds.view(-1, preds.size(-1)), targets)
        loss.backward()
        self.__optimiser.step()

        return torch.Tensor.item(loss.data)

    def fit(self, X: np.ndarray, y: np.ndarray):
        return self.__fit_CalcLoss(X, y)

    def get_currentState(self):
        currentState = {'model_state_dict': self.__model.get_state_dict(),
                        'optimizer_state_dict': self.__optimiser.state_dict()}
        return currentState

    def set_currentState(self, currentState: Dict):
        self.__model.set_state_dict(currentState['model_state_dict'])
        self.__optimiser.load_state_dict(currentState['optimizer_state_dict'])
