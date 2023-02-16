from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
from torch import device


class AbstractPytorchModel(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def parameters(self):
        pass

    @abstractmethod
    def set_pytorchDevice(self, pytorch_device: device):
        pass

    @abstractmethod
    def get_state_dict(self):
        pass

    @abstractmethod
    def set_state_dict(self, state_dict: Dict):
        pass
