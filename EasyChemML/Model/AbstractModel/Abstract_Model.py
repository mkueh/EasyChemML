from abc import ABC, abstractmethod
from typing import Dict


class Abstract_Model(ABC):

    @abstractmethod
    def fit(self, X, y):
        raise Exception('not implemented yet')

    @abstractmethod
    def predict(self, X):
        raise Exception('not implemented yet')

    @abstractmethod
    def save_model(self, path: str):
        raise Exception('not implemented yet')

    @abstractmethod
    def load_model(self, path: str):
        raise Exception('not implemented yet')

    @staticmethod
    @abstractmethod
    def getMetricMode():
        raise Exception('not implemented yet')
