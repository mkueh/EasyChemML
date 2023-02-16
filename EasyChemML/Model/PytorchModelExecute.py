from typing import Type, Dict, Any

from EasyChemML.Model.AbstractModel import Abstract_Model, WithBatches, WithEpochs, WithCheckpoints
from EasyChemML.Metrik.MetricStack import MetricClass
import numpy as np

from EasyChemML.Model.impl_Pytorch.Pytorch_Model import Pytorch_Model


class PytorchModelExecute(Abstract_Model, WithBatches, WithEpochs, WithCheckpoints):
    clf: Pytorch_Model = None
    batch_size: int = -1
    epochs: int = -1
    checkpoints_afterIterations:int = -1

    def __init__(self, batchSize: int, epochs: int, pytorchModel: Pytorch_Model, checkpoints_afterIterations: int = -1,
                 log_folder: str = ''):
        self.batch_size = batchSize
        self.epochs = epochs
        self.clf = pytorchModel
        self.checkpoints_afterIterations = checkpoints_afterIterations

    def getCheckpointsAfterIterations(self) -> int:
        return self.checkpoints_afterIterations

    def getCurrentState(self) -> Dict:
        return self.clf.get_currentState()

    def setCurrentState(self, saved_data: Dict[str, Any]):
        self.clf.set_currentState(saved_data)

    def set_param(self, param: dict):
        raise Exception('Not implemented yet')

    def get_param(self) -> dict:
        raise Exception('Not implemented yet')

    def fit(self, X: np.ndarray, y: np.ndarray):
        return self.clf.fit(X, y)

    def predict(self, X):
        raise Exception('Not implemented yet')

    def save_model(self, path: str):
        raise Exception('Not implemented yet')

    def load_model(self, path: str):
        raise Exception('Not implemented yet')

    def getBatchsize(self) -> int:
        return self.batch_size

    def getEpochs(self) -> int:
        return self.epochs

    @staticmethod
    def getMetricMode():
        return MetricClass.generativ
