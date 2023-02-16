from abc import ABC, abstractmethod

from EasyChemML.Metrik.MetricEnum import MetricDirection, MetricOutputType


class Abstract_Metric(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def calc(self, y_true, y_pred, y_predict_proba):
        raise Exception('not implemented')

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def get_settings(self) -> dict:
        return {}

    @staticmethod
    @abstractmethod
    def getMetric_Outputtype() -> MetricOutputType:
        raise Exception('not implemented')

    @staticmethod
    @abstractmethod
    def getMetricType():
        raise Exception('not implemented')

    @staticmethod
    @abstractmethod
    def getMetricClass():
        raise Exception('not implemented')

    @staticmethod
    @abstractmethod
    def getDirection() -> MetricDirection:
        raise Exception('not implemented')
