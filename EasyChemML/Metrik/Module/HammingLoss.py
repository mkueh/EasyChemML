from typing import Dict

from sklearn.metrics import hamming_loss

from .Abstract_Metric import Abstract_Metric
from EasyChemML.Metrik.MetricStack import MetricClass
from EasyChemML.Metrik.MetricEnum import MetricType, MetricOutputType, MetricDirection

class HammingLoss(Abstract_Metric):

    settings:Dict

    def __init__(self, settings: Dict = {}):
        self._settings = settings
        super().__init__()

    def calc(self, y_true, y_pred, y_predict_proba, **kwargs):
        return hamming_loss(y_true, y_pred, **kwargs)

    @staticmethod
    def getMetric_Outputtype():
        return MetricOutputType.singleValue

    @staticmethod
    def getMetricType():
        return MetricType.relative

    @staticmethod
    def getMetricClass():
        return MetricClass.classification

    @staticmethod
    def getItemname():
        return "hamming_loss"

    @staticmethod
    def getDirection():
        return MetricDirection.lowerIsBetter