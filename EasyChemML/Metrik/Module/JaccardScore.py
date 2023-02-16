from typing import Dict

from sklearn.metrics import jaccard_score

from .Abstract_Metric import Abstract_Metric
from EasyChemML.Metrik.MetricStack import MetricClass
from EasyChemML.Metrik.MetricEnum import MetricType, MetricOutputType, MetricDirection


class JaccardScore(Abstract_Metric):
    settings: Dict

    def __init__(self, settings: Dict = {}):
        self._settings = settings
        super().__init__()

    def calc(self, y_true, y_pred, y_predict_proba, **kwargs):
        return jaccard_score(y_true, y_pred, **kwargs)

    @staticmethod
    def getMetric_Outputtype():
        return MetricOutputType.singleValue

    @staticmethod
    def getMetricType():
        return MetricType.unkown

    @staticmethod
    def getMetricClass():
        return MetricClass.classification

    @staticmethod
    def getItemname():
        return "jaccard_score"

    @staticmethod
    def getDirection():
        return MetricDirection.higherIsBetter
