from typing import Dict

import deepdiff
from sklearn.metrics import f1_score

from .Abstract_Metric import Abstract_Metric
from EasyChemML.Metrik.MetricStack import MetricClass
from EasyChemML.Metrik.MetricEnum import MetricType, MetricOutputType, MetricDirection

class F1_Score(Abstract_Metric):
    _settings: Dict

    def __init__(self, settings: Dict = {}):
        self._settings = settings
        super().__init__()

    def calc(self, y_true, y_pred, y_predict_proba):
        return f1_score(y_true, y_pred, **self._settings)

    def __eq__(self, other):
        if isinstance(other, F1_Score):
            result = len(deepdiff.DeepDiff(self._settings, other.get_settings())) == 0
            return result
        else:
            return False

    def get_settings(self) -> dict:
        return self._settings

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
    def getDirection():
        return MetricDirection.higherIsBetter