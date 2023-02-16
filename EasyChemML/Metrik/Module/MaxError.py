from typing import Dict

import deepdiff
from sklearn.metrics import max_error

from .Abstract_Metric import Abstract_Metric
from EasyChemML.Metrik.MetricStack import MetricClass
from EasyChemML.Metrik.MetricEnum import MetricType, MetricOutputType, MetricDirection

class MaxError(Abstract_Metric):

    settings:Dict

    def __init__(self, settings: Dict = {}):
        self._settings = settings
        super().__init__()

    def __eq__(self, other):
        if isinstance(other, MaxError):
            result = len(deepdiff.DeepDiff(self._settings, other.get_settings())) == 0
            return result
        else:
            return False

    def get_settings(self) -> dict:
        return self._settings

    def calc(self, y_true, y_pred, y_predict_proba, **kwargs):
        return max_error(y_true, y_pred, **kwargs)

    @staticmethod
    def getMetric_Outputtype():
        return MetricOutputType.singleValue

    @staticmethod
    def getMetricType():
        return MetricType.absolute

    @staticmethod
    def getMetricClass():
        return MetricClass.regression

    @staticmethod
    def getItemname():
        return "max_error"

    @staticmethod
    def getDirection():
        return MetricDirection.lowerIsBetter