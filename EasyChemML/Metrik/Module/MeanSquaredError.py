from sklearn.metrics import mean_squared_error

from .Abstract_Metric import Abstract_Metric
from EasyChemML.Metrik.MetricEnum import MetricType, MetricClass, MetricOutputType, MetricDirection
import deepdiff
from typing import Dict

class MeanSquaredError(Abstract_Metric):

    _settings: Dict

    def __init__(self, settings: Dict = {}):
        self._settings = settings
        super().__init__()

    def calc(self, y_true, y_pred, y_predict_proba):
        return mean_squared_error(y_true, y_pred, **self._settings)

    @staticmethod
    def getMetric_Outputtype():
        return MetricOutputType.singleValue

    @staticmethod
    def getMetricType():
        return MetricType.unkown

    @staticmethod
    def getMetricClass():
        return MetricClass.regression

    def __eq__(self, other):
        if isinstance(other, MeanSquaredError):
            result = len(deepdiff.DeepDiff(self._settings, other.get_settings())) == 0
            return result
        else:
            return False

    def get_settings(self) -> dict:
        return self._settings

    @staticmethod
    def getDirection():
        return MetricDirection.lowerIsBetter