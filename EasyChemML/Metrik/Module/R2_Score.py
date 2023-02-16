from sklearn.metrics import r2_score


from .Abstract_Metric import Abstract_Metric
from EasyChemML.Metrik.MetricEnum import MetricType, MetricClass, MetricOutputType, MetricDirection
import deepdiff
from typing import Dict


class R2_Score(Abstract_Metric):

    _settings: Dict

    def __init__(self, settings: Dict = {}):
        self._settings = settings
        super().__init__()

    def calc(self, y_true, y_pred, y_predict_proba):
        return r2_score(y_true, y_pred, **self._settings)

    def __eq__(self, other):
        if isinstance(other, R2_Score):
            result = len(deepdiff.DeepDiff(self._settings, other.get_settings())) == 0
            return result
        else:
            return False

    @staticmethod
    def getMetric_Outputtype():
        return MetricOutputType.singleValue

    @staticmethod
    def getMetricType():
        return MetricType.relative

    @staticmethod
    def getMetricClass():
        return MetricClass.regression

    def get_settings(self) -> dict:
        return self._settings

    @staticmethod
    def getDirection():
        return MetricDirection.oneIsBest
