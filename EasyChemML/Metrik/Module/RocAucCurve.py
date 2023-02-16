from sklearn.metrics import roc_curve

from .Abstract_Metric import Abstract_Metric
from EasyChemML.Metrik.MetricEnum import MetricType, MetricClass, MetricOutputType, MetricDirection
import deepdiff
from typing import Dict


class RocAucCurve(Abstract_Metric):

    _settings: Dict

    def __init__(self, settings: Dict = {}):
        self._settings = settings
        super().__init__()

    def calc(self, y_true, y_pred, y_predict_proba):
        lr_fpr, lr_tpr, thresholds = roc_curve(y_true, y_predict_proba[:, 1])
        return {'lr_fpr': lr_fpr, 'lr_tpr': lr_tpr, 'thresholds': thresholds}

    @staticmethod
    def getMetric_Outputtype():
        return MetricOutputType.matrixValue

    @staticmethod
    def getMetricType():
        return MetricType.unkown

    def get_settings(self) -> dict:
        return self._settings

    def __eq__(self, other):
        if isinstance(other, RocAucCurve):
            result = len(deepdiff.DeepDiff(self._settings, other.get_settings())) == 0
            return result
        else:
            return False

    @staticmethod
    def getMetricClass():
        return MetricClass.classification

    @staticmethod
    def getDirection():
        return MetricDirection.mixed
