from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import multilabel_confusion_matrix

from .Abstract_Metric import Abstract_Metric
from EasyChemML.Metrik.MetricEnum import MetricType, MetricClass, MetricOutputType, MetricDirection
import deepdiff
from typing import Dict


class SpecificityMatrix(Abstract_Metric):
    # True Negatives / True Negatives + False Negatives

    _settings: Dict

    def __init__(self, settings: Dict = {}):
        self._settings = settings
        super().__init__()

    def calc(self, y_true, y_pred, y_predict_proba):
        return self._calc(y_true, y_pred, y_predict_proba, **self._settings)

    def _calc(self, y_true, y_pred, y_predict_proba, average='macro'):
        labels_true = unique_labels(y_true)
        labels_pred = unique_labels(y_pred)

        lable_set = set.union(set(labels_true), set(labels_pred))
        labels_full = list(lable_set)
        c_ma = multilabel_confusion_matrix(y_true, y_pred, labels=labels_full)

        out = {}
        for i,label in enumerate(lable_set):
            tn, fp, fn, tp = c_ma[i].ravel()
            out[f'{label}'] = tn / (tn + fp)
        return out

    def getMetric_Outputtype(self):
        return MetricOutputType.matrixValue

    def getMetricType(self):
        return MetricType.unkown

    def __eq__(self, other):
        if isinstance(other, SpecificityMatrix):
            result = len(deepdiff.DeepDiff(self._settings, other.get_settings())) == 0
            return result
        else:
            return False

    @staticmethod
    def getMetricClass():
        return MetricClass.classification

    def get_settings(self) -> dict:
        return self._settings

    @staticmethod
    def getDirection():
        return MetricDirection.higherIsbetter
