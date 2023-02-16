from typing import Dict

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from EasyChemML.Metrik.MetricStack import MetricClass
from EasyChemML.Metrik.MetricEnum import MetricType, MetricOutputType, MetricDirection
from .Abstract_Metric import Abstract_Metric

class ConfusionMatrix(Abstract_Metric):

    settings:Dict

    def __init__(self, settings: Dict = {}):
        self._settings = settings
        super().__init__()

    def calc(self, y_true, y_pred, y_predict_proba, **kwargs):
        labels_true = unique_labels(y_true)
        labels_pred = unique_labels(y_pred)

        lable_set = set.union(set(labels_true), set(labels_pred))
        labels_full = list(lable_set)

        c_ma = confusion_matrix(y_true, y_pred, labels=labels_full)

        out = {}
        first_row = []
        first_row.extend(labels_full)
        out['labels'] = first_row
        for i, label in enumerate(lable_set):
            out[f'{label}'] = list(c_ma[i])

        return out

    def getMetric_Outputtype(self):
        return MetricOutputType.matrixValue

    def getMetricType(self):
        return MetricType.unkown

    @staticmethod
    def getMetricClass():
        return MetricClass.classification

    @staticmethod
    def getItemname():
        return "confusion_matrix"

    @staticmethod
    def getDirection():
        return MetricDirection.mixed