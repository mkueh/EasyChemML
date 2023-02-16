from typing import Dict

from .Abstract_Metric import Abstract_Metric
from .SpecificityScore import SpecificityScore
from .RecallScore import RecallScore
from EasyChemML.Metrik.MetricStack import MetricClass
from EasyChemML.Metrik.MetricEnum import MetricType, MetricOutputType, MetricDirection

class BalancedAccuracyScore(Abstract_Metric):

    settings:Dict

    def __init__(self, settings: Dict = {}):
        self._settings = settings
        super().__init__()

    def calc(self, y_true, y_pred, y_predict_proba, **kwargs):
        s_score = SpecificityScore(self.APP_ENV)
        spe_val = s_score.calc(y_true, y_pred, y_predict_proba, average='macro')

        s_score = RecallScore(self.APP_ENV)
        rec_val = s_score.calc(y_true, y_pred, y_predict_proba, average='macro')

        return (spe_val + rec_val) / 2

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
        return "balanced_accuracy_score"

    @staticmethod
    def getDirection():
        return MetricDirection.higherIsBetter