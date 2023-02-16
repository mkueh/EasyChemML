from typing import Dict, Type, List

from EasyChemML.Metrik.MetricStack import MetricStack
from EasyChemML.Model.AbstractModel import Abstract_Model
from EasyChemML.Splitter.Splitcreator import Split
from EasyChemML.Utilities.DataUtilities.BatchTable import BatchTable


class ModelTrainEvalJob:
    job_name: str
    result_metric_TEST: MetricStack
    result_metric_TRAIN: MetricStack
    trained_Model: Abstract_Model

    algorithm: Type[Abstract_Model]
    explicit_algorithm_para: Dict

    X: BatchTable
    X_cols = List[str]
    y: BatchTable
    y_cols = List[str]

    split: Split

    targetMetric: MetricStack

    def __init__(self, job_name: str, algorithm: Type[Abstract_Model], explicit_algorithm_para: Dict,
                 X: BatchTable,
                 X_cols: List[str], y: BatchTable, y_cols: List[str], result_metric: MetricStack, split: Split):
        self.job_name = job_name
        self.X = X
        self.X_cols = X_cols
        self.y = y
        self.y_cols = y_cols

        self.algorithm = algorithm
        self.explicit_algorithm_para = explicit_algorithm_para

        self.targetMetric = result_metric
        self.split = split

    def set_resultMetric(self, result_metric_TEST: MetricStack, result_metric_TRAIN: MetricStack):
        self.result_metric_TEST = result_metric_TEST
        self.result_metric_TRAIN = result_metric_TRAIN

    def set_trained_Model(self, model: Abstract_Model):
        self.trained_Model = model
