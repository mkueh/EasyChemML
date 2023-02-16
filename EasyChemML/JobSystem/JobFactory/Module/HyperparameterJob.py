from EasyChemML.Model.AbstractModel.Abstract_Model import Abstract_Model
from EasyChemML.Metrik.MetricStack import MetricStack

from typing import Dict, Type, List

from EasyChemML.Splitter.Splitcreator import Split
from EasyChemML.Utilities.DataUtilities.BatchTable import BatchTable


# Todo copy of TrainEval without saved Model
class HyperparameterJob:
    job_name:str
    result_metric_TEST: MetricStack
    result_metric_TRAIN: MetricStack

    algorithm: Type[Abstract_Model]
    explicit_algorithm_para:Dict

    X: BatchTable
    X_cols = List[str]
    y: BatchTable
    y_cols = List[str]

    split: Split
    evalMetric: MetricStack

    def __init__(self, job_name: str, algorithm: Type[Abstract_Model], explicit_algorithm_para: Dict,
                 X: BatchTable,
                 X_cols: List[str], y: BatchTable, y_cols: List[str], evalMetric: MetricStack, split: Split):
        self.job_name = job_name
        self.X = X
        self.X_cols = X_cols
        self.y = y
        self.y_cols = y_cols

        self.algorithm = algorithm
        self.explicit_algorithm_para = explicit_algorithm_para

        self.evalMetric = evalMetric
        self.split = split

    def __repr__(self):
        return f'HyperJOB: JobID:' + str(self.job_id) + ' | Config: ' + str(
            self.config.config_id) + ' | Outer_Step: ' + str(self.config.outer_index) + ' | Inner_Step: ' + str(
            self.inner_Step)

    def set_resultMetric(self, metric: MetricStack):
        self.evalMetric = metric
