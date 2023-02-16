from typing import List

from EasyChemML.Metrik.MetricStack import MetricStack
from EasyChemML.Model.AbstractModel import Abstract_Model
from EasyChemML.Utilities.DataUtilities.BatchTable import BatchTable


class ModelEvalJob:
    job_name: str
    trained_Model: Abstract_Model

    X: BatchTable
    X_cols = List[str]

    y: BatchTable
    y_cols = List[str]

    result_metric: MetricStack


    #Todo rework metric Job system
    def __init__(self, job_name: str, trained_Model: Abstract_Model, X: BatchTable, X_cols: List[str], y: BatchTable,
                 y_cols: List[str], metric: MetricStack):
        self.job_name = job_name
        self.trained_Model = trained_Model
        self.X = X
        self.y = y

        self.X_cols = X_cols
        self.y_cols = y_cols
        self.result_metric = metric

    def set_resultMetric(self, result_metric: MetricStack):
        self.result_metric = result_metric

    def set_trained_Model(self, model: Abstract_Model):
        self.trained_Model = model
