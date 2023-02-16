from typing import Type, Dict, List

from EasyChemML.Model.AbstractModel import Abstract_Model
from EasyChemML.Splitter.Splitcreator import Split
from EasyChemML.Utilities.DataUtilities.BatchTable import BatchTable


class ModelTrainJob():
    job_name:str
    algorithm: Type[Abstract_Model]
    explicit_algorithm_para: Dict

    X: BatchTable
    X_cols = List[str]
    y: BatchTable
    y_cols = List[str]

    split: Split

    trained_Model: Abstract_Model

    def __init__(self, job_name: str, algorithm: Type[Abstract_Model], explicit_algorithm_para: Dict, X: BatchTable,
                 X_cols: List[str], y: BatchTable, y_cols: List[str], split: Split):
        self.job_name = job_name
        self.algorithm = algorithm
        self.explicit_algorithm_para = explicit_algorithm_para

        self.X_cols = X_cols
        self.X = X
        self.y_cols = y_cols
        self.y = y

        self.split = split

    def set_result_trained_Model(self, trained_Model: Abstract_Model):
        self.trained_Model = trained_Model
