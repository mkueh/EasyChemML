from typing import List, Union, Optional, Tuple

import numpy as np
from EasyChemML.Model.AbstractModel import Abstract_Model
from EasyChemML.Utilities.DataUtilities.BatchTable import BatchTable
from EasyChemML.Utilities.DataUtilities.BatchPartition import BatchPartition


class ModelPredictJob:
    job_name: str
    trained_Model: Abstract_Model

    X: BatchTable
    X_cols = List[str]
    X_indices: List[int]

    predicted_vals: Union[BatchTable, List, np.numarray]
    writeInBatchSystem: Tuple[BatchPartition, str] = False
    skipSavePrediction: bool = False
    processInBatches: bool = False

    def __init__(self, job_name: str, trained_Model: Abstract_Model, X: BatchTable, X_cols: List[str],
                 X_indices: Optional[List[int]] = None, processInBatches: bool = False,
                 writeInBatchSystem: Tuple[BatchPartition, str] = False,
                 skipSavePrediction: bool = False):
        self.job_name = job_name
        self.trained_Model = trained_Model
        self.X = X
        self.X_cols = X_cols
        self.writeInBatchSystem = writeInBatchSystem
        self.skipSavePrediction = skipSavePrediction
        self.processInBatches = processInBatches

        if X_indices is not None:
            self.X_indices = X_indices
        else:
            self.X_indices = list(range(len(X)))

    def set_predicted_vals(self, predicted_vals: Union[BatchTable, List, np.numarray]):
        self.predicted_vals = predicted_vals
