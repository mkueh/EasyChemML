from EasyChemML.Encoder.AbstractEncoder.AbstractEncoder import AbstractEncoder
from EasyChemML.Utilities.DataUtilities.BatchTable import BatchTable, BatchAccess
from EasyChemML.Utilities.DataUtilities.BatchDatatyp import BatchDatatyp, BatchDatatypClass
from EasyChemML.Utilities.DataUtilities.BatchDatatypHolder import BatchDatatypHolder
import numpy as np
from sortedcontainers import SortedSet
from typing import List, Tuple


class RandomFeatureEncoder(AbstractEncoder):
    from EasyChemML.Utilities.Dataset import Dataset

    def __init__(self):
        super().__init__()

    def convert(self, batchtable:BatchTable, columns:List[str], n_jobs: int, feature_vector_size:int = 32, feature_vector_range=(-1, 1)):
        return self._singleThreaded_generateRandomFeature(batchtable, columns, n_jobs, feature_vector_size, feature_vector_range)

    def _singleThreaded_generateRandomFeature(self, InputBuffer:BatchTable, columns:List[str], n_jobs: int, feature_vector_size: int, feature_vector_range: Tuple[int, int]):
        iterator: BatchAccess = iter(InputBuffer)
        batch: np.ndarray
        dataTypHolder: BatchDatatypHolder = InputBuffer.getDatatypes()

        containItems = SortedSet()
        for batch in iterator:
            for row in batch[columns]:
                for item in row:
                    if not item in containItems:
                        containItems.add(item)

        feature_dict = {}
        for item in containItems:
            feature_dict[item] = np.random.uniform(feature_vector_range[0], feature_vector_range[1], (feature_vector_size,))

        for column in columns:
            dataTypHolder[column] = BatchDatatyp(BatchDatatypClass.NUMPY_FLOAT32, (feature_vector_size,))

        iterator: BatchAccess = iter(InputBuffer)
        for batch in iterator:
            out = dataTypHolder.createAEmptyNumpyArray(len(batch))

            for x, row in enumerate(batch):
                for exists_col in list(InputBuffer.getColumns()):
                    if exists_col not in columns:
                        out[x][exists_col] = row[exists_col]

                for column in columns:
                    out[x][column] = feature_dict[row[column]]

            iterator <<= out

    @staticmethod
    def getItemname():
        return "e_random_feature"

    @staticmethod
    def is_parallel():
        return False

    @staticmethod
    def convert_foreach_outersplit():
        return False