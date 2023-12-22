from typing import List

import numpy as np
from sklearn.decomposition import IncrementalPCA

from EasyChemML.Utilities.DataUtilities.BatchDatatyp import BatchDatatyp, BatchDatatypClass
from EasyChemML.Utilities.DataUtilities.BatchDatatypHolder import BatchDatatypHolder
from EasyChemML.Utilities.DataUtilities.BatchTable import BatchTable, BatchAccess


class SklearnPCA():

    def __init__(self):
        pass

    def convert(self, batchtable: BatchTable, columns: List[str], n_jobs: int, n_components: int = 32):
        return self._singleThreaded_generatePCA(batchtable, columns, n_jobs, n_components)

    def _singleThreaded_generatePCA(self, InputBuffer: BatchTable, columns: List[str], n_jobs: int, n_components: int):
        iterator: BatchAccess = iter(InputBuffer)
        batch: np.ndarray
        dataTypHolder: BatchDatatypHolder = InputBuffer.getDatatypes()

        column_pca = {}
        for column in columns:
            dataTypHolder[column] = BatchDatatyp(BatchDatatypClass.NUMPY_FLOAT32, (n_components,))
            column_pca[column] = IncrementalPCA(n_components=n_components, batch_size=InputBuffer.getChunksize())

        iterator: BatchAccess = iter(InputBuffer)
        for batch in iterator:
            for column in columns:
                column_pca[column].fit(batch[column])

        iterator: BatchAccess = iter(InputBuffer)
        for batch in iterator:
            out = dataTypHolder.createAEmptyNumpyArray(len(batch))

            for x, row in enumerate(batch):
                for exists_col in list(InputBuffer.getColumns()):
                    if exists_col not in columns:
                        out[x][exists_col] = row[exists_col]

            for column in columns:
                out[column] = column_pca[column].transform(batch[column])

            iterator <<= out

    @staticmethod
    def getItemname():
        return "prepro_sklear_pca"

    @staticmethod
    def is_parallel():
        return False

    @staticmethod
    def convert_foreach_outersplit():
        return True
