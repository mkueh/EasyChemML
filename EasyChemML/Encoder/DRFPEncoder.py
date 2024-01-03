from EasyChemML.Encoder.AbstractEncoder.AbstractEncoder import AbstractEncoder
from EasyChemML.Utilities.DataUtilities.BatchTable import BatchTable, BatchAccess
from EasyChemML.Utilities.DataUtilities.BatchDatatyp import BatchDatatyp, BatchDatatypClass
from EasyChemML.Utilities.DataUtilities.BatchDatatypHolder import BatchDatatypHolder
import numpy as np
from sortedcontainers import SortedSet
from typing import List, Tuple

try:
    from drfp import DrfpEncoder
except ImportError:
    print('DRFP is a optional package, pls install it via pip: pip install drfp')



class DRFPEncoder(AbstractEncoder):
    from EasyChemML.Utilities.Dataset import Dataset

    def __init__(self):
        super().__init__()

    def convert(self, datatable: BatchTable, columns: List[str], n_jobs: int, output_column_name:str, fingerprint_size: int = 2048):
        return self._singleThreaded_generateRandomFeature(datatable, columns, n_jobs, output_column_name,
                                                          fingerprint_size)

    def _singleThreaded_generateRandomFeature(self, InputBuffer: BatchTable, columns: List[str], n_jobs: int,
                                              output_column_name: str, fingerprint_size: int):
        batch: np.ndarray
        dataTypHolder: BatchDatatypHolder = InputBuffer.getDatatypes()

        for column in columns:
            del dataTypHolder[column]

        dataTypHolder[output_column_name] = BatchDatatyp(BatchDatatypClass.NUMPY_INT32, (fingerprint_size,))

        iterator: BatchAccess = iter(InputBuffer)
        for batch in iterator:
            out = dataTypHolder.createAEmptyNumpyArray(len(batch))

            for x, row in enumerate(batch):
                for exists_col in list(InputBuffer.getColumns()):
                    if exists_col not in columns:
                        out[x][exists_col] = row[exists_col]

                reaction_smiles = ''
                for i, column in enumerate(columns):
                    if i > 0:
                        reaction_smiles = reaction_smiles + '.' + row[column].decode("utf-8")
                    else:
                        reaction_smiles = row[column].decode("utf-8")

                encode = DrfpEncoder.encode(reaction_smiles, n_folded_length=fingerprint_size)
                out[x][output_column_name] = encode[0]
            iterator <<= out

    @staticmethod
    def getItemname():
        return "e_drfp"

    @staticmethod
    def is_parallel():
        return False

    @staticmethod
    def convert_foreach_outersplit():
        return False