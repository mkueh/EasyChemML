from EasyChemML.Encoder.AbstractEncoder.AbstractEncoder import AbstractEncoder
from EasyChemML.Utilities.DataUtilities.BatchTable import BatchTable, BatchAccess
from EasyChemML.Utilities.DataUtilities.BatchDatatyp import BatchDatatyp, BatchDatatypClass
from EasyChemML.Utilities.DataUtilities.BatchDatatypHolder import BatchDatatypHolder
from sortedcontainers import SortedSet
import numpy as np

from typing import List

from EasyChemML.Utilities.ParallelUtilities.IndexQueues.IndexQueue_settings import IndexQueue_settings
from EasyChemML.Utilities.ParallelUtilities.ParallelHelper import ParallelHelper
from EasyChemML.Utilities.ParallelUtilities.Shared_PythonList import Shared_PythonList
from EasyChemML.Utilities.ParallelUtilities.Shared_Value import Shared_Value

class OnehotEncoder(AbstractEncoder):
    from EasyChemML.Utilities.Dataset import Dataset

    def __init__(self):
        super().__init__()

    def fit(self, datatable: BatchTable, columns: List[str], n_jobs: int, **kwargs):
        return

    def convert(self, dataset:Dataset, columns:List[str], n_jobs: int, **kwargs):
        return self._generateOnehot_parallel(dataset.getFeature_data(), columns, n_jobs, **kwargs)

    def _singleThreaded_generateOnehot(self, dataset:Dataset, columns:List[str], n_jobs: int, **kwargs):
        InputBuffer = dataset.getFeature_data()
        iterator:BatchAccess = iter(InputBuffer)
        batch: np.ndarray
        dataTypHolder: BatchDatatypHolder = InputBuffer.getDatatypes()

        containItems = SortedSet()
        for batch in iterator:
            for row in batch:
                for item in row:
                    containItems.add(item)

        new_dataTypHolder = BatchDatatypHolder()
        new_dataTypHolder['onehot'] = BatchDatatyp(BatchDatatypClass.NUMPY_INT8, (len(containItems),))

        iterator: BatchAccess = iter(InputBuffer)
        for batch in iterator:
            out = new_dataTypHolder.createAEmptyNumpyArray(len(batch))

            for x, row in enumerate(batch):
                for y, smile in enumerate(containItems):
                    if smile in row:
                        out[x]['onehot'][y] = 1
                    else:
                        out[x]['onehot'][y] = 0

            iterator <<= out

    def _generateOnehot_parallel(self, InputBuffer:BatchTable, columns:List[str], n_jobs: int, **kwargs):
        #Add Columns_processing
        iterator: BatchAccess = iter(InputBuffer)
        batch: np.ndarray
        dataTypHolder: BatchDatatypHolder = InputBuffer.getDatatypes()

        containItems = SortedSet()
        for batch in iterator:
            for row in batch[columns]:
                for item in row:
                    if not item in containItems:
                        containItems.add(item)

        col_str = 'onehot_'
        for col in columns:
            col_str += col
            del dataTypHolder[col]
        dataTypHolder[col_str] = BatchDatatyp(BatchDatatypClass.NUMPY_INT8, (len(containItems),))


        iterator: BatchAccess = iter(InputBuffer)

        for batch in iterator:
            out = dataTypHolder.createAEmptyNumpyArray(len(batch))
            shared_batch = Shared_PythonList(batch, InputBuffer.getDatatypes())
            shared_containItems = Shared_Value(containItems, BatchDatatypClass.PYTHON_OBJECT)
            parallel_executer = ParallelHelper(n_jobs)
            IQ_settings = IndexQueue_settings(start_index=0, end_index=len(batch), chunksize=512)
            out = parallel_executer.execute_map_orderd_return(self._parallel_func, IQ_settings, out.dtype,
                                                              input_arr=shared_batch, columns=columns,  containItems_val=shared_containItems
                                                              ,new_colName = col_str)
            iterator <<= out


    def _parallel_func(self, input_arr: Shared_PythonList, columns: List[str], containItems_val:Shared_Value, out_dtypes, current_chunk:int, new_colName:str):
        containItems:SortedSet = containItems_val.get_val()
        out_array = np.empty(shape=(len(current_chunk),), dtype=out_dtypes)
        index_counter = 0

        for current_index in current_chunk:
            data = input_arr[current_index][columns]
            for y, smile in enumerate(containItems):
                if smile in data:
                    out_array[index_counter][new_colName][y] = 1
                else:
                    out_array[index_counter][new_colName][y] = 0

            for exists_col in list(input_arr.getcolumns()):
                if exists_col not in columns:
                    out_array[index_counter][exists_col] = input_arr[current_index][exists_col]

            index_counter += 1
        return out_array

    @staticmethod
    def getItemname():
        return "e_onehot"

    @staticmethod
    def isparallel():
        return True

    @staticmethod
    def convert_foreach_outersplit():
        return False