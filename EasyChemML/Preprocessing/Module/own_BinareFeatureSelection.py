from typing import List

from Preprocessing.Module.Abstract_Preprocessing import Abstract_Preprocessing
from Utilities.DataUtilities.BatchTable import BatchTable, Batch_Access
from Utilities.DataUtilities.BatchDatatyp import BatchDatatypClass
from Utilities.DataUtilities.BatchDatatypHolder import BatchDatatypHolder
import numpy as np
from Utilities.Dataset import Dataset
from Preprocessing.Preprocessing import typs
from tqdm import tqdm


#Removed Numba


class own_BinareFeatureSelection(Abstract_Preprocessing):

    def __init__(self, param={}):
        pass

    def __init_list(self, dataTypHolder: BatchDatatypHolder):
        out = dataTypHolder.createAEmptyNumpyArray(1)
        return out

    def _get_indicies_of(self, arr:np.ndarray, value:int):
        out = np.full(len(arr), -1)
        counter = 0

        for i, v in enumerate(arr):
            if v == value:
                out[counter] = i
                counter += 1

        out.resize(counter, refcheck=False)
        return out

    def convert(self, dataset: Dataset, n_jobs: int, typ: typs):
        table: BatchTable
        if typ == typs.TARGET:
            table = dataset.getTargets_data()
        else:
            table = dataset.getFeature_data()

        print('Start BinareFeatureSelection')
        firstItemList = self.__init_list(table.getDatatypes())
        print(f'** BinareFeatureSelection found len: {table.getWith()}')

        dataTypHolder: BatchDatatypHolder = table.getDatatypes()

        if isinstance(firstItemList, np.ndarray):
            self._select_ndarray(table, dataTypHolder.getColumns(), firstItemList)
        else:
            raise Exception(
                f'own_BinareFeatureSelection cannot process this datatyp: {type(firstItemList).__name__}')

        print(f'** BinareFeatureSelection reduce to len: {table.getWith()}')


    def _select_ndarray(self, table: BatchTable, cols: List[str], first_items: np.ndarray):
        remove_indieces_list = {}

        for col in cols:
            remove_indieces_list[col] = np.full(len(first_items[col][0]), 1)

        iterator: Batch_Access = iter(table)
        batch: np.ndarray

        total_count = len(cols) * len(iterator)
        print(f'found {len(cols)} columns and {len(iterator)} batches')
        print(f'this results in cols*batches = {total_count} steps')
        with tqdm(total=total_count, desc='BinareFeatureSelection: ') as pbar:
            for batch in iterator:
                for col in cols:
                    if table.getDatatypes()[col] == BatchDatatypClass.PYTHON_OBJECT:
                        pass
                    else:
                        new_RM_indieces = self._select_ndarray_NUMBA(batch[col], first_items[col][0])
                        np.logical_and(new_RM_indieces, remove_indieces_list[col], out=remove_indieces_list[col])
                    pbar.update(1)

        for col in cols:
            remove_indieces_list[col] = self._get_indicies_of(remove_indieces_list[col], 1)

        iterator: Batch_Access = iter(table)
        dataTypHolder: BatchDatatypHolder = table.getDatatypes()

        for col in cols:
            dataTypHolder[col].set_shape((dataTypHolder[col].get_shape()[0]-len(remove_indieces_list[col]),))

        for batch in iterator:
            out = dataTypHolder.createAEmptyNumpyArray(len(batch))

            for col in cols:
                out[col] = np.delete(batch[col], remove_indieces_list[col], axis=1)
            iterator <<= out

    def _select_ndarray_NUMBA(self, arr:np.ndarray, first_row:np.ndarray):
        remove_indieces = np.full(len(first_row), 1)

        for col_index in range(len(remove_indieces)):
            for row_index in range(len(arr)):
                if not arr[row_index][col_index] == first_row[col_index]:
                    remove_indieces[col_index] = 0
                    break

        return remove_indieces



    def _select_list(self, arr: List):
        pass

    @staticmethod
    def getItemname():
        return "own_BinareFeatureSelection"
