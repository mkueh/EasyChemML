from typing import List, Union, Tuple, Optional
import numpy as np

from EasyChemML.DataImport.Module.Abstract_DataImporter import Abstract_DataImporter
from EasyChemML.Utilities.DataUtilities.BatchDatatypHolder import BatchDatatypHolder
from EasyChemML.Utilities.DataUtilities.BatchPartition import BatchPartition, BatchPartitionMode
from EasyChemML.Utilities.DataUtilities.BatchTable import BatchTable


class HDF5(Abstract_DataImporter):
    _batchPartition:BatchPartition
    _batchTable:BatchTable
    # params
    _path: str
    _tableName: str
    _selection: Optional[np.ndarray]
    _columns: List[str]

    _nJobs:int
    _batchSize:int
    _shape: Tuple[int]

    def __init__(self, path: str, tableName: str, selection: Union[slice, List[int]] = None, columns: List[str] = None,
                 n_jobs: int = 1, batch_size: int = 100000):
        super().__init__()
        self._path = path
        self._batchSize = batch_size
        self._nJobs = n_jobs
        self._columns = columns

        if selection is not None:
            if isinstance(selection, slice):
                if selection.step is None:
                    self._selection = np.asarray(list(range(selection.start, selection.stop)))
                else:
                    self._selection = np.asarray(list(range(selection.start, selection.stop, selection.step)))
            elif isinstance(selection, list):
                self._selection = np.asarray(selection)
            elif isinstance(selection, np.ndarray):
                self._selection = selection
            else:
                raise Exception('selection should be a slice or list')
        else:
            self._selection = None

        self._tableName = tableName

        self._batchPartition = BatchPartition(path, load_existing=True, read_only=True, mode=BatchPartitionMode.direct_UnBufferedIO)
        self._batchTable = self._batchPartition[tableName]

        if self._columns is None:
            self._columns = self._batchTable.getColumns()

        self._shape = self._batchTable.shape()

        if self._selection is not None:
            self._shape = (len(self._selection),)

    def set_columns(self, columns:List[str]):
        self._columns = columns

    def get_columns(self) -> List[str]:
        return self._columns

    def set_selection(self,  selection: Union[slice, List[int]]):
        if isinstance(selection, slice):
            if selection.step is None:
                self._selection = np.asarray(list(range(selection.start, selection.stop)))
            else:
                self._selection = np.asarray(list(range(selection.start, selection.stop, selection.step)))
        elif isinstance(selection, list):
            self._selection = np.asarray(selection)
        elif isinstance(selection, np.ndarray):
            self._selection = selection
        else:
            raise Exception('selection should be a slice or list')

        self._shape = (len(self._selection),)

    def get_selection(self) -> Union[slice, List[int]]:
        return self._selection

    def get_shape(self) -> Tuple[int]:
        if self._selection is not None:
            if isinstance(self._selection, np.ndarray):
                return len(self._selection),
            else:
                raise Exception('selectiontyp unkown')
        else:
            return self._batchTable.shape()

    def get_dataTyps(self) -> BatchDatatypHolder:
        bdt_Holder = self._batchTable.getDatatypes().get_DtypSubset(self._columns)
        return bdt_Holder

    def get_Data(self, indicies: Union[slice, List[int]] = None):
        if isinstance(indicies, slice) or isinstance(indicies, list):
            if isinstance(indicies, slice):
                data = self._batchTable[self._selection[indicies]]
                return data[self._columns]
            else:
                if self._selection is not None:
                    sel = []
                    for i in indicies:
                        sel.append(self._selection[i])
                    data = self._batchTable[sel]
                    return data[self._columns]
                else:
                    data = self._batchTable[indicies]
                    return data[self._columns]
        else:
            raise Exception('selection should be a slice or list')

    def get_Split(self):
        return None

    def contains_Split(self):
        return False

    def contains_Data(self):
        return False

    def load_InBatches(self) -> int:
        return self._batchSize

    def get_nJobs(self) -> int:
        return self._nJobs


    @staticmethod
    def getTableNames(path:str) -> List[str]:
        bp = BatchPartition(path, load_existing=True, read_only=True)
        return list(bp.keys())

    @staticmethod
    def getShape(path:str, tableName:str) -> Tuple[int]:
        bp = BatchPartition(path, load_existing=True, read_only=True)
        return bp[tableName].shape()
