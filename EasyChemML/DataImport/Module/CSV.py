import numpy
import pandas, numpy as np
from EasyChemML.DataImport.Module.Abstract_DataImporter import Abstract_DataImporter

from typing import Tuple, List, Union

from EasyChemML.Utilities.DataUtilities.BatchDatatypHolder import BatchDatatypHolder


class CSV(Abstract_DataImporter):
    # params
    _path: str
    _columns: List[str]
    _range: Tuple[int, int]

    _data: numpy.ndarray

    def __init__(self, path: str, columns: List[str] = None, range: Tuple[int, int] = None):
        super().__init__()
        self._path = path
        self._columns = columns
        self._range = range

        self._data = self._load()

    def get_shape(self) -> Tuple[int, int]:
        return self._data.shape

    def get_Data(self, indicies: Union[slice, List[int]] = None):
        return self._data[indicies]

    def get_dataTyps(self) -> BatchDatatypHolder:
        batchDtyps:BatchDatatypHolder = BatchDatatypHolder()
        batchDtyps.fromNUMPY_dtyp(self._data.dtype)

        return batchDtyps

    def get_Split(self):
        return None

    def contains_Split(self):
        return False

    def contains_Data(self):
        if self._data is None:
            return False
        else:
            return True

    def _load(self):
        if self._range is None:
            panda_dataframe = pandas.read_csv(self._path, usecols=self._columns, index_col=False)
        else:
            start_range = self._range[0]
            end_range = self._range[1]
            panda_dataframe = pandas.read_csv(self._path, usecols=self._columns, index_col=False)
            panda_dataframe = panda_dataframe[start_range:end_range]
        return np.rec.fromrecords(panda_dataframe, names=panda_dataframe.columns.tolist())

    def load_InBatches(self):
        return False

    def get_nJobs(self) -> int:
        return 1
