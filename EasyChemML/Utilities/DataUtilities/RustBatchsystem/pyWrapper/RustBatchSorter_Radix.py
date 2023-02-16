import os.path
from typing import Dict, List

import numpy as np
from numpy.lib import recfunctions as rfn

from EasyChemML.Utilities.DataUtilities.BatchDatatyp import BatchDatatypClass
from EasyChemML.Utilities.DataUtilities.RustBatchsystem.pyRustBatchsystem import BatchHolder_py
from EasyChemML.Utilities.DataUtilities.RustBatchsystem.pyRustBatchsystem import BatchSorter_Radix_py
from EasyChemML.Utilities.DataUtilities.RustBatchsystem.pyWrapper.RustBatchholder import RustBatchholder


class RustBatchSorter_Radix:
    _sorter: BatchSorter_Radix_py

    def __init__(self, tmp_path: str):
        self._sorter = BatchSorter_Radix_py(os.path.join(tmp_path, 'sort_tmp'))

    def sort(self, rustbatchholder: RustBatchholder, tableName):
        bt = rustbatchholder.getRustBatchTable(tableName)
        dtype = rustbatchholder.rustBatchtable[tableName][1]
        dtype = BatchDatatypClass.get_by_lvl(dtype.get_highest_number_complexity())

        if dtype == dtype.NUMPY_INT8:
            self._sorter.sort_i8(bt)
        elif dtype == dtype.NUMPY_INT16:
            self._sorter.sort_i16(bt)
        elif dtype == dtype.NUMPY_INT32:
            self._sorter.sort_i32(bt)
        elif dtype == dtype.NUMPY_INT64:
            self._sorter.sort_i64(bt)
        elif dtype == dtype.NUMPY_FLOAT32:
            raise Exception('float32 is not sortable at the moment')
        elif dtype == dtype.NUMPY_FLOAT64:
            raise Exception('float64 is not sortable at the moment')
        else:
            raise Exception('datatype is not sortable at the moment')
