import os.path
from typing import Dict, List, Tuple, Any
from enum import Enum

import numpy as np
from numpy.lib import recfunctions as rfn
from tqdm import tqdm

from EasyChemML.Utilities.DataUtilities.BatchDatatyp import BatchDatatypClass
from EasyChemML.Utilities.DataUtilities.BatchDatatypHolder import BatchDatatypHolder
from EasyChemML.Utilities.DataUtilities.BatchPartition import BatchPartition
from EasyChemML.Utilities.DataUtilities.BatchTable import BatchTable, BatchAccess

from EasyChemML.Utilities.DataUtilities.RustBatchsystem.pyRustBatchsystem import BatchHolder_py
from EasyChemML.Utilities.DataUtilities.RustBatchsystem.pyRustBatchsystem import BatchSorter_Radix_py

class MemoryMode(Enum):
    InMemory = 'InMemory'
    DirectIO = 'DirectIO'

class RustBatchholder:
    rustBatchholder: BatchHolder_py

    rustBatchtable: Dict[str, Tuple[Any, BatchDatatypHolder]]
    rustchunksize = 100000

    def __init__(self, tmp_path: str, rustchunksize:int = 100000):
        self.rustBatchholder = BatchHolder_py(tmp_path)
        self.rustBatchtable = {}
        self.rustchunksize = rustchunksize

    def clean(self):
        self.rustBatchholder.clean()
        self.rustBatchholder = None
        self.rustBatchtable = {}

    def transferToRust(self, bp: BatchPartition, table: str, columns: List[str] = None, memMode: MemoryMode = MemoryMode.InMemory):
        bt = bp[table]

        if columns is None:
            columns = bt.getColumns()

        dataTypHolder: BatchDatatypHolder = bt.getDatatypes()

        if not columns is None:
            dataTypHolder = dataTypHolder.get_DtypSubset(columns)

        self.rustBatchholder.create_new_table(table, list(bt.shape()), memMode.value)
        new_rbt = self._create_typed_BatchTable(dataTypHolder, table)

        self.rustBatchtable[table] = new_rbt, dataTypHolder

        chunksize = self.rustchunksize

        with tqdm(total=len(bt)) as bar:
            for i in range(0, len(bt), chunksize):
                if i + chunksize > len(bt):
                    loaded_chunk = bt.convert_2_ndarray(indicies=list(range(i, len(bt))), columns=columns)
                else:
                    loaded_chunk = bt.convert_2_ndarray(indicies=list(range(i, i + chunksize)), columns=columns)

                new_rbt.add_chunk(loaded_chunk)
                bar.update(len(loaded_chunk))

    def _create_typed_BatchTable(self, dtype: BatchDatatypHolder, tableName:str):
        dtype = BatchDatatypClass.get_by_lvl(dtype.get_highest_number_complexity())

        if dtype == dtype.NUMPY_INT8:
            return self.rustBatchholder.get_batchtable_i8(tableName)
        elif dtype == dtype.NUMPY_INT16:
            return self.rustBatchholder.get_batchtable_i16(tableName)
        elif dtype == dtype.NUMPY_INT32:
            return self.rustBatchholder.get_batchtable_i32(tableName)
        elif dtype == dtype.NUMPY_INT64:
            return self.rustBatchholder.get_batchtable_i64(tableName)
        elif dtype == dtype.NUMPY_FLOAT32:
            return self.rustBatchholder.get_batchtable_f32(tableName)
        elif dtype == dtype.NUMPY_FLOAT64:
            return self.rustBatchholder.get_batchtable_f64(tableName)

    def getRustBatchTable(self, tableName: str):
        if tableName in self.rustBatchtable:
            return self.rustBatchtable[tableName][0]
        else:
            raise Exception(f'rust batchtable with name {tableName} was not found')


    def transferToBatchtable(self, rustTableName: str, bp: BatchPartition, newBatchTableName: str):
        rustbt = self.getRustBatchTable(rustTableName)
        shape = rustbt.shape
        dtype = self.rustBatchtable[rustTableName][1]
        chunksize = self.rustchunksize

        bp.createDatabase(newBatchTableName, dtype, shape)
        bt = bp[newBatchTableName]

        for index, i in enumerate(range(0, len(bt), chunksize)):
            loaded_chunk = rustbt.load_chunk(index)
            loaded_chunk = rfn.unstructured_to_structured(loaded_chunk, dtype.toNUMPY_dtypes())
            if i + chunksize > len(bt):
                bt[i:len(bt)] = loaded_chunk
            else:
                bt[i:i + chunksize] = loaded_chunk
