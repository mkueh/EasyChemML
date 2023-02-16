from typing import List
from EasyChemML.Utilities.DataUtilities.BatchTable import BatchAccess
from EasyChemML.Utilities.DataUtilities.BatchDatatyp import BatchDatatypClass
from EasyChemML.Utilities.DataUtilities.BatchDatatypHolder import BatchDatatypHolder

import math
import os, shutil, numpy as np
from EasyChemML.Utilities.DataUtilities.BatchPartition import BatchPartition

from EasyChemML.Utilities.ParallelUtilities.ParallelHelper import ParallelHelper
from EasyChemML.Utilities.ParallelUtilities.IndexQueues.IndexQueue_settings import IndexQueue_settings
from EasyChemML.Utilities.ParallelUtilities.Shared_PythonList import Shared_PythonList


class test_object():
    val = None
    test1 = 10
    test2 = 10
    test3 = 10
    test4 = 10

    def __init__(self, value, value2, value3):
        self.val = value + 1
        self.test1 = value2
        self.test2 = value3
        self.test3 = 10
        self.test4 = 10


def initFiles(name: str = 'memoryDisk.h5'):
    if not os.path.exists('./TMP'):
        print('TMP_PATH not found ... i create one for you')
        os.mkdir('./TMP')
    else:
        shutil.rmtree('./TMP')
        os.mkdir('./TMP')

    if not os.path.exists('./TMP/Output'):
        print('Output not found ... i create one for you')
        os.mkdir('./TMP/Output')
    else:
        shutil.rmtree('./TMP/Output')
        os.mkdir('./TMP/Output')

    dataHolder = BatchPartition(os.path.join('./TMP', 'memoryDisk.h5'), 500000)

    columns = ['STRINGS', 'INTS', 'FLOATS']
    dtypes = BatchDatatypHolder()
    dtypes['STRINGS'] = BatchDatatypClass.NUMPY_STRING
    dtypes['INTS'] = BatchDatatypClass.NUMPY_INT32
    dtypes['FLOATS'] = BatchDatatypClass.NUMPY_FLOAT32
    dataHolder.createDatabase('X', dtypes, 1000)

    x = dataHolder['X']

    for i in range(len(x)):
        val = x[i]
        x[i] = (str(i), int(i), float(i) / 10)

    return dataHolder


def _test_parallel(input_arr: List, out_dtypes, current_chunk: int):
    output_arr = []
    for current_index in current_chunk:
        output_arr.append(input_arr[current_index]+1000)
    return output_arr


def test_Shared_PythonList_objects_inRange():
    test_data = []

    for i in range(1000):
        test_data.append(i)

    for i in range(900,1000):
        test_data[i] = i

    parallel_executer = ParallelHelper(12)
    IQ_settings = IndexQueue_settings(start_index=900, end_index=len(test_data), chunksize=10)
    out = parallel_executer.execute_function_returnArrays(_test_parallel, IQ_settings, np.dtype(int), input_arr=test_data)

    last_val = -1
    for item in out:
        if item < 1900:
            assert False
        if last_val == -1:
            last_val = item
        elif last_val + 1 == item:
            last_val = item
        else:
            assert False


def test_Shared_PythonList_objects():
    dataholder_new = initFiles('memoryDisk_1.h5')
    x_table = dataholder_new['X']

    iterator: BatchAccess = iter(x_table)
    batch: np.ndarray
    dataTypHolder: BatchDatatypHolder = x_table.getDatatypes()
    dataTypHolder['FLOATS'] = BatchDatatypClass.PYTHON_OBJECT

    for batch in iterator:
        out = dataTypHolder.createAEmptyNumpyArray(len(batch))
        check_value = 0
        for i, item in enumerate(batch):
            check_value += item['FLOATS']
            out[i]['STRINGS'] = item['STRINGS']
            out[i]['INTS'] = item['INTS']
            out[i]['FLOATS'] = test_object(item['FLOATS'], i, i + 10)
        iterator <<= out

    iterator: BatchAccess = iter(x_table)
    batch: np.ndarray
    dataTypHolder: BatchDatatypHolder = x_table.getDatatypes()
    dataTypHolder['FLOATS'] = BatchDatatypClass.NUMPY_FLOAT32

    for batch in iterator:
        shared_batch = Shared_PythonList(batch, x_table.getDatatypes())
        out = dataTypHolder.createAEmptyNumpyArray(len(batch))
        parallel_executer = ParallelHelper(12)

        IQ_settings = IndexQueue_settings(start_index=0, end_index=len(x_table), chunksize=10)
        out = parallel_executer.execute_map_orderd_return(_parallel_objectManipulation, IQ_settings, out.dtype,
                                                          columns=['FLOATS'], input_arr=shared_batch)
        iterator <<= out
        shared_batch.destroy()

    iterator: BatchAccess = iter(x_table)
    batch: np.ndarray
    value_float = 0
    for batch in iterator:
        last_val = -1
        for item in batch:
            value_float += item['FLOATS']

            if last_val == -1:
                last_val = item['INTS']
            elif last_val + 1 == item['INTS']:
                last_val = item['INTS']
            else:
                assert False

    assert math.isclose(check_value + 1000.0, value_float, rel_tol=0.01, abs_tol=0.0)


def _parallel_objectManipulation(input_arr: Shared_PythonList, columns: List[str], out_dtypes, current_chunk: int):
    out_array = np.empty(shape=(len(current_chunk),), dtype=out_dtypes)
    index_counter = 0
    for current_index in current_chunk:
        for exists_col in list(input_arr.getcolumns()):
            if exists_col in columns:
                value = input_arr[current_index][exists_col].val
                out_array[index_counter][exists_col] = value
            else:
                out_array[index_counter][exists_col] = input_arr[current_index][exists_col]
        index_counter += 1

    return out_array
