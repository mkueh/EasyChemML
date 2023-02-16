import math
from typing import List, Tuple, Dict
from EasyChemML.Utilities.DataUtilities.BatchTable import BatchTable, BatchAccess
from EasyChemML.Utilities.DataUtilities.BatchDatatyp import BatchDatatypClass
from EasyChemML.Utilities.DataUtilities.BatchDatatypHolder import BatchDatatypHolder
from EasyChemML.Utilities.ParallelUtilities.ParallelHelper import ParallelHelper
from EasyChemML.Utilities.ParallelUtilities.IndexQueues.IndexQueue_settings import IndexQueue_settings
from EasyChemML.Utilities.ParallelUtilities.Shared_PythonList import Shared_PythonList
import numpy as np


def writeResultsBack(val: Tuple, out_array: np.ndarray, createNewColumns: List, columns: List[str],
                       exists_col: str, index_counter: int):
    if len(createNewColumns) > 0:
        out_array[index_counter][exists_col] = val[1]
        indexOfNewcol = columns.index(exists_col)
        output_col = createNewColumns[indexOfNewcol]
    else:
        output_col = exists_col

    out_array[index_counter][output_col] = val[0]


def checkIfTypIsSMILES(val: str) -> bool:
    # check for NAN
    if isinstance(val, float):
        if math.isnan(val):
            return False
    elif val == 'nan' or val == 'NA' or val == b'ValueNotFound' or val == 'ValueNotFound':
        return False
    return True

class Utility_Converter():

    @staticmethod
    def convertBatches_mapOrderdReturn_outerFunc(InputBuffer: BatchTable, columns: List[str], n_jobs: int,
                                       createNewColumns: List[str], createNewColumn_dtype: Dict[str, BatchDatatypClass],
                                       executeFunction, excuteFunction_args:Dict,
                                       chunksize: int = 128, **kwargs) -> BatchTable:
        """
        executeFunction gets excuteFunction_args,
                             IQ_settings,
                             dataTypHolder.toNUMPY_dtypes(),
                             columns=columns,
                             createNewColumns=createNewColumns,
                             input_arr=shared_batch,
        as parameter
        """
        if isinstance(InputBuffer, BatchTable):
            iterator: BatchAccess = iter(InputBuffer)
            batch: np.ndarray
            dataTypHolder: BatchDatatypHolder = InputBuffer.getDatatypes()

            if len(createNewColumns) > 0:
                if len(createNewColumns) == len(columns):
                    for column in createNewColumns:
                        dataTypHolder[column] = createNewColumn_dtype[column]
                else:
                    raise Exception(
                        'You want to create new columns, but not for all columns are new columns are specified in '
                        'createNewColumns list')

            for batch in iterator:
                shared_batch = Shared_PythonList(batch, InputBuffer.getDatatypes())
                parallel_executer = ParallelHelper(n_jobs)
                IQ_settings = IndexQueue_settings(start_index=0, end_index=len(batch), chunksize=chunksize)
                out = parallel_executer.execute_map_orderd_return(executeFunction, IQ_settings,
                                                                  dataTypHolder.toNUMPY_dtypes(),
                                                                  columns=columns, createNewColumns=createNewColumns,
                                                                  input_arr=shared_batch, **excuteFunction_args)

                iterator <<= out
                shared_batch.destroy()
        else:
            raise Exception('Encoder can only process BatchTable')

        return InputBuffer

    @staticmethod
    def converter_innerFunc_Out1(input_arr: Shared_PythonList, columns: List[str], createNewColumns: List[str],
                                 out_dtypes, current_chunk: int, convert_func, **kwargs):
        out_array = np.empty(shape=(len(current_chunk),), dtype=out_dtypes)
        index_counter = 0

        for current_index in current_chunk:
            for exists_col in list(input_arr.getcolumns()):
                if exists_col in columns:

                    # check if there is a SMILES-like Typ
                    if not checkIfTypIsSMILES(input_arr[current_index][exists_col]):
                        writeResultsBack(('NA', 'NA'), out_array, createNewColumns, columns, exists_col,
                                               index_counter)

                    # try to translate
                    else:
                        input_data = input_arr[current_index][exists_col]

                        out_one = convert_func(input_data, **kwargs)

                        if out_one is None:
                            print("cant convert " + str(input_arr[current_index][exists_col]))
                            print(f'set {current_index} {exists_col} to NA')
                            writeResultsBack(('NA'), out_array, createNewColumns, columns, exists_col,
                                                   index_counter)
                        else:
                            writeResultsBack((out_one), input_data, out_array, createNewColumns,
                                                   columns,
                                                   exists_col, index_counter)
                else:
                    out_array[index_counter][exists_col] = input_arr[current_index][exists_col]
            index_counter += 1

        return out_array

    @staticmethod
    def converter_innerFunc_Out2(input_arr: Shared_PythonList, columns: List[str], createNewColumns: List[str],
                                 out_dtypes, current_chunk: int, convert_func, **kwargs):
        out_array = np.empty(shape=(len(current_chunk),), dtype=out_dtypes)
        index_counter = 0

        for current_index in current_chunk:
            for exists_col in list(input_arr.getcolumns()):
                if exists_col in columns:

                    # check if there is a SMILES-like Typ
                    if not checkIfTypIsSMILES(input_arr[current_index][exists_col]):
                        writeResultsBack(('NA', 'NA'), out_array, createNewColumns, columns, exists_col,
                                         index_counter)

                    # try to translate
                    else:
                        input_data = input_arr[current_index][exists_col]

                        out_one, out_two = convert_func(input_data, **kwargs)

                        if out_one is None or out_two is None:
                            print("cant convert " + str(input_arr[current_index][exists_col]))
                            print(f'set {current_index} {exists_col} to NA')
                            writeResultsBack(('NA', 'NA'), out_array, createNewColumns, columns, exists_col,
                                             index_counter)
                        else:
                            writeResultsBack((out_one, out_two), input_data, out_array, createNewColumns,
                                             columns,
                                             exists_col, index_counter)
                else:
                    out_array[index_counter][exists_col] = input_arr[current_index][exists_col]
            index_counter += 1

        return out_array

