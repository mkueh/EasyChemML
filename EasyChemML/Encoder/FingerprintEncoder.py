import copy

from .impl_FingerprintEncoder.FingerprintGenerator import FingerprintGenerator, FingerprintGenerator_Mode
from EasyChemML.Utilities.DataUtilities.BatchTable import BatchTable, BatchAccess
from EasyChemML.Utilities.DataUtilities.BatchDatatyp import BatchDatatyp, BatchDatatypClass
from EasyChemML.Utilities.DataUtilities.BatchDatatypHolder import BatchDatatypHolder
from enum import Enum

import numpy as np, sys, traceback
from typing import List, Dict

from EasyChemML.Utilities.ParallelUtilities.IndexQueues.IndexQueue_settings import IndexQueue_settings
from EasyChemML.Utilities.ParallelUtilities.ParallelHelper import ParallelHelper
from EasyChemML.Utilities.ParallelUtilities.Shared_PythonList import Shared_PythonList


class FingerprintTyp(Enum):
    RDKit = ('rdkit', BatchDatatypClass.NUMPY_INT8, 'var')
    ECFP = ('morgan_circular', BatchDatatypClass.NUMPY_INT8, 'var')
    ECFC = ('morgan_circular_count', BatchDatatypClass.NUMPY_INT32, 'var')
    AVALON = ('avalon', BatchDatatypClass.NUMPY_INT8, 'var')
    LAYERDFINGERPRINT = ('layerdfingerprint', BatchDatatypClass.NUMPY_INT8, 'var')
    MACCS = ('maccs', BatchDatatypClass.NUMPY_INT8, '167')
    ATOM_PAIRS = ('atom_pairs', BatchDatatypClass.NUMPY_INT8, 'var')
    TOPOLOGICAL_TORSIONS = ('topological_torsions', BatchDatatypClass.NUMPY_INT8, 'var')


class FingerprintHolder:
    fingerprint_typ: FingerprintTyp
    fingerprint_settings: Dict

    def __init__(self, fingerprint_typ: FingerprintTyp, fingerprint_settings: Dict):
        self.fingerprint_typ = fingerprint_typ
        self.fingerprint_settings = fingerprint_settings


class FingerprintEncoder:

    def convert(self, datatable: BatchTable, columns: List[str], n_jobs: int, fingerprints: List[FingerprintHolder], createNewColumns: List[str] = [],
                ignore_errors: bool = False, return_nonZero_indices: bool = False, default_NoneToke: int = -1, offsetForNonZeroIndices:int=0):
        """
        Regenerates Fingerprints out of rdkitMol-objects

        Parameters
        ----------
        datatable : BatchTable
            The input Batchtable

        n_jobs : int
            On how many Threads should the job use

        columns : List[str]
            Defines the coulumns which are converted to RDKITMol

        createNewColumns : List[str] Create a new column for every column in columns. if you have 3 define in columns
        you need also define 3 new columns in createNewColumns

        offsetForNonZeroIndices: int
            add this value to all nonZero indicies to shift the whole fingerprint by this value

        """
        fp_names = []
        fp_settings = []

        for fingerprint in fingerprints:
            fp_names.append(fingerprint.fingerprint_typ.value)
            fp_settings.append(fingerprint.fingerprint_settings)

        self.__convertData(datatable, columns, n_jobs, createNewColumns, fp_names, fp_settings, ignore_errors, return_nonZero_indices,
                           default_NoneToke, offsetForNonZeroIndices=offsetForNonZeroIndices)

    def __convertData(self, InputBuffer: BatchTable, columns: List[str], n_jobs: int, createNewColumns: List[str], fp_names: List, fp_settings: List,
                      ignore_errors: bool, return_nonZero_indices: bool, default_NoneToke: int = 0, offsetForNonZeroIndices:int = 0):
        if return_nonZero_indices:
            FG = FingerprintGenerator(mode=FingerprintGenerator_Mode.NONZERO_INDICES, default_NoneToke=default_NoneToke, offsetForNonZeroIndices=offsetForNonZeroIndices)
        else:
            FG = FingerprintGenerator(mode=FingerprintGenerator_Mode.PLAIN_FP, default_NoneToke=default_NoneToke, offsetForNonZeroIndices=offsetForNonZeroIndices)

        size = int(np.sum(FG.getFullShape(fp_names, fp_settings)))
        minimum_dtype = FG.getMinimumDtyp(fingerprints=fp_names)

        iterator: BatchAccess = iter(InputBuffer)
        batch: np.ndarray
        dataTypHolder: BatchDatatypHolder = InputBuffer.getDatatypes()

        if len(createNewColumns) > 0:
            if len(createNewColumns) == len(columns):
                for column in createNewColumns:
                    dataTypHolder[column] = BatchDatatyp(minimum_dtype, (size,))
            else:
                raise Exception(
                    'You want to create new columns, but not for all columns are new columns are specified in '
                    'createNewColumns list')
        elif len(createNewColumns) == 0:
            for column in columns:
                dataTypHolder[column] = BatchDatatyp(minimum_dtype, (size,))

        for batch in iterator:
            out = dataTypHolder.createAEmptyNumpyArray(len(batch))
            shared_batch = Shared_PythonList(batch, InputBuffer.getDatatypes())
            parallel_executer = ParallelHelper(n_jobs)
            IQ_settings = IndexQueue_settings(start_index=0, end_index=len(batch), chunksize=128)
            out = parallel_executer.execute_map_orderd_return(self._parallel_convert, IQ_settings, out.dtype,
                                                              input_arr=shared_batch, columns=columns, createNewColumns=createNewColumns,
                                                              fp_names=fp_names,
                                                              fp_settings=fp_settings,
                                                              ignore_errors=ignore_errors,
                                                              FP_GENERATOR=FG)

            iterator <<= out
            shared_batch.destroy()

    def _parallel_convert(self, input_arr: Shared_PythonList, columns: List[str], out_dtypes, current_chunk: List[int],
                          fp_names, fp_settings, ignore_errors: bool, FP_GENERATOR: FingerprintGenerator, createNewColumns:List[str]):
        out_array = np.empty(shape=(len(current_chunk),), dtype=out_dtypes)
        empty_arr = FP_GENERATOR.generateArrOfFingerprints(['NA'], fp_names, fp_settings)
        index_counter = 0

        for current_index in current_chunk:
            try:
                col_arr = FP_GENERATOR.generateArrOfFingerprints(input_arr[current_index][columns], fp_names,
                                                                 fp_settings)
            except Exception as e:
                print(f'Data (row: {current_index}) can not translate in a fingerprint')
                if ignore_errors:
                    col_arr = copy.copy(empty_arr)
                else:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    traceback.print_tb(exc_traceback, limit=6, file=sys.stdout)
                    print('Exception : ' + str(e))
                    raise Exception('Data could not be converted')

            if len(createNewColumns) == 0:
                for exists_col in list(input_arr.getcolumns()):
                    if exists_col not in columns:
                        out_array[index_counter][exists_col] = input_arr[current_index][exists_col]

                for i, arr in enumerate(col_arr):
                    out_array[index_counter][columns[i]] = col_arr[i]
            else:
                for exists_col in list(input_arr.getcolumns()):
                    out_array[index_counter][exists_col] = input_arr[current_index][exists_col]

                for i, arr in enumerate(createNewColumns):
                    out_array[index_counter][createNewColumns[i]] = col_arr[i]

            index_counter += 1
        return out_array


    @staticmethod
    def is_parallel():
        return True

    @staticmethod
    def convert_foreach_outersplit():
        return False
