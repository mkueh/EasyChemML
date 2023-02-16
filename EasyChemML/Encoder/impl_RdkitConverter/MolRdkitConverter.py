import copy

from EasyChemML.Encoder.AbstractEncoder.AbstractEncoder import AbstractEncoder
from typing import List, Tuple

from EasyChemML.Encoder.AbstractEncoder.WithJobsystem import WithJobsystem
from EasyChemML.Utilities.DataUtilities.BatchTable import BatchTable, BatchAccess
from EasyChemML.Utilities.DataUtilities.BatchDatatyp import BatchDatatypClass
from EasyChemML.Utilities.DataUtilities.BatchDatatypHolder import BatchDatatypHolder
from EasyChemML.Utilities.ParallelUtilities.ParallelHelper import ParallelHelper
from EasyChemML.Utilities.ParallelUtilities.IndexQueues.IndexQueue_settings import IndexQueue_settings
from EasyChemML.Utilities.ParallelUtilities.Shared_PythonList import Shared_PythonList
import math, numpy as np

from EasyChemML.Encoder.impl_RdkitConverter.Utilities.Rdkit_MolGenerator import MolGenerator


class MolRdkitConverter(AbstractEncoder, WithJobsystem):

    def convert_Async2Job(self, datatable: BatchTable, columns: List[str], n_jobs: int,
                AddHs: bool = False, isomericSmiles=False, overrideSmiles=False, createNewColumns: List[str] = [],
                **kwargs):
        """
        Generates RdKit-Molobjects out of SMILES-strings

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

        overrideSmiles : bool
            If createNewColumns is not empty it override the old SMILES with the "new-one" that is generated with
            RDKIT (like the RdkitSmilesConverter). If createNewColumns is empty this option do nothing
            because the column is overridden by the generated molobject

        isomericSmiles : bool
            If true include information about stereochemistry in the SMILES. Defaults to False.

        AddHs : bool
            A molecule/smiles with added Hs
        """
        pass

    def __init__(self):
        super().__init__()

    def convert(self, datatable: BatchTable, columns: List[str], n_jobs: int,
                AddHs: bool = False, isomericSmiles=False, overrideSmiles=False, createNewColumns: List[str] = [],
                **kwargs):
        """
        Generates RdKit-Molobjects out of SMILES-strings

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

        overrideSmiles : bool
            If createNewColumns is not empty it override the old SMILES with the "new-one" that is generated with
            RDKIT (like the RdkitSmilesConverter). If createNewColumns is empty this option do nothing
            because the column is overridden by the generated molobject

        isomericSmiles : bool
            If true include information about stereochemistry in the SMILES. Defaults to False.

        AddHs : bool
            A molecule/smiles with added Hs
        """
        return self.__convertSMILEstoMOL(InputBuffer=datatable, columns=columns, n_jobs=n_jobs, AddHs=AddHs,
                                         isomericSmiles=isomericSmiles, overrideSmiles=overrideSmiles,
                                         createNewColumns=createNewColumns)

    def __convertSMILEstoMOL(self, InputBuffer: BatchTable, columns: List[str],
                             n_jobs: int, AddHs: bool, isomericSmiles: bool, overrideSmiles: bool,
                             createNewColumns: List[str]):

        if isinstance(InputBuffer, BatchTable):
            iterator: BatchAccess = iter(InputBuffer)
            batch: np.ndarray
            dataTypHolder: BatchDatatypHolder = InputBuffer.getDatatypes()

            if len(createNewColumns) > 0:
                if len(createNewColumns) == len(columns):
                    for column in createNewColumns:
                        dataTypHolder[column] = BatchDatatypClass.PYTHON_OBJECT
                else:
                    raise Exception(
                        'You want to create new columns, but not for all columns are new columns are specified in createNewColumns list')
            else:
                if overrideSmiles:
                    overrideSmiles = False

                for column in columns:
                    dataTypHolder[column] = BatchDatatypClass.PYTHON_OBJECT

            for batch in iterator:
                shared_batch = Shared_PythonList(batch, InputBuffer.getDatatypes())
                parallel_executer = ParallelHelper(n_jobs)
                IQ_settings = IndexQueue_settings(start_index=0, end_index=len(batch), chunksize=128)
                out = parallel_executer.execute_map_orderd_return(self._parallel_convert_return, IQ_settings,
                                                                  dataTypHolder.toNUMPY_dtypes(),
                                                                  columns=columns, createNewColumns=createNewColumns,
                                                                  input_arr=shared_batch, AddHs=AddHs,
                                                                  isomericSmiles=isomericSmiles,
                                                                  overrideSmiles=overrideSmiles)

                iterator <<= out
                shared_batch.destroy()
        else:
            raise Exception('Encoder can only process BatchTable')

        return InputBuffer

    def _parallel_convert_return(self, input_arr: Shared_PythonList, columns: List[str], createNewColumns: List,
                                 out_dtypes,
                                 current_chunk: int, AddHs: bool = False, isomericSmiles=False, overrideSmiles=False):
        out_array = np.empty(shape=(len(current_chunk),), dtype=out_dtypes)
        index_counter = 0

        for current_index in current_chunk:
            self._convert(input_arr, out_array, columns, createNewColumns, current_index, index_counter, AddHs,
                              isomericSmiles, overrideSmiles)
            index_counter += 1

        return out_array

    def _convert(self, input_arr: Shared_PythonList, out_array: np.ndarray, columns: List[str], createNewColumns: List,
                current_index: int, index_counter: int, AddHs: bool = False, isomericSmiles=False,
                overrideSmiles=False):
        for exists_col in list(input_arr.getcolumns()):
            if exists_col in columns:

                # check if there is a SMILES-like Typ
                if not self._checkIfTypIsSMILES(input_arr[current_index][exists_col]):
                    self._writeResultsBack(('NA', 'NA', 'NA'), out_array, createNewColumns, columns, exists_col,
                                           index_counter, overrideSmiles)

                # try to translate
                else:
                    input_smiles = input_arr[current_index][exists_col]
                    out_mol, out_smiles = MolGenerator.translateSMILES(input_smiles, AddHs, isomericSmiles)

                    if out_mol is None:
                        print("cant convert " + str(input_arr[current_index][exists_col]))
                        print(f'set {current_index} {exists_col} to NA')
                        self._writeResultsBack(('NA', 'NA', 'NA'), out_array, createNewColumns, columns, exists_col,
                                               index_counter, overrideSmiles)
                    else:
                        self._writeResultsBack((out_mol, out_smiles, input_smiles), out_array, createNewColumns,
                                               columns,
                                               exists_col, index_counter, overrideSmiles)
            else:
                out_array[index_counter][exists_col] = input_arr[current_index][exists_col]

    def _writeResultsBack(self, val: Tuple, out_array: np.ndarray, createNewColumns: List, columns: List[str],
                          exists_col: str, index_counter: int, overrideSmiles: bool):
        if len(createNewColumns) > 0:
            if overrideSmiles:
                out_array[index_counter][exists_col] = val[1]
            else:
                out_array[index_counter][exists_col] = val[2]

            indexOfNewcol = columns.index(exists_col)
            output_col = createNewColumns[indexOfNewcol]
        else:
            output_col = exists_col

        out_array[index_counter][output_col] = val[0]

    def _checkIfTypIsSMILES(self, val: str) -> bool:
        # check for NAN
        if isinstance(val, float):
            if math.isnan(val):
                return False
        elif val == 'nan' or val == 'NA' or val == b'ValueNotFound' or val == 'ValueNotFound':
            return False
        return True

    @staticmethod
    def is_parallel():
        return True

    @staticmethod
    def convert_foreach_outersplit():
        return False
