from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt

from EasyChemML.Encoder.impl_Tokenizer.SmilesTokenizer_SchwallerEtAll import SmilesTokenzier
from typing import List
from enum import Enum

from EasyChemML.Utilities.DataUtilities.BatchTable import BatchTable, BatchAccess
from EasyChemML.Utilities.DataUtilities.BatchDatatyp import BatchDatatypClass
from EasyChemML.Utilities.DataUtilities.BatchDatatypHolder import BatchDatatypHolder
from EasyChemML.Utilities.ParallelUtilities.ParallelHelper import ParallelHelper
from EasyChemML.Utilities.ParallelUtilities.IndexQueues.IndexQueue_settings import IndexQueue_settings
from EasyChemML.Utilities.ParallelUtilities.Shared_PythonList import Shared_PythonList
import math, numpy as np


class Descriptors(Enum):
    MolMass: 0


class EasyDescriptor:

    def convert(self, datatable: BatchTable, columns: List[str], n_jobs: int, list_of_descriptor: List[Descriptors] = [], column_suffix:str ='_descriptor', decimal_places=0):
        """
        Generates descriptors for each column.
        The converter generates per column a new column with a list of all descriptors.

        Parameters
        ----------
        list_of_descriptor
        datatable : BatchTable
            The input batchtable

        n_jobs : int
            On how many Threads should the job use

        columns : List[str]
            Defines the columns which are converted to RDKITMol

        list_of_descriptor : List[Descriptors]
            The Descriptors that the converter should generate

        column_suffix : str
        the suffix that should be added at the new generated column

        decimal_places : int
        round at decimal_places
        """
        self.__convertEasyDescriptor(datatable, columns=columns, n_jobs=n_jobs, list_of_descriptor=list_of_descriptor, column_suffix=column_suffix, decimal_places=decimal_places)

    def __convertEasyDescriptor(self, datatable: BatchTable, columns: List[str], n_jobs: int,
                                list_of_descriptor: List[Descriptors], column_suffix, decimal_places):
        iterator: BatchAccess = iter(datatable)
        batch: np.ndarray
        dataTypHolder: BatchDatatypHolder = datatable.getDatatypes()

        for column in columns:
            new_columnName = column + column_suffix

            if new_columnName in datatable.getColumns():
                raise Exception(f'a column with the name {new_columnName} already exist')

            dataTypHolder[new_columnName] = BatchDatatypClass.NUMPY_INT32
            dataTypHolder[new_columnName].set_shape((1,))

        for batch in iterator:
            shared_batch = Shared_PythonList(batch, datatable.getDatatypes())
            parallel_executer = ParallelHelper(n_jobs)
            IQ_settings = IndexQueue_settings(start_index=0, end_index=len(batch), chunksize=128)
            out = parallel_executer.execute_map_orderd_return(self._parallel_convert, IQ_settings,
                                                              dataTypHolder.toNUMPY_dtypes(),
                                                              columns=columns, input_arr=shared_batch,
                                                              column_suffix=column_suffix, decimal_places=decimal_places)
            iterator <<= out
            shared_batch.destroy()

    def _parallel_convert(self, input_arr: Shared_PythonList, columns: List[str], out_dtypes, current_chunk: int,  column_suffix: str, decimal_places:int):
        out_array = np.empty(shape=(len(current_chunk),), dtype=out_dtypes)
        index_counter = 0
        for current_index in current_chunk:
            for exists_col in list(input_arr.getcolumns()):
                if exists_col in columns:
                    if isinstance(input_arr[current_index][exists_col], float):
                        if math.isnan(input_arr[current_index][exists_col]):
                            out_array[current_index][exists_col] = 'NA'
                    elif input_arr[current_index][exists_col] == 'nan' or input_arr[current_index][exists_col] == 'NA':
                        out_array[current_index][exists_col] = 'NA'

                    else:
                        descritors = None
                        new_columnName_descritors = exists_col + column_suffix

                        try:
                            descritors = round(ExactMolWt(input_arr[current_index][exists_col]),decimal_places)

                            if decimal_places == 0:
                                descritors = int(descritors)

                        except Exception as e:
                            print(e)

                        if descritors is None:
                            print("cant convert " + str(input_arr[current_index][exists_col]))
                            print(f'set {current_index} {new_columnName_descritors} to NA')

                            out_array[index_counter][exists_col] = input_arr[current_index][exists_col]
                            out_array[index_counter][new_columnName_descritors] = 'NA'
                        else:
                            out_array[index_counter][exists_col] = input_arr[current_index][exists_col]
                            out_array[index_counter][new_columnName_descritors] = descritors
                else:
                    out_array[index_counter][exists_col] = input_arr[current_index][exists_col]
            index_counter += 1
        return out_array

    @staticmethod
    def isparallel():
        return False

    @staticmethod
    def convert_foreach_outersplit():
        return False
