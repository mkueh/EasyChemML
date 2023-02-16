from EasyChemML.Encoder.impl_Tokenizer.SmilesTokenizer_SchwallerEtAll import SmilesTokenzier
from typing import List
from EasyChemML.Utilities.DataUtilities.BatchTable import BatchTable, BatchAccess
from EasyChemML.Utilities.DataUtilities.BatchDatatyp import BatchDatatypClass
from EasyChemML.Utilities.DataUtilities.BatchDatatypHolder import BatchDatatypHolder
from EasyChemML.Utilities.ParallelUtilities.ParallelHelper import ParallelHelper
from EasyChemML.Utilities.ParallelUtilities.IndexQueues.IndexQueue_settings import IndexQueue_settings
from EasyChemML.Utilities.ParallelUtilities.Shared_PythonList import Shared_PythonList
import math, numpy as np


class BertTokenizer:

    def convert(self, datatable: BatchTable, columns: List[str], n_jobs: int, suffix_ids: str = '_ids',
                suffix_tokens: str = '_tokens', max_length:int=100, padding:bool=True, truncation:bool=True):
        """
        Generates tokenized representation for a Bert transformer.
        The converter generates two new columns (oldColumnName)_ids and (oldColumnName)_tokens. You can change this suffix
        by set suffix_ids or suffix_tokens.

        Parameters
        ----------
        datatable : BatchTable
            The input Batchtable

        n_jobs : int
            On how many Threads should the job use

        columns : List[str]
            Defines the coulumns which are converted to RDKITMol

        suffix_ids : str
            Define the suffix for the ids columns

        suffix_tokens : str
            Derfine the suffix for the tokens columns

        max_length : int
            set the max_length of the tokenization

        padding : bool
            set if padding should be added

        truncation: bool
            set if the input should be truncated when it is larger than max_length

        """

        self.__convertRXNSMILEStoTOKENS(InputBuffer=datatable, columns=columns, n_jobs=n_jobs, suffix_ids=suffix_ids,
                                        suffix_tokens=suffix_tokens, max_length=max_length, padding=padding, truncation=truncation)

    def __convertRXNSMILEStoTOKENS(self, InputBuffer: BatchTable, columns: List[str], n_jobs: int, suffix_ids: str,
                                   suffix_tokens: str, max_length:int, padding:bool, truncation:bool):
        iterator: BatchAccess = iter(InputBuffer)
        batch: np.ndarray
        dataTypHolder: BatchDatatypHolder = InputBuffer.getDatatypes()

        for column in columns:
            new_columnName_ids = column + suffix_ids
            new_columnName_tokens = column + suffix_tokens

            if new_columnName_tokens in InputBuffer.getColumns():
                raise Exception(f'a column with the name {new_columnName_tokens} already exist')

            if new_columnName_ids in InputBuffer.getColumns():
                raise Exception(f'a column with the name {new_columnName_ids} already exist')

            dataTypHolder[new_columnName_ids] = BatchDatatypClass.NUMPY_INT32
            dataTypHolder[new_columnName_ids].set_shape((100,))
            dataTypHolder[new_columnName_tokens] = BatchDatatypClass.PYTHON_OBJECT

        for batch in iterator:
            shared_batch = Shared_PythonList(batch, InputBuffer.getDatatypes())
            parallel_executer = ParallelHelper(n_jobs)
            IQ_settings = IndexQueue_settings(start_index=0, end_index=len(batch), chunksize=128)
            out = parallel_executer.execute_map_orderd_return(self._parallel_convert, IQ_settings,
                                                              dataTypHolder.toNUMPY_dtypes(),
                                                              columns=columns, input_arr=shared_batch,
                                                              suffix_ids=suffix_ids,
                                                              suffix_tokens=suffix_tokens,max_length=max_length,
                                                              padding=padding, truncation=truncation)
            iterator <<= out
            shared_batch.destroy()

    def _parallel_convert(self, input_arr: Shared_PythonList, columns: List[str], out_dtypes, current_chunk: int,
                          suffix_ids: str, suffix_tokens: str, max_length:int, padding:bool, truncation:bool):
        tokenizer = SmilesTokenzier(max_length=max_length, padding=padding, truncation=truncation)
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
                        token_tokenized_rxn, ids_tokenized_rxn = None, None
                        try:
                            token_tokenized_rxn, ids_tokenized_rxn = tokenizer.encode(input_arr[current_index][exists_col].decode("utf-8"))
                        except Exception as e:
                            print(e)

                        if token_tokenized_rxn is None or ids_tokenized_rxn is None:
                            print("cant convert " + str(input_arr[current_index][exists_col]))
                            print(f'set {current_index} {new_columnName_tokens} to NA')
                            print(f'AND {current_index} {new_columnName_ids} to NA')

                            new_columnName_ids = exists_col + suffix_ids
                            new_columnName_tokens = exists_col + suffix_tokens

                            out_array[index_counter][exists_col] = input_arr[current_index][exists_col]
                            out_array[index_counter][new_columnName_tokens] = 'NA'
                            out_array[index_counter][new_columnName_ids] = 'NA'
                        else:
                            new_columnName_ids = exists_col + suffix_ids
                            new_columnName_tokens = exists_col + suffix_tokens

                            out_array[index_counter][exists_col] = input_arr[current_index][exists_col]
                            out_array[index_counter][new_columnName_tokens] = token_tokenized_rxn
                            out_array[index_counter][new_columnName_ids] = ids_tokenized_rxn

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
