from rdkit import Chem
from EasyChemML.Encoder.AbstractEncoder.AbstractEncoder import AbstractEncoder
from Utilities.Dataset import Dataset
from typing import List
from EasyChemML.Utilities.Application_env import Application_env
from Utilities.DataUtilities.BatchTable import BatchTable, Batch_Access
from Utilities.DataUtilities.BatchDatatyp import BatchDatatypClass
from Utilities.DataUtilities.BatchDatatypHolder import BatchDatatypHolder

import math, numpy as np
from multiprocessing import freeze_support, RLock
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


class MolRdkitConverter(AbstractEncoder):

    def __init__(self, APP_ENV:Application_env):
        super().__init__(APP_ENV)

    """
    Parameter
    coulmns: defines the coulumns which are converted to RDKITMol
    nanvalue: defines the NAN value for that is used in the dataset
    """
    def convert(self, dataset:Dataset, columns:List[str], n_jobs:int, **kwargs):
        self.__convertSMILEstoMOL(InputBuffer=dataset.getFeature_data(), columns=columns, n_jobs=n_jobs)

    def __convertSMILEstoMOL(self, InputBuffer:BatchTable, columns:List[str], n_jobs:int):
        iterator:Batch_Access = iter(InputBuffer)
        batch: np.ndarray
        dataTypHolder: BatchDatatypHolder = InputBuffer.getDatatypes()

        for column in columns:
            dataTypHolder[column] = BatchDatatypClass.PYTHON_OBJECT

        for batch in iterator:
            out = dataTypHolder.createAEmptyNumpyArray(len(batch))
            self._runParallel(batch, out, columns, n_jobs)
            iterator <<= out

    def _runParallel(self, batch, out, columns, n_jobs):
        freeze_support()
        with SharedMemoryManager() as smm:
            # Create a shared memory of size np_arry.nbytes
            batch_shm_manager = smm.SharedMemory(batch.nbytes)
            # Create a np.recarray using the buffer of shm
            shm_batch = np.recarray(shape=batch.shape, dtype=batch.dtype, buf=batch_shm_manager.buf)
            # Copy the data into the shared memory
            np.copyto(shm_batch, batch)
            # Spawn some processes to do some work
            tmp = {}
            with ProcessPoolExecutor(n_jobs, initargs=(RLock(),), initializer=tqdm.set_lock) as exe:
                fs = [exe.submit(self._parallel_convert, batch_shm_manager.name, batch.shape, batch.dtype, out.dtype, columns, worker_index, n_jobs)
                      for worker_index in range(n_jobs)]

        for _ in as_completed(fs):
            pass

        for i, f in enumerate(fs):
            tmp[i] = list(f.result())

        current_start = 0
        items_perThread = math.ceil(len(batch) / n_jobs)
        for x in range(n_jobs):
            if x == n_jobs-1:
                current_end = len(batch)
                tmp_start = 0
                tmp_end = len(batch) - current_start
            else:
                current_end = current_start + items_perThread
                tmp_start = 0
                tmp_end = items_perThread
            out[current_start:current_end] = tmp[x][tmp_start:tmp_end]
            current_start += items_perThread


    def _parallel_convert(self, batch_sh, batch_shape, batch_dtyp, outdtyp, columns, worker_index, worker_count):
        tqdm_text = "#" + "{}".format(worker_index).zfill(3)

        # Locate the shared memory by its name
        batch_shm = SharedMemory(batch_sh)
        batch_array = np.recarray(shape=batch_shape, dtype=batch_dtyp, buf=batch_shm.buf)
        items_perThread = math.ceil(len(batch_array)/worker_count)
        out_array = np.empty(shape=items_perThread, dtype=outdtyp)
        with tqdm(total=items_perThread, desc=tqdm_text, position=worker_index, leave=True) as pbar:
            for index in range(items_perThread):
                pbar.update(1)
                current_batch_index = worker_index * items_perThread + index
                if current_batch_index >= len(batch_array):
                    return out_array
                raise_exception = False
                for exists_col in list(batch_array.dtype.names):
                    if exists_col in columns:

                        if isinstance(batch_array[current_batch_index][exists_col], float):
                            if math.isnan(batch_array[current_batch_index][exists_col]):
                                out_array[index][exists_col] = 'NA'
                                pass
                        elif batch_array[current_batch_index][exists_col] == 'nan' or batch_array[current_batch_index][exists_col] == 'NA':
                            out_array[index][exists_col] = 'NA'
                            pass
                        else:
                            mol = None
                            try:
                                mol = Chem.MolFromSmiles(batch_array[current_batch_index][exists_col])
                            except Exception as e:
                                print(f'Data (row: __ ,column:{current_batch_index}) can not translate in a RDKit mol object')
                                print('Exception : ' + str(e))
                                raise_exception = True
                            if mol is None:
                                print("cant convert " + str(batch_array[current_batch_index][exists_col]))
                            out_array[index][exists_col] = mol

                    else:
                        out_array[index][exists_col] = batch_array[current_batch_index][exists_col]
                if raise_exception:
                    raise Exception('Data could not be converted')
        return out_array

    @staticmethod
    def getItemname():
        return "e_rdkitmol"

    @staticmethod
    def isparallel():
        return False