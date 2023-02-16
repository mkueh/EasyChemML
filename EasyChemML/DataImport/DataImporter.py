import math
import os

from tqdm import tqdm

from EasyChemML.DataImport.Module.Abstract_DataImporter import Abstract_DataImporter
from EasyChemML.Utilities.DataUtilities.BatchPartition import BatchPartition, BatchPartitionMode
from EasyChemML.Environment import Environment

from typing import List, Dict, Union

from EasyChemML.Utilities.ParallelUtilities.IndexQueues.IndexQueue_settings import IndexQueue_settings
from EasyChemML.Utilities.ParallelUtilities.ParallelHelper import ParallelHelper


class DataImporter:
    _TMP_path: str
    _WORKING_path: str

    _default_chunksize: int

    def __init__(self, env: Environment):
        self._TMP_path = env.TMP_path
        self._WORKING_path = env.WORKING_path

    def load_data_InExistingBatchPartition(self,
                                           list_of_dataloader: Union[
                                               List[Abstract_DataImporter], Dict[str, Abstract_DataImporter]],
                                           batchpartition: BatchPartition):
        """

        Args:
            batchpartition: existing BatchPartition
            list_of_dataloader: Dict oder List von Dataloader
        """
        if isinstance(list_of_dataloader, list):
            self._load_list(list_of_dataloader, batchpartition)
        elif isinstance(list_of_dataloader, dict):
            self._load_dict(list_of_dataloader, batchpartition)
        else:
            raise Exception(f'list_of_dataloader is a not supported typ: {type(list_of_dataloader)}')

    def load_data_InNewBatchPartition(self, list_of_dataloader: Union[
        List[Abstract_DataImporter], Dict[str, Abstract_DataImporter]],
                                      memDisk_name='memdisk.disk', max_chunksize: int = 10000,
                                      batchmode: BatchPartitionMode = BatchPartitionMode.direct_BufferedIO) -> BatchPartition:
        """

        Args:
            batchdriver: BatchPartition_mode
            max_chunksize: goal chunksize, maximum size of one chunk is 4GB RAM
            memDisk_name: filename of the memdisk
            list_of_dataloader: Dict oder List von Dataloader
        """
        new_batchPartition = BatchPartition(os.path.join(self._TMP_path, memDisk_name), max_chunksize, mode=batchmode)

        if isinstance(list_of_dataloader, list):
            self._load_list(list_of_dataloader, new_batchPartition)
        elif isinstance(list_of_dataloader, dict):
            self._load_dict(list_of_dataloader, new_batchPartition)
        else:
            raise Exception(f'list_of_dataloader is a not supported typ: {type(list_of_dataloader)}')

        return new_batchPartition

    def _load_list(self, dataloader: List[Abstract_DataImporter], new_batchPartition: BatchPartition):
        for i, loader in enumerate(dataloader):
            self._load(str(i), loader, new_batchPartition)

    def _load_dict(self, dataloader: Dict[str, Abstract_DataImporter], new_batchPartition: BatchPartition):
        for key in list(dataloader.keys()):
            self._load(str(key), dataloader[key], new_batchPartition)

    def _load(self, key: str, loader: Abstract_DataImporter, new_batchPartition: BatchPartition):
        if loader.load_InBatches() > 0:
            print(f'-- Load {key} in batches --')
            self._load_inBatches(key, loader, new_batchPartition)

        else:
            datatyps = loader.get_dataTyps()
            data_shape = loader.get_shape()
            data_length = loader.get_shape()[0]

            data = loader.get_Data(slice(0, loader.get_shape()[0], 1))
            new_batchPartition.createDatabase(key, datatyps, data_shape, data=data)

    def _load_inBatches_parallel(self, key: str, loader: Abstract_DataImporter, out_dtypes, current_chunk: List[int]):
        data = loader.get_Data(current_chunk)
        return data

    def _load_inBatches(self, key: str, loader: Abstract_DataImporter, new_batchPartition: BatchPartition):
        datatyps = loader.get_dataTyps()
        data_shape = loader.get_shape()
        data_length = loader.get_shape()[0]

        batchTable = new_batchPartition.createDatabase(key, datatyps, data_shape)
        batch_size = loader.load_InBatches()
        parallel_executer = ParallelHelper(loader.get_nJobs(), True)
        current_pos = 0

        with tqdm(total=math.ceil(data_length / batch_size)) as pbar:
            while current_pos <= data_length:
                if batch_size > data_length or batch_size > data_length - current_pos:
                    end_pos = data_length
                else:
                    end_pos = current_pos + batch_size

                print(f' -- Load batch from {current_pos} to {end_pos} --')
                IQ_settings = IndexQueue_settings(start_index=current_pos, end_index=end_pos, chunksize=1024)
                dtype = loader.get_dataTyps().toNUMPY_dtypes()
                loaded_data = parallel_executer.execute_function_returnArrays(self._load_inBatches_parallel,
                                                                              IQ_settings,
                                                                              loader.get_dataTyps().toNUMPY_dtypes(), progressbar_args={'position':1,'leave':False},
                                                                              key=key, loader=loader)
                batchTable[slice(current_pos, current_pos + batch_size, 1)] = loaded_data

                current_pos += batch_size
                pbar.n = pbar.n + 1
                pbar.refresh()

        print(f'-- Dataloading completed --')
