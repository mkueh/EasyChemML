from EasyChemML.Utilities.DataUtilities.BatchTable import BatchTable
from EasyChemML.Utilities.DataUtilities.BatchDatatyp import BatchDatatyp, BatchDatatypClass
from EasyChemML.Utilities.DataUtilities.BatchDatatypHolder import BatchDatatypHolder
from EasyChemML.Utilities.DataUtilities.BatchTable import BatchAccess
from EasyChemML.Splitter.Splitcreator import Split
from typing import Dict
import h5py, pandas, os, numpy as np, pickle, base64
import shutil
from tqdm import tqdm
from enum import Enum
import hdf5plugin


class BatchPartitionMode(Enum):
    direct_BufferedIO = None
    direct_UnBufferedIO = 'stdio'
    in_memory = 'core'


class BatchStructure:
    data_grp: h5py.Group
    tmp_grp: h5py.Group
    partition: 'BatchPartition'
    root_file: h5py.File
    readOnly: bool

    def __init__(self, data_grp: h5py.Group, tmp_grp: h5py.Group, partition: 'BatchPartition', root_file: h5py.File,
                 readOnly: bool):
        self.data_grp = data_grp
        self.tmp_grp = tmp_grp
        self.partition = partition
        self.root_file = root_file
        self.readOnly = readOnly

    def __getstate__(self):
        state = {}
        state['H5_path'] = self.partition.getHDF5Path()
        state['default_chunk_size'] = self.partition.getDefaultChunksize()
        state['tmp_grp'] = str(self.tmp_grp.name)[1:]
        state['data_grp'] = str(self.data_grp.name)[1:]
        state['readOnly'] = self.readOnly
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.partition = BatchPartition(state['H5_path'], state['default_chunk_size'], load_existing=True,
                                        read_only=self.readOnly)
        self.tmp_grp = self.partition.get_struc().tmp_grp
        self.data_grp = self.partition.get_struc().data_grp


class BatchPartition(object):
    _default_chunk_size: int

    _struc: BatchStructure
    _h5Path: str
    _Databuffer: Dict[str, BatchTable]

    def __init__(self, h5_path: str, default_chunk_size: int = 20000, load_existing: bool = False, tmp_grp: str = 'tmp',
                 data_grp: str = 'data', read_only: bool = False,
                 mode: BatchPartitionMode = BatchPartitionMode.direct_BufferedIO,
                 rdcc_nslots=7919, rdcc_nbytes=10  * 1024 ^ 2):
        """
        The batch partition includes all data relevant for EasyChemML in the appropriate format. Using the mode parameter,
        different operating modes can be selected. direct_BufferedIO or direct_UnBufferedIO ensure that all data is buffered
        on the hard disk. Thus, large amounts of data can be stored and processed. With the setting in_memory all data are
        processed only in the working memory, if the memory is to small the program will crash.


        Parameters
        ----------
        h5_path
        default_chunk_size
        load_existing
        tmp_grp
        data_grp
        read_only
        mode
        rdcc_nslots
        rdcc_nbytes
        """

        self._h5Path = h5_path
        if not load_existing and not read_only:
            if os.path.exists(h5_path) and os.path.isdir(h5_path):
                shutil.rmtree(h5_path)
            elif os.path.exists(h5_path) and os.path.isfile(h5_path):
                os.remove(h5_path)
            else:
                pass

        if read_only:
            root_grp = h5py.File(h5_path, 'r', driver=mode.value, libver='latest', rdcc_nslots=rdcc_nslots,
                                 rdcc_nbytes=rdcc_nbytes)
        else:
            root_grp = h5py.File(h5_path, 'w', driver=mode.value, libver='latest', rdcc_nslots=rdcc_nslots,
                                 rdcc_nbytes=rdcc_nbytes)

        tmp_grp_o = root_grp.require_group(tmp_grp)
        data_grp_o = root_grp.require_group(data_grp)

        self._struc = BatchStructure(data_grp_o, tmp_grp_o, self, root_grp, read_only)
        self._default_chunk_size = default_chunk_size
        self._Databuffer = {}

        if load_existing:
            databases = data_grp_o.keys()
            for key in databases:
                raw_table: h5py.Dataset = data_grp_o[key]
                self._Databuffer[key] = BatchTable(key, raw_table.chunks[0], self._struc)

    def __contains__(self, item):
        if item in self._Databuffer:
            return True
        else:
            return False

    def __getitem__(self, item: str):
        return self._Databuffer[item]

    def get_struc(self):
        return self._struc

    def copy_dataset(self, src_Batchtable: BatchTable, destination_BP: 'BatchPartition', rename: str = None):
        src_Batchtable_name = src_Batchtable.get_HDF5_DatasetName()
        if rename is None:
            self._struc.data_grp.copy(source=self._struc.data_grp[src_Batchtable_name],
                                      dest=destination_BP._struc.data_grp)
        else:
            self._struc.data_grp.copy(source=self._struc.data_grp[src_Batchtable_name],
                                      dest=destination_BP._struc.data_grp,
                                      name=rename)

    def getHDF5Path(self):
        return self._h5Path

    def getDefaultChunksize(self):
        return self._default_chunk_size

    def createDatabase(self, key: str, batchDtyps: BatchDatatypHolder, shape, chunksize: int = -1, data=None,
                       inTMP=False, without_compressor: bool = False):
        if chunksize == -1:
            chunksize = self._default_chunk_size

        # check where
        if not inTMP:
            grp = self._struc.data_grp
            if key in grp:
                raise Exception('tablename is already exist')

        else:
            grp = self._struc.tmp_grp

            if key in grp:
                raise Exception('tablename is already exist')

        if isinstance(shape, tuple) or isinstance(shape, list):
            size = shape[0]
        else:
            size = shape

        # check size and chunksize settings
        if size <= chunksize:
            chunksize = size

        h5_dtypes = batchDtyps.toH5PY_dtypes()
        if not without_compressor:
            lz4_compressor = hdf5plugin.LZ4(nbytes=0)
        else:
            lz4_compressor = {}

        if data is None:
            if isinstance(shape, tuple) and len(shape) > 1:
                grp.create_dataset(key, dtype=h5_dtypes, shape=shape, chunks=(chunksize, shape[1]), **lz4_compressor)
            else:
                grp.create_dataset(key, dtype=h5_dtypes, shape=(size,), chunks=(chunksize,), **lz4_compressor)
        else:
            data = data.astype(batchDtyps.toNUMPY_dtypes())
            if isinstance(shape, tuple) and len(shape) > 1:
                grp.create_dataset(key, data=data, dtype=h5_dtypes, shape=shape, chunks=(chunksize, shape[1]),
                                   **lz4_compressor)
            else:
                grp.create_dataset(key, data=data, dtype=h5_dtypes, shape=(size,), chunks=(chunksize,),
                                   **lz4_compressor)

        grp[key].attrs['dtyps'] = base64.b64encode(pickle.dumps(obj=batchDtyps))

        if not inTMP:
            self._Databuffer[key] = BatchTable(key, chunksize, self._struc)
            return self._Databuffer[key]

        return grp[key]

    def addPandaDataFrame(self, PD: pandas.DataFrame, asKey: str) -> BatchTable:
        batchDtyps: BatchDatatypHolder = BatchDatatypHolder()

        for key in PD.keys():
            col_typ = PD[key].dtype
            col_len = len(PD[key])
            chunksize = self._default_chunk_size

            if str(col_typ) == 'object':
                batchDtyps << (key, BatchDatatyp(BatchDatatypClass.NUMPY_STRING))
            else:
                batchDtyps << (key, BatchDatatyp.Fabricator_BY_str(str(col_typ)))

            if self._default_chunk_size > col_len:
                chunksize = col_len + 1

        dtypes = batchDtyps.toNUMPY_dtypes()
        out = np.empty(shape=col_len, dtype=dtypes)

        for key in PD.keys():
            out[key] = PD[key].values

        self.createDatabase(asKey, batchDtyps, col_len, chunksize, out, True)

    def addNumpyArray(self, numArray: np.ndarray, asKey: str) -> BatchTable:
        """
        Convert a structured or non-structured numpy-array to BatchTable.

        Parameters
        ----------
        numArray
        asKey

        Returns
        -------

        """
        datatype = BatchDatatypHolder()
        datatype.fromNUMPY_dtyp(numArray.dtype)
        return self.createDatabase(asKey, datatype, numArray.shape, data=numArray)

    def createReorderedSubset(self, old_table: BatchTable, split: Split, new_tableName) -> (BatchTable, Split):
        batchDtyps: BatchDatatypHolder = old_table.getDatatypes()
        chunksize = old_table.getChunksize()

        self.createDatabase(new_tableName, batchDtyps, len(old_table), chunksize)

        new_table: BatchTable = self[new_tableName]
        start_train = 0
        end_train = len(split.train)
        start_test = len(split.train)
        end_test = len(split.train) + len(split.test)

        iterator: BatchAccess = iter(new_table)
        batch: np.ndarray

        count_train = len(split.train)
        current_pos_in_train = 0
        count_test = len(split.test)
        current_pos_in_test = 0

        with tqdm(total=len(iterator), desc='create a reordered table: ') as pbar:
            for i, batch in enumerate(iterator):
                if count_train > 0:
                    if len(batch) > count_train:
                        batch[0:count_train] = old_table.get_notConverted(
                            split.train[current_pos_in_train: current_pos_in_train + count_train])

                        # fill batch with test_items
                        check_value = len(batch) - count_train
                        batch[count_train:len(batch)] = old_table.get_notConverted(
                            split.test[0:len(batch) - count_train])
                        current_pos_in_test += len(batch) - count_train

                        count_test -= len(batch) - count_train
                        count_train = 0
                    else:
                        batch = old_table.get_notConverted(
                            split.train[current_pos_in_train: current_pos_in_train + len(batch)])
                        count_train -= len(batch)
                        current_pos_in_train += len(batch)
                else:
                    if len(batch) > count_test:
                        batch[0:count_test] = old_table.get_notConverted(
                            split.test[current_pos_in_test: current_pos_in_test + count_test])
                        count_test = 0
                    else:
                        batch = old_table.get_notConverted(
                            split.test[current_pos_in_test: current_pos_in_test + len(batch)])
                        count_test -= len(batch)
                        current_pos_in_test += len(batch)

                iterator << batch
                pbar.update(1)

        return self[new_tableName], Split(train=np.arange(start=start_train, stop=end_train),
                                          test=np.arange(start=start_test, stop=end_test))

    def cloneBatchTable(self, src_key: str, new_key: str):
        if not new_key in self._Databuffer:
            self._struc.data_grp.copy(src_key, new_key, name=new_key)
            self._Databuffer[new_key] = BatchTable(new_key, self._Databuffer[src_key].getChunksize(), self._struc)
        else:
            raise Exception('A batchtable with this key is already in the database')

    def keys(self):
        return self._Databuffer.keys()

    def getSoftLinkedlist(self, key: str):
        raise Exception('Not implemented yet')

    def flush(self):
        self._struc.root_file.flush()

    def deleteDatabase(self, key:str):
        del self._struc.data_grp[key]

    def close(self):
        self._struc.root_file.close()
        self._struc = None
        self._h5Path = None
        self._Databuffer = None
