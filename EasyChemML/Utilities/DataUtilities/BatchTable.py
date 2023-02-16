import h5py, math, numpy as np, pickle, base64

from EasyChemML.Utilities.DataUtilities.BatchDatatyp import BatchDatatyp, BatchDatatypClass
from EasyChemML.Utilities.DataUtilities.BatchDatatypHolder import BatchDatatypHolder
from numpy.lib import recfunctions as rfn
from typing import List, TYPE_CHECKING, Tuple

from EasyChemML.Utilities.ParallelUtilities.Shared_PythonList import Shared_PythonList

if TYPE_CHECKING:
    from .BatchPartition import BatchStructure
    from EasyChemML.Splitter.Splitcreator import Split


def _convertPickelObjects(arr: np.ndarray, columns: List[str], flat=False):
    if not flat:
        for i, row in enumerate(arr):
            for col in columns:
                arr[i][col] = base64.b64encode(pickle.dumps(arr[i][col], protocol=pickle.HIGHEST_PROTOCOL))
    elif len(columns) > 0:
        if len(arr.shape) > 1 and arr.shape[1] > 1:
            for i, row in enumerate(arr):
                for j in range(arr.shape[1]):
                    arr[i][j] = base64.b64encode(pickle.dumps(arr[i][j], protocol=pickle.HIGHEST_PROTOCOL))
        else:
            for i, row in enumerate(arr):
                arr[i] = base64.b64encode(pickle.dumps(arr[i], protocol=pickle.HIGHEST_PROTOCOL))


class BatchTable:
    _h5_dataset_key: str
    _chunksize: int
    _Bstruct: 'BatchStructure'

    def __init__(self, h5_dataset_key: str, chunksize: int, Bstruct: 'BatchStructure'):
        self._h5_dataset_key = h5_dataset_key
        self._chunksize = chunksize
        self._Bstruct = Bstruct

    def resize(self, shape: Tuple):
        var: h5py.Dataset = self._Bstruct.data_grp[self._h5_dataset_key]
        var.resize(shape)

    def delete(self):
        del self._Bstruct.partition[self._h5_dataset_key]

    def getBatchStructure(self) -> 'BatchStructure':
        return self._Bstruct

    def __setitem__(self, key, value):
        if not (isinstance(value, np.ndarray) or isinstance(value, np.void)):
            raise Exception(f'the passed batch is not a numpy array or numpy array entry| {type(value)}')

        if not self.getDatatypes().check_containsObjects() and not value.dtype == self[0].dtype:
            raise Exception(f'the dtype of the numpy array is different to the batchtable | {type(value)}')
        value: np.array = value

        my_datatypes: BatchDatatypHolder = self.getDatatypes()

        if BatchDatatypClass.PYTHON_OBJECT in my_datatypes:
            my_datatypes.removeAllnoneObject()
            # notObjects = [item for item in self.getDatatypes().getColumns() if item not in my_datatypes.getColumns()]

            object_cols = my_datatypes.getColumns()
            if isinstance(value, np.void):
                _convertPickelObjects([value], columns=object_cols, flat=my_datatypes.is_flat())
            else:
                _convertPickelObjects(value, columns=object_cols, flat=my_datatypes.is_flat())

        self._Bstruct.data_grp[self._h5_dataset_key][key] = value

    def __getitem__(self, item):
        my_datatypes: BatchDatatypHolder = self.getDatatypes()

        if BatchDatatypClass.PYTHON_OBJECT in my_datatypes or BatchDatatypClass.NUMPY_STRING in my_datatypes:
            my_datatypes.removeAllnoneObject()

            if isinstance(item, tuple):
                requested_data = self._Bstruct.data_grp[self._h5_dataset_key][item]
                for elem in item:
                    if elem in my_datatypes:
                        self._convertLISTBack2Object(requested_data[elem])

            elif isinstance(item, int) or isinstance(item, np.int64):
                # TODO check performance because it is scaling with chunksize
                requested_data = self._Bstruct.data_grp[self._h5_dataset_key][item]
                if my_datatypes.is_flat():
                    for i, _ in enumerate(requested_data):
                        requested_data[i] = pickle.loads(base64.b64decode(requested_data[i]))
                else:
                    object_cols = my_datatypes.getColumns()
                    for item in object_cols:
                        requested_data[item] = pickle.loads(base64.b64decode(requested_data[item]))

            elif isinstance(item, list) or isinstance(item, np.ndarray):
                item = np.sort(item)
                requested_data = self._Bstruct.data_grp[self._h5_dataset_key][item]
                if my_datatypes.is_flat():
                    self._convertLISTBack2Object(requested_data)
                else:
                    object_cols = my_datatypes.getColumns()
                    for item in object_cols:
                        self._convertLISTBack2Object(requested_data[item])

            elif isinstance(item, slice):
                requested_data = self._Bstruct.data_grp[self._h5_dataset_key][item]
                if my_datatypes.is_flat():
                    self._convertLISTBack2Object(requested_data)
                else:
                    object_cols = my_datatypes.getColumns()
                    for item in object_cols:
                        self._convertLISTBack2Object(requested_data[item])

            elif isinstance(item, str):
                requested_data = self._Bstruct.data_grp[self._h5_dataset_key][item]
                if item in my_datatypes:
                    self._convertLISTBack2Object(requested_data)
            else:
                raise Exception('item is not a valide selection')
        else:
            # item = np.sort(item)
            requested_data = self._Bstruct.data_grp[self._h5_dataset_key].__getitem__(item)
        return requested_data

    def getComplexSelection(self, columns: List[str] = None, indicies: List[int] = None):
        if columns is None:
            if indicies is None:
                return self[:]
            else:
                return self[indicies]
        else:
            if indicies is None:
                h5pyDataset: h5py.Dataset = self._Bstruct.data_grp[self._h5_dataset_key]
                return h5pyDataset.fields(names=columns)[:]
            else:
                h5pyDataset: h5py.Dataset = self._Bstruct.data_grp[self._h5_dataset_key]
                return h5pyDataset.fields(names=columns)[indicies]

    def shape(self):
        return self._Bstruct.data_grp[self._h5_dataset_key].shape

    def get_notConverted(self, slice_):
        requested_data = self._Bstruct.data_grp[self._h5_dataset_key].__getitem__(slice_)
        return requested_data

    def getColumns(self):
        my_datatypes: BatchDatatypHolder = self.getDatatypes()
        return my_datatypes.getColumns()

    def _convertLISTBack2Object(self, listof):
        for i, item in enumerate(listof):
            if isinstance(item, np.ndarray):
                for j, row in enumerate(item):
                    if len(item) > 0:
                        listof[i][j] = pickle.loads(base64.b64decode(row))
                    else:
                        listof[i][j] = None
            else:
                if len(item) > 0:
                    listof[i] = pickle.loads(base64.b64decode(item))
                else:
                    listof[i] = None

    def reorder(self, split: "Split", new_tableName) -> ("BatchTable", "Split"):
        return self._Bstruct.partition.createReorderedSubset(self, split, new_tableName)

    def to_SharedPythonList(self, indicies: List[int] = None, columns: List[str] = None) -> Shared_PythonList:
        if indicies is None:
            indicies = list(range(0, len(self)))

        datatyps = self.getDatatypes()
        if columns is None:
            data = self[indicies]
        else:
            data = self[indicies][columns]
            datatyps = datatyps.redefine(columns)

        shared_list = Shared_PythonList(data, datatyps)
        return shared_list

    def __iter__(self):
        return BatchAccess(self._Bstruct, self, self._h5_dataset_key, self._chunksize)

    def __len__(self):
        return len(self._Bstruct.data_grp[self._h5_dataset_key])

    def __repr__(self):
        if len(self) > 0:
            if len(self) > 2:
                return f'dtyp: {self.getDatatypes()} |0: {str(self._shortenArrayPrint(self[0]))} \n 1: {self._shortenArrayPrint(self[1])} \n ...'
            else:
                return f'dtyp: {self.getDatatypes()} |0: {str(self._shortenArrayPrint(self[0]))} \n ...'
        return f'BatchTable len <= 0 '

    def _shortenArrayPrint(self, arr: np.ndarray):
        if len(arr) > 1:
            return f'[{arr[0]} ; ... ; {arr[-1]}]'
        else:
            return f'[{arr[0]}]'

    def to_String(self, rows: int = None) -> str:
        if rows is None:
            if len(self) > 0:
                out_str = f'dtyp: {self.getDatatypes()} '

                for i in range(len(self)):
                    out_str += f'{i}: {str(self._longArrayPrint(self[i]))} \n'

                return out_str
            return f'BatchTable len <= 0 '
        else:
            if len(self) > 0:
                out_str = f'dtyp: {self.getDatatypes()} '

                rows += 1
                if rows > len(self):
                    rows = len(self)

                for i in range(rows):
                    out_str += f'{i}: {str(self._longArrayPrint(self[i]))} \n'

                return out_str
            return f'BatchTable len <= 0 '

    def _longArrayPrint(self, arr: np.ndarray):
        if len(arr) > 1:
            out_str = '['

            for i in range(len(arr)):
                if i == len(arr) - 1:
                    out_str += str(arr[i]) + ']'
                else:
                    out_str += str(arr[i]) + ' ; '

            return out_str
        else:
            return f'[{arr[0]}]'

    def getDatatypes(self) -> BatchDatatypHolder:
        db = self._Bstruct.data_grp[self._h5_dataset_key]
        att_value = db.attrs['dtyps']
        dtype = pickle.loads(base64.b64decode(att_value))
        return dtype

    def getChunksize(self):
        return self._chunksize

    def createShadowTable(self, indicies: List[int]) -> "BatchShadowTable":
        return BatchShadowTable(indicies, self)

    def getWith(self, columns: List[str] = None):
        count: int = 0
        if len(self) > 0:
            dtype_list = self.getDatatypes()

            for dtyp in dtype_list:
                if not columns is None and not dtyp in columns:
                    continue
                dtyp = dtype_list[dtyp]
                if dtyp.get_shape() is None:
                    count += 1
                elif len(dtyp.get_shape()) == 0:
                    count += 1
                elif len(dtyp.get_shape()) == 1:
                    count += dtyp.get_shape()[0]
                else:
                    raise Exception('find a dtyp with a 2D data shape. That is not support!')

            return count
        else:
            return 0

    def convert_2_ndarray(self, indicies: List[int] = None, columns: List[str] = None) -> np.ndarray:
        if columns is None:
            columns = self.getColumns()

        dataTypHolder: BatchDatatypHolder = self.getDatatypes().get_DtypSubset(columns)
        datatyp: BatchDatatyp

        # if dataTypHolder.check_containsObjects():
        #    raise Exception('convert to ndarray is not possible when objects/strings are inside the batchtable')

        if dataTypHolder.checkAll_same() and not dataTypHolder.checkAll_numbers():
            datatyp = dataTypHolder[dataTypHolder.getColumns()[0]]
            datatyp.set_shape(None)
        elif dataTypHolder.checkAll_numbers():
            #if not dataTypHolder.checkAll_same():
            #    raise Exception('different dTypes are not supported')
            highst_dtype_complexity = dataTypHolder.get_highest_number_complexity()
            datatyp = BatchDatatyp(BatchDatatypClass.get_by_lvl(highst_dtype_complexity))
            datatyp.set_shape(None)
        else:
            raise Exception('different dTypes are not supported')

        np_dtype = datatyp.toNUMPY()
        # if indicies is not None:
        #    shape = (len(indicies), self.getWith(columns))
        # else:
        #    shape = (len(self), self.getWith(columns))

        if indicies is None:
            if datatyp == BatchDatatypClass.NUMPY_STRING:
                raw_data = self.getComplexSelection(columns, indicies)
                data = rfn.structured_to_unstructured(raw_data, dtype=np_dtype)
                return data
            else:
                raw_data = self.getComplexSelection(columns, indicies)
                if len(columns) == 1:
                    data = raw_data[columns[0]]
                else:
                    data = rfn.structured_to_unstructured(raw_data)
                return data
        else:
            if datatyp == BatchDatatypClass.NUMPY_STRING:
                raw_data = self.getComplexSelection(columns, indicies)
                data = rfn.structured_to_unstructured(raw_data, dtype=np_dtype)
                return data
            else:
                raw_data: np.ndarray = self.getComplexSelection(columns, indicies)
                if len(columns) == 1:
                    data = raw_data[columns[0]]
                else:
                    data = rfn.structured_to_unstructured(raw_data)
                return data

    def get_index(self, index: int) -> np.ndarray:
        start = index * self._chunksize
        end = ((self._last_index + 1) * self._chunksize)

        if end >= len(self._BTable):
            end = len(self._BTable)

        data = self[start:end]

        return data

    def get_HDF5_DatasetName(self) -> str:
        return self._h5_dataset_key

    def overrideWithBatchtable(self, other_bt:'BatchTable'):
        otherKey = other_bt._h5_dataset_key
        del self._Bstruct.data_grp[self._h5_dataset_key]
        self._Bstruct.data_grp.move(otherKey, self._h5_dataset_key)


class BatchAccess:
    _last_index: int
    _Bstruct: 'BatchStructure'
    _BTable: BatchTable
    _chunksize: int
    _writeBackNeeded = False
    _h5_keyname: str

    def __init__(self, Bstruct: 'BatchStructure', BTable: BatchTable, h5_keyname, chunksize, _last_index=-1):
        self._last_index = _last_index
        self._Bstruct = Bstruct
        self._chunksize = chunksize
        self._writeBackNeeded = False
        self._h5_keyname = h5_keyname
        self._BTable = BTable

    def __iter__(self):
        return self

    def _getDB(self):
        return self._Bstruct.data_grp[self._h5_keyname]

    # @benchmark_function
    def __next__(self) -> h5py.Dataset:
        self._last_index += 1
        if self._last_index >= self.batchcount():
            if self._writeBackNeeded:
                self.writeBack()
            raise StopIteration

        start = self._last_index * self._chunksize
        end = ((self._last_index + 1) * self._chunksize)

        if end >= len(self._BTable):
            end = len(self._BTable)

        data = self._BTable[start:end]

        return data

    def batchcount(self) -> int:
        chunks = math.ceil(float(len(self._BTable)) / float(self._chunksize))
        return chunks

    def __len__(self):
        return self.batchcount()

    def last_index(self):
        return self._last_index

    # @benchmark_function
    def __lshift__(self, other):
        if self._writeBackNeeded:
            raise Exception('dtypes of the batch are change during the process :(')

        if not isinstance(other, np.ndarray):
            raise Exception('the passed batch is not a numpy array')
        other: np.array = other

        dtyp = self._BTable.getDatatypes()
        if not dtyp == BatchDatatypHolder().fromNUMPY_dtyp(other.dtype):
            # check for string object problematic
            if not dtyp.toNUMPY_dtypes() == other.dtype:
                raise Exception('the dtypes of the numpy arrays is not the same as in the batchtable')

        if self._last_index >= self.batchcount():
            raise Exception('index is equal or higher than batchcount')

        if self._writeBackNeeded:
            raise Exception('mixed some overrider methods <<= and <<')

        return self._fastoverride(other, self._last_index)

    def _fastoverride(self, data, index):
        start = index * self._chunksize
        end = ((index + 1) * self._chunksize)

        if end >= len(self._BTable):
            end = len(self._BTable)

        self._getDB()[start:end] = data
        return self

    # @benchmark_function
    def __ilshift__(self, other):
        if not isinstance(other, np.ndarray):
            raise Exception('the passed batch is not a numpy array')
        other: np.array = other

        if self._last_index >= self.batchcount():
            raise Exception('index is equal or higher than lenBatch')

        name = f'batchTMP_{hex(id(self))}'
        if name not in self._Bstruct.tmp_grp.keys():
            shape = len(self._BTable)
            dtype = other.dtype
            chunksize = self._chunksize

            if chunksize > len(self._BTable):
                chunksize = len(self._BTable)

            dataTypHolder = BatchDatatypHolder().fromNUMPY_dtyp(dtype)
            if len(dataTypHolder) == 1 and len(dataTypHolder[dataTypHolder.getColumns()[0]].get_shape()) == 0 and len(
                    other.shape) > 1:
                new_shape = list(other.shape)
                new_shape[0] = len(self._BTable)
                new_shape = tuple(new_shape)
                self._Bstruct.partition.createDatabase(name, dataTypHolder, new_shape, chunksize=chunksize, inTMP=True)
            else:
                self._Bstruct.partition.createDatabase(name, dataTypHolder, shape, chunksize=chunksize, inTMP=True)

            dataTypHolder.removeAllnoneObject()

            start = 0
            end = self._chunksize

            if end >= len(self._BTable):
                end = len(self._BTable)

            self._writeBackNeeded = True
            _convertPickelObjects(other, columns=dataTypHolder.getColumns(), flat=dataTypHolder.is_flat())
            self._Bstruct.tmp_grp[name][start:end] = other
        else:
            start = self._last_index * self._chunksize
            end = ((self._last_index + 1) * self._chunksize)

            if end >= len(self._BTable):
                end = len(self._BTable)

            if not self._writeBackNeeded:
                raise Exception('dtypes of the batch are change during the process :(')

            dtype = other.dtype

            dataTypHolder = BatchDatatypHolder().fromNUMPY_dtyp(dtype)
            dataTypHolder.removeAllnoneObject()
            _convertPickelObjects(other, columns=dataTypHolder.getColumns(), flat=dataTypHolder.is_flat())
            self._Bstruct.tmp_grp[name][start:end] = other

        return self

    def writeBack(self):
        name = f'batchTMP_{hex(id(self))}'
        new_db = self._Bstruct.tmp_grp[name]
        del self._Bstruct.data_grp[self._h5_keyname]
        self._Bstruct.data_grp.move(new_db.name, self._h5_keyname)


class BatchShadowTable:
    _indicies: List[int]
    _batchtable: BatchTable

    def __init__(self, indicies: List[int], batchtable: BatchTable):
        self._indicies = indicies
        self._batchtable = batchtable

    def __getitem__(self, item):
        return self._batchtable[self._indicies[item]]

    def shape(self):
        return [len(self._indicies)]

    def __len__(self):
        return len(self._indicies)
