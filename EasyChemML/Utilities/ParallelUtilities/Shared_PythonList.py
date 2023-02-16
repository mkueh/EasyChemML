from EasyChemML.Utilities.DataUtilities.BatchDatatypHolder import BatchDatatypHolder
from EasyChemML.Utilities.DataUtilities.BatchDatatyp import BatchDatatypClass

import numpy as np, base64, pickle, copy
from multiprocessing.shared_memory import SharedMemory


class Shared_PythonList():

    _shared_mem_data:np.ndarray = -1
    _shared_mem_manager = -1

    _pickeled = False

    _shared_mem_name = -1
    _shared_mem_dtype = -1
    _shared_mem_shape = -1

    _object_columns = None
    _batchdataTypHolder: BatchDatatypHolder

    def __init__(self, arr:np.ndarray, dataTyps:BatchDatatypHolder):
        datatyps_without_o = dataTyps.removeAllnoneObject(False)
        self._batchdataTypHolder = dataTyps
        self._object_columns = []

        if len(datatyps_without_o) > 0:
            self._create_ObjectShared(arr, dataTyps)
        else:
            self._create_directShared(arr, dataTyps.toNUMPY_dtypes())

    def getcolumns(self):
        return self._shared_mem_data.dtype.names

    def get_dtype(self) -> np.dtype:
        return self._shared_mem_dtype

    def shape(self):
        return self._shared_mem_data.shape

    def get_BatchDatatypHolder(self) -> BatchDatatypHolder:
        return self._batchdataTypHolder

    def __len__(self):
        return len(self._shared_mem_data)

    def __getitem__(self, item):
            if len(self._object_columns) == 0:
                return self._shared_mem_data.__getitem__(item)

            requested_data = copy.copy(self._shared_mem_data.__getitem__(item))

            if isinstance(item, tuple):
                for elem in item:
                    if elem in self._object_columns:
                        self._convertLISTBack2Object(requested_data[elem])

            elif isinstance(item, int):
                object_cols = self._object_columns
                for item in object_cols:
                    requested_data[item] = pickle.loads(base64.b64decode(requested_data[item]))

            elif isinstance(item, list) or isinstance(item, np.ndarray):
                object_cols = self._object_columns
                for item in object_cols:
                    self._convertLISTBack2Object(requested_data[item])

            elif isinstance(item, slice):
                object_cols = self._object_columns
                for item in object_cols:
                    self._convertLISTBack2Object(requested_data[item])

            elif isinstance(item, str):
                if item in self._object_columns:
                    self._convertLISTBack2Object(requested_data)

            return requested_data

    def __setitem__(self, key, value):
        #TODO not implemented Yet
        pass

    def __getstate__(self):
        state = {}
        state['_shared_mem_name'] = self._shared_mem_name
        state['_shared_mem_shape'] = self._shared_mem_shape
        state['_shared_mem_dtype'] = self._shared_mem_dtype
        state['_object_columns'] = self._object_columns
        state['_batchdataTypHolder'] = self._batchdataTypHolder
        return state

    def __setstate__(self, newstate):
        self.__dict__.update(newstate)
        self._pickeled = True
        self._shared_mem_manager = SharedMemory(self._shared_mem_name)
        self._shared_mem_data = np.ndarray(shape=self._shared_mem_shape, dtype=self._shared_mem_dtype, buffer=self._shared_mem_manager.buf)

    def _convertLISTBack2Object(self, listof):
        for i, item in enumerate(listof):
            if item != '':
                listof[i] = pickle.loads(base64.b64decode(item))
            else:
                listof[i] = None

    def close(self):
        self._shared_mem_manager.close()

    def unlink(self):
        self._shared_mem_manager.unlink()

    def destroy(self):
        self._shared_mem_manager.close()
        self._shared_mem_manager.unlink()

    def _create_directShared(self, arr:np.ndarray, dtype:np.dtype):
        self._shared_mem_manager = SharedMemory(create=True, size=arr.nbytes)
        shm_batch = np.ndarray(shape=arr.shape, dtype=dtype, buffer=self._shared_mem_manager.buf)
        np.copyto(shm_batch, arr)
        self._shared_mem_name, self._shared_mem_shape, self._shared_mem_dtype = self._shared_mem_manager.name, arr.shape, dtype
        self._shared_mem_data = shm_batch

    def _create_ObjectShared(self, arr:np.ndarray, dataTyps:BatchDatatypHolder):
        new_np_array = dataTyps.createAEmptyNumpyArray(len(arr))

        for key in dataTyps.getColumns():
            if dataTyps[key] == BatchDatatypClass.PYTHON_OBJECT:
                self._object_columns.append(key)
                self._convert_pickelObjects(arr, new_np_array, key)
            else:
                new_np_array[key][:] = arr[key][:]

        self._create_directShared(new_np_array, new_np_array.dtype)

    def _convert_pickelObjects(self, in_arr: np.ndarray, out_arr: np.ndarray, col:str):
        for i, row in enumerate(in_arr):
            out_arr[i][col] = base64.b64encode(pickle.dumps(in_arr[i][col], protocol=5))
