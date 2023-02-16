from EasyChemML.Utilities.DataUtilities.BatchDatatyp import BatchDatatypClass, BatchDatatyp

import numpy as np, base64, pickle
from multiprocessing.shared_memory import SharedMemory

class Shared_Value():
    """
    Shared_Value without atomic operations
    -> read is not race condition safe

    #Todo add atomic methods
    """

    _shared_mem_data = -1
    _shared_mem_manager = -1

    _pickeled = False

    _shared_mem_name = -1
    _shared_mem_dtype = -1
    _shared_mem_shape = -1

    _object_typ = None

    def __init__(self, val, dataTyp:BatchDatatypClass, extra_space_inbytes:int=0):
        typ = BatchDatatyp(dataTyp)
        self._object_typ = dataTyp

        if dataTyp == BatchDatatypClass.PYTHON_OBJECT:
            self._create_ObjectShared(val, extra_space_inbytes)
        else:
            arr = np.empty((1,), typ.toNUMPY(), )
            arr[0] = val
            self._create_directShared(arr, extra_space_inbytes)

    def get_dtype(self) -> np.dtype:
        return self._object_typ

    def get_val(self):
        if not self._object_typ == BatchDatatypClass.PYTHON_OBJECT:
            return self._shared_mem_data[0]
        else:
            return pickle.loads(base64.b64decode(self._shared_mem_data[0]))

    def set_val(self, new_value):
        if not self._object_typ == BatchDatatypClass.PYTHON_OBJECT:
            self._shared_mem_data[0] = new_value
        else:
            raise Exception('not implemented yet')

    def __getstate__(self):
        state = {}
        state['_shared_mem_name'] = self._shared_mem_name
        state['_shared_mem_shape'] = self._shared_mem_shape
        state['_shared_mem_dtype'] = self._shared_mem_dtype
        state['_object_typ'] = self._object_typ
        return state

    def __setstate__(self, newstate):
        self.__dict__.update(newstate)
        self._pickeled = True
        self._shared_mem_manager = SharedMemory(self._shared_mem_name)
        self._shared_mem_data = np.ndarray(shape=self._shared_mem_shape, dtype=self._shared_mem_dtype, buffer=self._shared_mem_manager.buf)

    def __del__(self):
        if self._pickeled:
            self._shared_mem_manager.close()

    def destroy(self):
        self._shared_mem_manager.close()
        self._shared_mem_manager.unlink()

    def _create_directShared(self, arr:np.ndarray, extra_space_inbytes):
        self._shared_mem_manager = SharedMemory(create=True, size=arr.nbytes+extra_space_inbytes)
        shm_batch = np.ndarray(arr.shape, dtype=arr.dtype, buffer=self._shared_mem_manager.buf)
        np.copyto(shm_batch, arr)
        self._shared_mem_name, self._shared_mem_shape, self._shared_mem_dtype = self._shared_mem_manager.name, arr.shape, arr.dtype
        self._shared_mem_data = shm_batch

    def _create_ObjectShared(self, value, extra_space_inbytes):
        typ = BatchDatatyp(BatchDatatypClass.PYTHON_OBJECT)
        arr = np.empty((1,),typ.toNUMPY(),)
        arr[0] = self._convert_pickelObject(value)

        self._create_directShared(arr, extra_space_inbytes)

    def _convert_pickelObject(self, value):
        return base64.b64encode(pickle.dumps(value, protocol=5))