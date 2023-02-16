from enum import Enum
from typing import Tuple, List, Dict, Union

import numpy as np, h5py


class BatchDatatypClass(Enum):
    NUMPY_STRING = 'STRING'
    PYTHON_OBJECT = 'OBJECT'
    NUMPY_INT8 = 'INT8'
    NUMPY_INT16 = 'INT16'
    NUMPY_INT32 = 'INT32'
    NUMPY_INT64 = 'INT64'
    NUMPY_FLOAT16 = 'FLOAT16'
    NUMPY_FLOAT32 = 'FLOAT32'
    NUMPY_FLOAT64 = 'FLOAT64'
    NUMPY_COMPLEX64 = 'COMPLEX64'
    NUMPY_COMPLEX128 = 'COMPLEX128'

    @staticmethod
    def get_dtype_lvl(dtype_class: 'BatchDatatypClass'):
        if dtype_class == BatchDatatypClass.NUMPY_STRING:
            return -1
        elif dtype_class == BatchDatatypClass.PYTHON_OBJECT:
            return -2
        elif dtype_class == BatchDatatypClass.NUMPY_INT8:
            return 1
        elif dtype_class == BatchDatatypClass.NUMPY_INT16:
            return 2
        elif dtype_class == BatchDatatypClass.NUMPY_FLOAT16:
            return 3
        elif dtype_class == BatchDatatypClass.NUMPY_INT32:
            return 4
        elif dtype_class == BatchDatatypClass.NUMPY_FLOAT32:
            return 5
        elif dtype_class == BatchDatatypClass.NUMPY_INT64:
            return 6
        elif dtype_class == BatchDatatypClass.NUMPY_FLOAT64:
            return 7
        elif dtype_class == BatchDatatypClass.NUMPY_COMPLEX64:
            return 8
        elif dtype_class == BatchDatatypClass.NUMPY_COMPLEX128:
            return 9

    @staticmethod
    def get_by_lvl(lvl: int):
        if lvl == -1:
            return BatchDatatypClass.NUMPY_STRING
        elif lvl == -2:
            return BatchDatatypClass.PYTHON_OBJECT
        elif lvl == 1:
            return BatchDatatypClass.NUMPY_INT8
        elif lvl == 2:
            return BatchDatatypClass.NUMPY_INT16
        elif lvl == 3:
            return BatchDatatypClass.NUMPY_FLOAT16
        elif lvl == 4:
            return BatchDatatypClass.NUMPY_INT32
        elif lvl == 5:
            return BatchDatatypClass.NUMPY_FLOAT32
        elif lvl == 6:
            return BatchDatatypClass.NUMPY_INT64
        elif lvl == 7:
            return BatchDatatypClass.NUMPY_FLOAT64
        elif lvl == 8:
            return BatchDatatypClass.NUMPY_COMPLEX64
        elif lvl == 9:
            return BatchDatatypClass.NUMPY_COMPLEX128

    def to_numpy(self) -> np.dtype:
        return np.dtype(str(self.value).lower())


class BatchDatatyp:
    _DatatypClass: BatchDatatypClass
    _shape: tuple

    def __init__(self, typclass: BatchDatatypClass, shape: Tuple = None):
        self._DatatypClass = typclass
        self._shape = shape

        if not isinstance(shape, tuple) and not shape is None:
            raise Exception('shape is not a tuple')

        if shape is None:
            self._shape = ()

    def __repr__(self):
        return f'(class: {self._DatatypClass} | shape: {self._shape})'

    def __eq__(self, other):
        if isinstance(other, BatchDatatyp):
            if other.get_DatatypClass() == self._DatatypClass:
                return True
            else:
                return False
        elif isinstance(other, BatchDatatypClass):
            if other == self._DatatypClass:
                return True
            return False
        else:
            return False
        return False

    def get_DatatypClass(self):
        return self._DatatypClass

    def get_shape(self):
        return self._shape

    def set_shape(self, shape: Tuple):
        self._shape = shape

    def toNUMPY(self, string_as_obj=True):
        out_string = ''

        if self._shape is None or len(self._shape) < 1 or self._shape[0] == 1:
            out_string = ''
        else:
            out_string = str(self._shape)

        if self._DatatypClass == BatchDatatypClass.NUMPY_STRING:
            if not (self._shape is None or len(self._shape) == 0 or self._shape[0] > 0):
                raise Exception('String column cannot be an Array')
            if string_as_obj:
                out_string = out_string + 'O'
            else:
                out_string = out_string + 'U'
        elif self._DatatypClass == BatchDatatypClass.PYTHON_OBJECT:
            if not (self._shape is None or len(self._shape) == 0 or self._shape[0] > 0):
                raise Exception('Object column cannot be an Array')
            out_string = out_string + 'O'
        elif self._DatatypClass == BatchDatatypClass.NUMPY_INT8:
            out_string = out_string + 'int8'
        elif self._DatatypClass == BatchDatatypClass.NUMPY_INT16:
            out_string = out_string + 'int16'
        elif self._DatatypClass == BatchDatatypClass.NUMPY_INT32:
            out_string = out_string + 'int32'
        elif self._DatatypClass == BatchDatatypClass.NUMPY_INT64:
            out_string = out_string + 'int64'
        elif self._DatatypClass == BatchDatatypClass.NUMPY_FLOAT16:
            out_string = out_string + 'float16'
        elif self._DatatypClass == BatchDatatypClass.NUMPY_FLOAT32:
            out_string = out_string + 'float32'
        elif self._DatatypClass == BatchDatatypClass.NUMPY_FLOAT64:
            out_string = out_string + 'float64'
        elif self._DatatypClass == BatchDatatypClass.NUMPY_COMPLEX64:
            out_string = out_string + 'complex64'
        elif self._DatatypClass == BatchDatatypClass.NUMPY_COMPLEX128:
            out_string = out_string + 'complex128'
        return out_string

    def toH5(self):
        out_string = ''

        if self._shape is None or len(self._shape) == 0:
            out_string = ''
        else:
            out_string = str(self._shape)

        if self._DatatypClass == BatchDatatypClass.NUMPY_STRING:
            if not (self._shape is None or len(self._shape) == 0 or self._shape[0] == 1):
                raise Exception('String column cannot be an Array')
            return h5py.string_dtype(encoding='utf-8')
        elif self._DatatypClass == BatchDatatypClass.PYTHON_OBJECT:
            if not (self._shape is None or len(self._shape) == 0 or self._shape[0] == 1):
                raise Exception('Object column cannot be an Array')
            return h5py.string_dtype(encoding='utf-8')
        elif self._DatatypClass == BatchDatatypClass.NUMPY_INT8:
            out_string = out_string + 'int8'
        elif self._DatatypClass == BatchDatatypClass.NUMPY_INT16:
            out_string = out_string + 'int16'
        elif self._DatatypClass == BatchDatatypClass.NUMPY_INT32:
            out_string = out_string + 'int32'
        elif self._DatatypClass == BatchDatatypClass.NUMPY_INT64:
            out_string = out_string + 'int64'
        elif self._DatatypClass == BatchDatatypClass.NUMPY_FLOAT16:
            out_string = out_string + 'float16'
        elif self._DatatypClass == BatchDatatypClass.NUMPY_FLOAT32:
            out_string = out_string + 'float32'
        elif self._DatatypClass == BatchDatatypClass.NUMPY_FLOAT64:
            out_string = out_string + 'float64'
        elif self._DatatypClass == BatchDatatypClass.NUMPY_COMPLEX64:
            out_string = out_string + 'complex64'
        elif self._DatatypClass == BatchDatatypClass.NUMPY_COMPLEX128:
            out_string = out_string + 'complex128'

        return out_string

    @staticmethod
    def Fabricator_BY_str(typAsString: str, shape: tuple = None):
        typAsString = typAsString.lower().strip()

        if 'u' in typAsString or typAsString == 'str':
            if not (shape is None or len(shape) == 0):
                raise Exception('String column cannot be an Array')
            return BatchDatatyp(BatchDatatypClass.NUMPY_STRING, shape)
        elif typAsString == 'o' or typAsString == 'object':
            if not (shape is None or len(shape) == 0):
                raise Exception('Object column cannot be an Array')
            return BatchDatatyp(BatchDatatypClass.PYTHON_OBJECT, shape)
        elif typAsString == 'int8':
            return BatchDatatyp(BatchDatatypClass.NUMPY_INT8, shape)
        elif typAsString == 'int16':
            return BatchDatatyp(BatchDatatypClass.NUMPY_INT16, shape)
        elif typAsString == 'int32' or typAsString == 'int' or typAsString == 'integer':
            return BatchDatatyp(BatchDatatypClass.NUMPY_INT32, shape)
        elif typAsString == 'int64':
            return BatchDatatyp(BatchDatatypClass.NUMPY_INT64, shape)
        elif typAsString == 'float16':
            return BatchDatatyp(BatchDatatypClass.NUMPY_FLOAT16, shape)
        elif typAsString == 'float32' or typAsString == 'float' or typAsString == 'flt':
            return BatchDatatyp(BatchDatatypClass.NUMPY_FLOAT32, shape)
        elif typAsString == 'float64':
            return BatchDatatyp(BatchDatatypClass.NUMPY_FLOAT64, shape)
        elif typAsString == 'complex' or typAsString == 'complex64':
            return BatchDatatyp(BatchDatatypClass.NUMPY_COMPLEX64, shape)
        elif typAsString == 'complex128':
            return BatchDatatyp(BatchDatatypClass.NUMPY_COMPLEX128, shape)
        else:
            raise Exception(f'cannot detect the Basedatatyp of {typAsString}')

    @staticmethod
    def dtype_with_highst_order(dtypes: List[Union[str, BatchDatatypClass]]) -> int:
        highest_dtype_lvl = -3
        for dtype in dtypes:
            if BatchDatatypClass.get_dtype_lvl(dtype) > highest_dtype_lvl:
                highest_dtype_lvl = BatchDatatypClass.get_dtype_lvl(dtype)
        return BatchDatatypClass.get_by_lvl(highest_dtype_lvl)

    @staticmethod
    def Fabricator_BY_Classes(typAsClass: BatchDatatypClass):
        return BatchDatatyp(typAsClass)
