import numpy as np, joblib, uuid, os, shutil, copy
from .BatchDatatyp import BatchDatatyp, BatchDatatypClass
from typing import List, Dict, Tuple, Optional, Union


class BatchDatatypHolder:
    _BDList: Dict[str, BatchDatatyp]
    _flat: bool = False

    def __init__(self, BDList: Dict[str, BatchDatatyp] = None, flat: bool = False):
        if BDList is None:
            BDList = {}
        self._BDList = BDList
        self._flat = flat

    def __repr__(self):
        return str(self._BDList)

    def __lshift__(self, other):
        if not isinstance(other, tuple):
            raise Exception('argument needs to be a tuple')

        if isinstance(other[1], BatchDatatyp) and isinstance(other[0], str):
            self._BDList[other[0]] = other[1]
        else:
            raise Exception('right side is not a instance of BatchDatatyp')

    def __getitem__(self, item) -> Union[BatchDatatyp, Dict[str, BatchDatatyp]]:
        if self._flat:
            return self._BDList['None']

        if isinstance(item, list):
            out = {}
            for elem in item:
                out[elem] = (self._BDList[elem])
            return out
        else:
            if self._flat:
                return self._BDList['None']
            else:
                return self._BDList[item]

    def get_DtypSubset(self, columns: List[str]) -> 'BatchDatatypHolder':
        subset = self[columns]
        return BatchDatatypHolder(copy.deepcopy(subset))

    def is_flat(self):
        return self._flat

    def __setitem__(self, key, value):
        """
        It is possible to set an BatchDatatypClass oder BatchDatatyp
        the BatchDatatypClass would automatic translate in a BatchDatatyp
        """
        if isinstance(value, BatchDatatypClass):
            self._BDList[key] = BatchDatatyp(value)
        elif isinstance(value, BatchDatatyp):
            self._BDList[key] = value
        else:
            raise Exception('It is not possible to set an item which is not a BatchDatatyp or BatchDatatypClass')

    def __contains__(self, item):
        if isinstance(item, str):
            return self._BDList.__contains__(item)
        elif isinstance(item, BatchDatatypClass):
            for key in self._BDList:
                if self._BDList[key] == item:
                    return True
            return False
        else:
            raise Exception('item is not a str oder BatchDatatypClass')

    def __len__(self):
        return len(self._BDList)

    def __eq__(self, other):
        if isinstance(other, BatchDatatypHolder):
            for key in self._BDList:
                if key in other:
                    if not other[key] == self[key]:
                        return False
                else:
                    return False
            if not len(self._BDList) == len(other):
                return False
        else:
            return False
        return True

    def __iter__(self):
        return self._BDList.__iter__()

    def __delitem__(self, key):
        del self._BDList[key]

    def redefine(self, columns: List[str]) -> 'BatchDatatypHolder':
        new_bdlist = {}

        for col_names in columns:
            new_bdlist[col_names] = self[col_names]

        return BatchDatatypHolder(new_bdlist, self._flat)

    def copy(self):
        return copy.deepcopy(self)

    def checkAll_same(self) -> bool:
        if len(self) > 1:
            first = self[self.getColumns()[0]]

            for item in self:
                item: BatchDatatyp = self[item]
                if not first.get_DatatypClass() == item.get_DatatypClass():
                    return False
            return True
        else:
            return True

    def checkAll_numbers(self) -> bool:
        if len(self) > 1:
            first = self[self.getColumns()[0]]

            for item in self:
                item: BatchDatatyp = self[item]
                if BatchDatatypClass.get_dtype_lvl(item) < 0:
                    return False
            return True
        else:
            return True

    def check_containsObjects(self) -> bool:
        if len(self) > 1:
            first = self[self.getColumns()[0]]

            for item in self:
                item: BatchDatatyp = self[item]
                if BatchDatatypClass.get_dtype_lvl(item) < 0:
                    return True
            return False
        else:
            return False

    def get_highest_number_complexity(self) -> int:
        highest_dtype_lvl = -3
        for key in self._BDList:
            if BatchDatatypClass.get_dtype_lvl(self._BDList[key]) > highest_dtype_lvl:
                highest_dtype_lvl = BatchDatatypClass.get_dtype_lvl(self._BDList[key])
        return highest_dtype_lvl

    def getColumns(self):
        return list(self._BDList.keys())

    def removeAllnoneObject(self, in_place: bool = True) -> Optional['BatchDatatypHolder']:
        if in_place:
            removeKeys = []
            for key in self._BDList:
                if self._BDList[key] == BatchDatatypClass.PYTHON_OBJECT:
                    continue
                else:
                    removeKeys.append(key)
            for key in removeKeys:
                del self._BDList[key]
        else:
            self_copy = copy.deepcopy(self)
            self_copy.removeAllnoneObject()
            return self_copy

    def createAEmptyNumpyArray(self, size: int, TMP_FOLDER: str = None):
        """
        create a memmap numpy array
        """

        if TMP_FOLDER is not None:
            print('BatchDatatypHolder:createAEmptyNumpyArray -> TMP_FOLDER is not longer needed')

        dtypes = self.toNUMPY_dtypes()
        return np.empty(shape=(size,), dtype=dtypes)

    def fromNUMPY_dtyp(self, dtyp: np.dtype):
        self._BDList = {}
        if dtyp.fields is None:
            if len(dtyp.shape) == 0:
                """
                If the dtyp is not a structured array dtyp, the dtyp has no fields.
                to fit into the BatchDatatypHolder construction, the dtyp is saved under the key 'None'
                
                if some finds a better solution, pls xD
                """
                self._BDList['None'] = BatchDatatyp.Fabricator_BY_str(str(dtyp.base), tuple())
                self._flat = True
        else:
            for name, typ in dtyp.fields.items():
                typ: np.dtype = typ[0]
                if typ.metadata is not None and 'vlen' in typ.metadata and typ.metadata['vlen'] == str:
                    self._BDList[name] = BatchDatatyp.Fabricator_BY_str('str')
                elif dtyp.metadata is not None and 'str' in dtyp.metadata and name in dtyp.metadata['str']:
                    self._BDList[name] = BatchDatatyp.Fabricator_BY_str('str')
                else:
                    self._BDList[name] = BatchDatatyp.Fabricator_BY_str(str(typ.base), typ.shape)
        return self

    def toH5PY_dtypes(self):
        if self._flat:
            return np.dtype(self._BDList['None'].toH5())
        else:
            out_list = []
            for item in list(self._BDList.keys()):
                t_item = (item, self[item].toH5())
                out_list.append(t_item)
            return np.dtype(out_list)

    def toNUMPY_dtypes(self, string_as_obj=True, flatMe=False):
        if self._flat:
            new_dtype = np.dtype(self._BDList['None'].toNUMPY(string_as_obj))
        else:
            if flatMe:
                out_list = ''
                str_list = []
                for item in list(self._BDList.keys()):
                    if self._BDList[item] == BatchDatatypClass.NUMPY_STRING:
                        str_list.append(item)
                    numpy_dtyp = self[item].toNUMPY(string_as_obj)
                    str_numpy_dtyp = str(numpy_dtyp)
                    out_list = out_list + f'{str_numpy_dtyp},'
                out_list = out_list[:-1]
                new_dtype = np.dtype(out_list, metadata={'str': str_list})
            else:
                out_list = []
                str_list = []
                for item in list(self._BDList.keys()):
                    if self._BDList[item] == BatchDatatypClass.NUMPY_STRING:
                        str_list.append(item)
                    t_item = (item, self[item].toNUMPY(string_as_obj))
                    out_list.append(t_item)

                new_dtype = np.dtype(out_list, metadata={'str': str_list})
        return new_dtype
