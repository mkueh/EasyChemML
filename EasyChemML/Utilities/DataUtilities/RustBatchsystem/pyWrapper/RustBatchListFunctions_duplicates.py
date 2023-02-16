import os.path, json
from typing import Dict, List, Tuple, Any

from EasyChemML.Utilities.DataUtilities.BatchDatatyp import BatchDatatypClass

from EasyChemML.Utilities.DataUtilities.RustBatchsystem.pyRustBatchsystem import BatchListFunctions_duplicates_py
from EasyChemML.Utilities.DataUtilities.RustBatchsystem.pyWrapper.RustBatchholder import RustBatchholder


class RustBatchListFunctions_duplicates_result:
    counted_entries: int
    counted_duplicates: int
    entry_most_duplicates: int
    duplicates_dist: Dict[int, int]
    duplicates_by_last_col: Dict[Any, int]

    def __init__(self , counted_entries, counted_duplicates, entry_most_duplicates, duplicates_dist,
                 duplicates_by_last_col):
        self.counted_entries = counted_entries
        self.counted_duplicates = counted_duplicates
        self.entry_most_duplicates = entry_most_duplicates
        self.duplicates_dist = duplicates_dist
        self.duplicates_by_last_col = duplicates_by_last_col

    def __str__(self):
        string = ''

        string += 'counted_entries: ' + str(self.counted_entries) + '\n'
        string += 'counted_duplicates: ' + str(self.counted_duplicates) + '\n'
        string += 'entry_most_duplicates: ' + str(self.entry_most_duplicates) + '\n'
        string += 'duplicates_dist: ' + str(self.duplicates_dist) + '\n'
        string += 'duplicates_by_last_col: ' + str(self.duplicates_by_last_col) + '\n'

        return string

    def save_to_json(self, path:str):
        out_dict = {}
        out_dict['counted_entries'] = self.counted_entries
        out_dict['counted_duplicates'] = self.counted_duplicates
        out_dict['entry_most_duplicates'] = self.entry_most_duplicates
        out_dict['duplicates_dist'] = self.duplicates_dist
        out_dict['duplicates_by_last_col'] = self.duplicates_by_last_col

        with open(os.path.join(path+'.json'), 'w') as fp:
            json.dump(out_dict, fp)

class RustBatchListFunctions_duplicates:
    batchlist_duplicates: BatchListFunctions_duplicates_py = None

    def __init__(self):
        self._batchlist_duplicates = BatchListFunctions_duplicates_py()

    def count_duplicates(self, rustbatchholder: RustBatchholder, tableName: str,
                         get_distibution_by_last_col: bool = False) -> RustBatchListFunctions_duplicates_result:
        bt = rustbatchholder.getRustBatchTable(tableName)
        dtype = rustbatchholder.rustBatchtable[tableName][1]
        dtype = BatchDatatypClass.get_by_lvl(dtype.get_highest_number_complexity())

        if dtype == dtype.NUMPY_INT8:
            result = self._batchlist_duplicates.count_duplicates_on_sorted_list_i8(bt, get_distibution_by_last_col)
            return RustBatchListFunctions_duplicates_result(result.counted_entries,result.counted_duplicates, result.entry_most_duplicates, result.duplicates_dist, result.duplicates_by_last_col)
        elif dtype == dtype.NUMPY_INT16:
            result = self._batchlist_duplicates.count_duplicates_on_sorted_list_i16(bt, get_distibution_by_last_col)
            return RustBatchListFunctions_duplicates_result(result.counted_entries, result.counted_duplicates,
                                                            result.entry_most_duplicates, result.duplicates_dist,
                                                            result.duplicates_by_last_col)
        elif dtype == dtype.NUMPY_INT32:
            result = self._batchlist_duplicates.count_duplicates_on_sorted_list_i32(bt, get_distibution_by_last_col)
            return RustBatchListFunctions_duplicates_result(result.counted_entries, result.counted_duplicates,
                                                            result.entry_most_duplicates, result.duplicates_dist,
                                                            result.duplicates_by_last_col)
        elif dtype == dtype.NUMPY_INT64:
            result = self._batchlist_duplicates.count_duplicates_on_sorted_list_i64(bt, get_distibution_by_last_col)
            return RustBatchListFunctions_duplicates_result(result.counted_entries, result.counted_duplicates,
                                                            result.entry_most_duplicates, result.duplicates_dist,
                                                            result.duplicates_by_last_col)
        elif dtype == dtype.NUMPY_FLOAT32:
            raise Exception('float32 is not sortable at the moment')
        elif dtype == dtype.NUMPY_FLOAT64:
            raise Exception('float64 is not sortable at the moment')
        else:
            raise Exception('datatype is not sortable at the moment')
