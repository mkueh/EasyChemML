from EasyChemML.Environment import Environment
from EasyChemML.Splitter.Splitcreator import Splitset
from EasyChemML.Utilities.DataUtilities.BatchTable import BatchTable

from typing import List

from EasyChemML.Utilities.SharedDataset import SharedDataset


class Dataset():
    _feature_data: BatchTable
    _feature_col: List[str]
    _feature_col_encode: List[str]

    _target_data: BatchTable
    _target_col: List[str]
    _target_col_encode: List[str]

    _WORK_FOLDER: str
    _TMP_FOLDER: str

    _split: Splitset
    _name: str

    def __init__(self, data: BatchTable, feature_col: List[str], name: str, target_data: BatchTable = None,
                 feature_col_encode: List[str] = None, target_col: List[str] = None,
                 target_col_encode: List[str] = None,
                 split: Splitset = None, env:Environment = None):

        self._feature_data = data
        self._feature_col = feature_col
        self._feature_col_encode = feature_col_encode

        self._target_data = target_data
        self._target_col = target_col
        self._target_col_encode = target_col_encode

        self._splits = split

        self._WORK_FOLDER = env.WORKING_path
        self._TMP_FOLDER = env.TMP_path
        self.name = name

        self._fill()

    def _fill(self):
        if self._target_data is None:
            self._target_data = self._feature_data
        if self._feature_col_encode is None:
            self._feature_col_encode = self._feature_col
        if self._target_col is None:
            all_cols = self._feature_data.getColumns()
            rest_cols = list(set(all_cols) - set(self._feature_col))
            self._target_col = rest_cols
        if self._target_col_encode is None:
            self._target_col_encode = self._target_col

    def to_SharedDataset(self, only_shareIndicies: List[int] = None, only_DefinedColumns: bool = True) -> SharedDataset:
        if only_shareIndicies is None:
            indicies = list(range(len(self.get_FeatureData())))
            return SharedDataset(self, indicies, only_DefinedColumns=only_DefinedColumns)
        else:
            return SharedDataset(self, only_shareIndicies, only_DefinedColumns)

    def get_Splitset(self) -> Splitset:
        return self._splits

    def get_FeatureData(self) -> BatchTable:
        return self._feature_data

    def get_FeatureData_Col(self) -> List[str]:
        return self._feature_col

    def get_FeatureData_Col_Encode(self) -> List[str]:
        return self._feature_col_encode

    def get_TargetData(self) -> BatchTable:
        return self._target_data

    def get_TargetData_Col(self) -> List[str]:
        return self._target_col

    def get_TargetData_Col_Encode(self):
        return self._target_col_encode

    def get_WORK_FOLDER(self) -> str:
        return self._WORK_FOLDER

    def get_TMP_FOLDER(self) -> str:
        return self._TMP_FOLDER

    def get_name(self) -> str:
        return self.name

    def set_FeatureData(self, new_data: BatchTable):
        self._feature_data = new_data

    def set_TargetData(self, new_data: BatchTable):
        self._target_data = new_data
