from EasyChemML.Splitter.Splitcreator import Splitset
from EasyChemML.Utilities.ParallelUtilities.Shared_PythonList import Shared_PythonList

from typing import List


class SharedDataset():
    _feature_data: Shared_PythonList
    _feature_col: List[str]
    _feature_col_encode: List[str]

    _target_data: Shared_PythonList
    _target_col: List[str]
    _target_col_encode: List[str]

    _WORK_FOLDER: str
    _TMP_FOLDER: str

    _split: Splitset
    only_shareIndicies = True
    name: str

    def __init__(self, dataset, only_shareIndicies: List[int] = None, only_DefinedColumns: bool = True):
        if only_DefinedColumns:
            self._feature_data = dataset.get_FeatureData().to_SharedPythonList(only_shareIndicies,
                                                                               dataset.get_FeatureData_Col())
        else:
            self._feature_data = dataset.get_FeatureData().to_SharedPythonList(only_shareIndicies)
        self._feature_col = dataset.get_FeatureData_Col()
        self._feature_col_encode = dataset.get_FeatureData_Col_Encode()

        if only_DefinedColumns:
            self._target_data = dataset.get_TargetData().to_SharedPythonList(only_shareIndicies,
                                                                             dataset.get_TargetData_Col())
        else:
            self._target_data = dataset.get_TargetData().to_SharedPythonList(only_shareIndicies)
        self._target_col = dataset.get_TargetData_Col()
        self._target_col_encode = dataset.get_TargetData_Col_Encode()

        self._splits = dataset.get_Splitset()

        self._WORK_FOLDER = dataset.get_WORK_FOLDER()
        self._TMP_FOLDER = dataset.get_TMP_FOLDER()
        self.name = dataset.get_name()

        if only_shareIndicies is None:
            return False

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

    def destroy(self):
        if self._target_data == self._feature_data:
            self._feature_data.destroy()
        else:
            self._feature_data.destroy()
            self._target_data.destroy()

            self._feature_data = None
            self._target_data = None

    def get_Split(self) -> Splitset:
        return self._splits

    def get_FeatureData(self) -> Shared_PythonList:
        return self._feature_data

    def get_FeatureData_Col(self) -> List[str]:
        return self._feature_col

    def get_FeatureData_Col_Encode(self) -> List[str]:
        return self._feature_col_encode

    def get_TargetData(self) -> Shared_PythonList:
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

    def set_FeatureData(self, new_data: Shared_PythonList):
        self._feature_data.destroy()
        self._feature_data = new_data

    def set_TargetData(self, new_data: Shared_PythonList):
        self._target_data.destroy()
        self._target_data = new_data
