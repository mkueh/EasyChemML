from typing import Tuple, Union, List
from abc import ABC, abstractmethod

from EasyChemML.Utilities.DataUtilities.BatchDatatypHolder import BatchDatatypHolder


class Abstract_DataImporter(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def get_shape(self) -> Tuple[int, int]:
        pass

    @abstractmethod
    def get_Data(self, selection: Union[slice, List[int]] = None):
        """
        Expected is a list-like object which overrides def __getitem__(self, item)

        Example: List, PandasDataframe ...

        If load_InBatches() is False this object is called with [0:len_Data()]
        """
        pass

    @abstractmethod
    def get_dataTyps(self) -> BatchDatatypHolder:
        pass

    @abstractmethod
    def get_Split(self):
        pass

    @abstractmethod
    def contains_Split(self):
        pass

    @abstractmethod
    def contains_Data(self):
        pass

    @abstractmethod
    def load_InBatches(self):
        raise Exception('Not implemented in abstract class')

    @abstractmethod
    def get_nJobs(self) -> int:
        """
        Returns: if load_InBatches > 0 this function returns how many parallel jobs should be used for the Dataimport
        """
        raise Exception('Not implemented in abstract class')
