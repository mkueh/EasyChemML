from abc import ABC, abstractmethod
from typing import Union

from EasyChemML.Utilities.DataUtilities.BatchTable import BatchTable
from EasyChemML.Utilities.ParallelUtilities.Shared_PythonList import Shared_PythonList


class Abstract_Splitter(ABC):

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def split(self, datatable: Union[Shared_PythonList, BatchTable]):
        raise Exception('not implemented')

    @abstractmethod
    def contains_random_state(self):
        pass
