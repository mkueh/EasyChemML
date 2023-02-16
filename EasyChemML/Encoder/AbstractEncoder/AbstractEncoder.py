from typing import List
from abc import ABC, abstractmethod

from EasyChemML.Utilities.DataUtilities.BatchTable import BatchTable


class AbstractEncoder(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def convert(self, datatable:BatchTable, columns:List[str], n_jobs: int, **kwargs):
        pass #set dataset and return databuffer

    @staticmethod
    @abstractmethod
    def is_parallel():
        return False
