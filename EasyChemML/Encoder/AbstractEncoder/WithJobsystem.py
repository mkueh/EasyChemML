from abc import ABC, abstractmethod
from typing import List

from EasyChemML.Utilities.DataUtilities.BatchTable import BatchTable


class WithJobsystem(ABC):

    @abstractmethod
    def convert_Async2Job(self, datatable:BatchTable, columns:List[str], n_jobs: int, **kwargs):
        pass