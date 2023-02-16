from typing import Union

from EasyChemML.Splitter.Module.Abstract_Splitter import Abstract_Splitter
from EasyChemML.Utilities.DataUtilities.BatchTable import BatchTable
from EasyChemML.Utilities.ParallelUtilities.Shared_PythonList import Shared_PythonList


class RangeSplitter(Abstract_Splitter):

    _testset_start: int
    _testset_end: int

    def __init__(self, testset_start:int, testset_end:int):
        self._testset_start = testset_start
        self._testset_end = testset_end

    def split(self, datatable: Union[Shared_PythonList, BatchTable]):
        out = list()

        testset = list(range(self._testset_start, self._testset_end))
        trainset = []

        for i in range(0, len(datatable)):
            if i < self._testset_start:
                trainset.append(i)
            elif i >= self._testset_end:
                trainset.append(i)
        out.append((trainset, testset))
        return out

    def contains_random_state(self):
        return False