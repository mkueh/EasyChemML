from EasyChemML.Splitter.Module.Abstract_Splitter import Abstract_Splitter
from EasyChemML.Utilities.DataUtilities.BatchTable import BatchTable
from EasyChemML.Utilities.ParallelUtilities.Shared_PythonList import Shared_PythonList

from typing import Union

class AllTrainSplitter(Abstract_Splitter):

    def __init__(self):
        pass

    def split(self, datatable: Union[Shared_PythonList, BatchTable]):
        out = list()

        length = len(datatable)
        out.append(((list(range(length))), []))
        return out

    def contains_random_state(self):
        return False
