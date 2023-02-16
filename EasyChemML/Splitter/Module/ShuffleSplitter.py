import math
from typing import Union

import numpy as np

from EasyChemML.Splitter.Module.Abstract_Splitter import Abstract_Splitter
from EasyChemML.Splitter.Utilities.Subsetmaker import Subsetmaker
from EasyChemML.Utilities.DataUtilities.BatchTable import BatchTable
from EasyChemML.Utilities.ParallelUtilities.Shared_PythonList import Shared_PythonList


# todo not working

class ShuffleSplitter(Abstract_Splitter):
    n_splits: int
    random_state: int
    test_size: float
    train_size: float

    def __init__(self, n_splits: int, random_state: int = 42, test_size: float = None, train_size: float = None):
        self.n_splits = n_splits
        self.random_state = random_state
        self.test_size = test_size
        self.train_size = train_size
        np.random.seed(self.random_state)

    def split(self, datatable: Union[Shared_PythonList, BatchTable]):
        return self._split(datatable.shape()[0])

    def _split(self, datasetSize:int):
        if self.train_size is not None and self.test_size is not None:
            raise Exception(
                'train_size and test_size are not None, it is not possible that both parameter are not None')

        if self.train_size is None:
            self.train_size = 1.0 - self.test_size
        elif self.test_size is None:
            self.test_size = 1.0 - self.train_size
        else:
            raise Exception('train_size and test_size are None')

        splits = []
        for n in range(self.n_splits):
            absolute_testsize = math.ceil(datasetSize * self.test_size)
            test_index = np.random.choice(datasetSize, absolute_testsize, replace=False)
            test_index = np.sort(test_index)

            train_index = Subsetmaker.generateInverseSubsetWithRange(test_index, datasetSize)

            # test_index.sort()
            splits.append((train_index, test_index))

        return splits

    def contains_random_state(self):
        return True
