from EasyChemML.Splitter.Module.Abstract_Splitter import Abstract_Splitter
from EasyChemML.Utilities.DataUtilities.BatchTable import BatchTable

from typing import List, Dict, Iterator
import numpy as np


class Splitsettings(object):
    param: Dict
    algo: str

    def __init__(self, param: Dict, algo: str):
        self.param = param
        self.algo = algo


class Split(object):
    train = None
    test = None

    def __init__(self, train: List, test: List):
        self.train = train
        self.test = test
        if not self.checkIfIncreasedOrder():
            raise Exception('The indicies of a split must be specified in ascending order.')

    def checkIfIncreasedOrder(self):
        return np.all(np.diff(self.train) > 0) and np.all(np.diff(self.test) > 0)

    def getTrainIterator(self, batchSize: int, skip: int = 0) -> Iterator[List[int]]:
        if batchSize <= 0:
            raise Exception('The batch_size needs to be bigger than zero')

        start = 0 + (batchSize*skip)
        end = start + batchSize

        while True:
            if start > len(self.train):
                yield []
                break
            elif end >= len(self.train):
                yield self.train[start:]
                break
            else:
                yield self.train[start:end]
                start += batchSize
                end = start + batchSize

    def getTestIterator(self, batchSize: int, skip: int = 0) -> Iterator[List[int]]:
        if batchSize <= 0:
            raise Exception('The batch_size needs to be bigger than zero')

        start = 0 + (batchSize*skip)
        end = start + batchSize

        while True:
            if start > len(self.test):
                yield []
                break
            elif end >= len(self.test):
                yield self.test[start:]  
                break
            else:
                yield self.test[start:end]
                start += batchSize
                end = start + batchSize


class Splitset(object):
    __out_splits: List[Split] = None
    __in_splits: List[Split] = None

    __outer_splitter: Abstract_Splitter = None
    __inner_splitter: Abstract_Splitter = None

    def __init__(self, out_split: List[Split], in_splits: List[Split], out_split_settings: Abstract_Splitter = None,
                 in_split_settings: Abstract_Splitter = None):
        self.__out_splits = out_split
        self.__in_splits = in_splits

        self._out_splits_settings = out_split_settings
        self._in_splits_settings = in_split_settings

    def get_outer_split(self, index_outer):
        return self.__out_splits[index_outer]

    def get_outer_splitts(self):
        return self.__out_splits

    def get_inner_split(self, index_outer, index_inner):
        return self.__in_splits[index_outer][index_inner]

    def get_inner_splitts(self, index_outer):
        return self.__in_splits[index_outer]

    def get_out_splits_settings(self):
        return self._out_splits_settings

    def get_in_splits_settings(self):
        return self._in_splits_settings

    def get_inner_split_absolut(self, index_outer, index_inner):
        """
        Since the inner distribution is calculated and stored relative to the outer one, this function helps to calculate absolute indices for the nested cv
        """
        train = [self.__out_splits[index_outer].train[i] for i in self.__in_splits[index_outer][index_inner].train]
        test = [self.__out_splits[index_outer].train[i] for i in self.__in_splits[index_outer][index_inner].test]
        return Split(train, test)

    def get_outerCount(self):
        return len(self.__out_splits)

    def get_innerCount(self, outer_index):
        return len(self.__in_splits[outer_index])


class Splitcreator(object):

    def __init__(self):
        pass

    """Generate the splitting indices"""

    def generate_split(self, dataTable: BatchTable, outer_split: Abstract_Splitter,
                       inner_split: Abstract_Splitter = None) -> Splitset:

        out_splits = self._warp2Split(outer_split.split(dataTable))
        in_splits = []

        for i, _ in enumerate(out_splits):
            if inner_split is not None:
                in_splits.append(
                    self._warp2Split(inner_split.split(self._getarrindices(dataTable, out_splits[i].train))))

        out = Splitset(out_splits, in_splits, outer_split, inner_split)
        return out

    def _warp2Split(self, splits: List[List]):
        tmp = []

        for split in splits:
            tmp.append(Split(split[0], split[1]))

        return tmp

    def _getarrindices(self, arr, indicies):
        return arr.createShadowTable(indicies=indicies)
