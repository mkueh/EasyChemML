import numpy as np
from typing import Callable, Any

from EasyChemML.Environment import Environment
from EasyChemML.Utilities.DataUtilities.BatchTable import BatchTable
from EasyChemML.Utilities.DataUtilities.TableAlgorithms.Sort.MergeSort import MergeSort
from EasyChemML.Utilities.DataUtilities.TableAlgorithms.ReorderInplace import ReorderInplace


class BatchTableAlgorithms:

    env: Environment

    def __init__(self, env:Environment):
        self.env = env

    def mergeSort(self, batchTable: BatchTable):
        MergeSort.sort(batchTable, env=self.env)

    def sort(self, batchTable: BatchTable, key_func: Callable[[Any, Any], int] = None):
        length = len(batchTable)
        chunksize = batchTable.getChunksize()

        if length > chunksize:
            self._chunkSort(batchTable, key_func)
        else:
            self._sort(batchTable, key_func)

    def _sort(self, batchTable: BatchTable, key_func: Callable[[Any, Any], int] = None):
        arr = batchTable[:]
        if key_func is None:
            arr.sort()
        else:
            sortAble = np.vectorize(key_func)(arr[:])
            sortedIndices = np.argsort(sortAble)
            ReorderInplace.reorder_inplace(arr, sortedIndices)

        batchTable[:] = arr

    def _chunkSort(self, batchTable: BatchTable, compare_func: Callable[[Any, Any], int] = None):
        raise Exception('not implemented yet')

    def _chunkMerge(self, batchTable: BatchTable):
        raise Exception('not implemented yet')
