import math, time
import numpy as np
from typing import Tuple, Dict, List

from EasyChemML.Utilities.DataUtilities.BatchDatatypHolder import BatchDatatypHolder
from EasyChemML.Utilities.DataUtilities.BatchTable import BatchTable, BatchAccess


class MergeSort:

    @staticmethod
    def sort(batchTable: BatchTable):
        length = len(batchTable)
        chunkSize = batchTable.getChunksize()

        if chunkSize > length:
            sortedarray = MergeSort._sort(batchTable[:], batchTable.getDatatypes())
            batchTable[:] = sortedarray
        else:
            MergeSort._chunkSort(batchTable)

    @staticmethod
    def _generateDiffDict(arr: np.ndarray, col_index) -> Tuple[Dict, List[str]]:
        difference = {}
        for row_index, val in enumerate(arr):
            val = val[col_index]
            if val in difference:
                difference[val] += 1
            else:
                difference[val] = 1

        keys = list(difference.keys())
        keys.sort()  # gibt die Sortierung vor!

        return difference, keys

    @staticmethod
    def _generateStartPointer(difference: Dict, keys: List[str]) -> Dict:
        current_pointer = {}
        posTmp = 0
        for val in keys:
            current_pointer[val] = posTmp
            posTmp += difference[val]
        return current_pointer

    @staticmethod
    def _swap(a, b, copyIndices=None):
        if copyIndices is None:
            return b, a
        else:
            b[copyIndices[0]:copyIndices[1]] = a[copyIndices[0]:copyIndices[1]]
            return b, a

    @staticmethod
    def _sort(arr: np.ndarray, dataType: BatchDatatypHolder):
        extra_space = dataType.createAEmptyNumpyArray(size=len(arr))
        amount_of_columns = len(dataType.getColumns())

        current_input = arr
        current_out = extra_space

        for col_index in reversed(range(amount_of_columns)):
            difference, keys = MergeSort._generateDiffDict(current_input, col_index)
            current_pointer = MergeSort._generateStartPointer(difference, keys)

            for row_index, val in enumerate(current_input):
                new_position = current_pointer[val[col_index]]
                current_pointer[val[col_index]] += 1
                current_out[new_position] = val
            current_input, current_out = MergeSort._swap(current_input, current_out)

        return current_input

    @staticmethod
    def _compareRow(a, b) -> int:
        for col_index in range(len(a)):
            if a[col_index] > b[col_index]:
                return -1
            elif a[col_index] < b[col_index]:
                return 1
        return 0

    @staticmethod
    def _chunkSort(bt: BatchTable):
        tic = time.perf_counter()
        iterator: BatchAccess = iter(bt)
        batch: np.ndarray
        dataTypHolder: BatchDatatypHolder = bt.getDatatypes()

        for batch in iterator:
            sorted = MergeSort._sort(batch, dataTypHolder)
            iterator << sorted

        # merge sort
        chunkCount = iterator.batchcount()
        iterationDepth = math.ceil(math.log2(chunkCount))  # tree depth
        stpoints = []
        baseBatchPartition = bt.getBatchStructure().partition

        extraSpace = baseBatchPartition.createDatabase('tmp_radixSortExtra', bt.getDatatypes(), len(bt),
                                                       bt.getChunksize())

        for chunk in range(chunkCount):
            if (chunk + 1) * bt.getChunksize() > len(bt):
                stpoints.append((chunk * bt.getChunksize(), len(bt)))
            else:
                stpoints.append((chunk * bt.getChunksize(), ((chunk + 1) * bt.getChunksize())))

        inSpace = bt
        outSpace = extraSpace

        print(f"LexSort partsorting in {time.perf_counter() - tic:0.4f} seconds")
        tic = time.perf_counter()
        for depth in range(iterationDepth):
            remove_stpoints_indices = []
            for chunk in range(0, chunkCount - 1, 2):
                MergeSort.merge(inSpace, stpoints[chunk], stpoints[chunk + 1], outSpace)
                remove_stpoints_indices.append(chunk + 1)
                stpoints[chunk] = (stpoints[chunk][0], stpoints[chunk + 1][1])

            if len(stpoints) % 2 == 1:
                inSpace, outSpace = MergeSort._swap(inSpace, outSpace, stpoints[-1])
            else:
                inSpace, outSpace = MergeSort._swap(inSpace, outSpace)

            stpoints = [i for j, i in enumerate(stpoints) if j not in remove_stpoints_indices]
            chunkCount = len(stpoints)

        print(f"LexSort merging in {time.perf_counter() - tic:0.4f} seconds")

        if not inSpace == bt:
            bt.overrideWithBatchtable(extraSpace)
        else:
            baseBatchPartition.deleteDatabase('tmp_radixSortExtra')

    @staticmethod
    def merge(inputBatchtable: BatchTable, a: Tuple[int, int], b: Tuple[int, int], outputBatchtable: BatchTable):
        tmp_space = outputBatchtable.getDatatypes().createAEmptyNumpyArray(inputBatchtable.getChunksize())

        currentPos_tmp = 0
        currentPos_a = 0
        currentPos_b = 0
        currentPos_out = a[0]
        len_a = a[1] - a[0]
        len_b = b[1] - b[0]
        totalLength = a[1] - a[0] + b[1] - b[0]

        for i in range(totalLength):
            if currentPos_tmp >= len(tmp_space):
                outputBatchtable[currentPos_out:currentPos_out + len(tmp_space)] = tmp_space
                currentPos_out += len(tmp_space)
                currentPos_tmp = 0

            if currentPos_a >= len_a:
                tmp_space[currentPos_tmp] = inputBatchtable[b[0] + currentPos_b]
                currentPos_b += 1
                currentPos_tmp += 1
                continue

            if currentPos_b >= len_b:
                tmp_space[currentPos_tmp] = inputBatchtable[a[0] + currentPos_a]
                currentPos_a += 1
                currentPos_tmp += 1
                continue

            compare = MergeSort._compareRow(inputBatchtable[a[0] + currentPos_a],
                                            inputBatchtable[b[0] + currentPos_b])

            if compare >= 0:
                tmp_space[currentPos_tmp] = inputBatchtable[a[0] + currentPos_a]
                currentPos_tmp += 1
                currentPos_a += 1
            else:
                tmp_space[currentPos_tmp] = inputBatchtable[b[0] + currentPos_b]
                currentPos_tmp += 1
                currentPos_b += 1

        outputBatchtable[currentPos_out:currentPos_out + currentPos_tmp] = tmp_space[0:currentPos_tmp]
        pass
