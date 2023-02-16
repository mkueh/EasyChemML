from random import randrange

import numpy as np
import scipy as sc
import os

from six import assertCountEqual

from EasyChemML.Environment import Environment
from EasyChemML.Utilities.DataUtilities.BatchPartition import BatchPartition, BatchPartitionMode
from EasyChemML.Utilities.DataUtilities.BatchTableAlgorithms import BatchTableAlgorithms
import time


env = Environment(TMP_path='./TMP')


def test_sort_1D_ndarray_short():
    bp = BatchPartition(os.path.join(env.TMP_path, 'memdisk.disk'), 10000)
    bta = BatchTableAlgorithms()

    x = np.random.randint(10000, size=1000)
    bt = bp.addNumpyArray(x, 'test')

    bta.sort(bt)
    arr = bt[:]

    valid_arr = np.sort(x)

    for i, val in enumerate(arr):
        assert valid_arr[i] == val

def test_lexsort_2D_ndarray_BITlarge():
    bp = BatchPartition(os.path.join(env.TMP_path, 'memdisk.disk'), 10, mode=BatchPartitionMode.direct_BufferedIO)
    bta = BatchTableAlgorithms(env=env)

    print(f" ")

    array_length = 50
    bitcount = 2048
    bitdtype = []
    default_row = []
    for i in range(bitcount):
        bitdtype.append((f'{i}', 'i4'))
        default_row.append(0)
    default_row = tuple(default_row)

    x = np.array([default_row],
                 dtype=bitdtype)
    x = np.resize(x, (array_length,))

    for index, val in enumerate(x):
        newRow = []
        for i in range(bitcount):
            newRow.append(randrange(2))
        x[index] = tuple(newRow)

    bt = bp.addNumpyArray(x, 'test')

    def ret(x):
        return tuple(x)

    tic = time.perf_counter()
    x_sorted = sorted(x, key=ret)
    toc = time.perf_counter()

    print(f"Python in-build sorted in {time.perf_counter() - tic:0.4f} seconds")
    print(f"----------------------------------------------------")


    tic = time.perf_counter()
    MergeSort.sort(bt)
    toc = time.perf_counter()

    print(f"--------------------------------------------------")
    print(f"EasyChem lexSort sorted in {toc - tic:0.4f} seconds")

    arr = bt[:]

    for i, val in enumerate(arr):
        try:
            assert ret(x_sorted[i]) == ret(val)
        except Exception as e:
            print(f'asser thrown at {i}')
            e.with_traceback()

def test_lexsort_2D_ndarray_large():
    bp = BatchPartition(os.path.join(env.TMP_path, 'memdisk.disk'), 65)
    bta = BatchTableAlgorithms()

    x = np.array([(0, 0, 0)],
                 dtype=[('one', 'i4'), ('two', 'i4'), ('three', 'i4')])
    x = np.resize(x, (251,))

    for index, val in enumerate(x):
        x[index] = (randrange(10), randrange(1000), randrange(10))

    bt = bp.addNumpyArray(x, 'test')

    def ret(x):
        return (x[0],x[1],x[2])

    x_sorted = sorted(x, key=ret)

    bta.lexSort(bt)
    arr = bt[:]

    for i, val in enumerate(arr):
        assert ret(x_sorted[i]) == ret(val)

def test_sort_1D_ndarray_short_lambda():
    bp = BatchPartition(os.path.join(env.TMP_path, 'memdisk.disk'), 10000)
    bta = BatchTableAlgorithms()

    x = np.array([('Peter_0', 0, 0.0)],
                 dtype=[('name', 'U20'), ('age', 'i4'), ('weight', 'f4')])
    x = np.resize(x, (1000,))

    for index, val in enumerate(x):
        x[index] = (f'Peter_{index}', len(x) - index, float(index))

    bt = bp.addNumpyArray(x, 'test')

    def ret(x):
        return x[1]

    x_sorted = sorted(x, key=ret)

    bta.sort(bt, key_func=ret)
    arr = bt[:]

    for i, val in enumerate(arr):
        assert ret(x_sorted[i]) == ret(val)


def test_convert_2D_ndarray_short():
    bp = BatchPartition(os.path.join(env.TMP_path, 'memdisk.disk'), 10000)
    bta = BatchTableAlgorithms()

    x = np.array([('Peter_0', 0, 0.0)],
                 dtype=[('name', 'U20'), ('age', 'i4'), ('weight', 'f4')])
    x = np.resize(x, (1000,))

    for index, val in enumerate(x):
        x[index] = (f'Peter_{index}', index, float(index))

    bt = bp.addNumpyArray(x, 'test')

    bta.sort(bt)
    arr = bt[:]

    valid_arr = np.sort(x)

    for i, val in enumerate(arr):
        val[0] = val[0].decode("utf-8")
        assert valid_arr[i][0] == val[0]
        assert valid_arr[i][1] == val[1]
        assert valid_arr[i][2] == val[2]
