import numpy as np
from timeit import default_timer as timer
from EasyChemML.Splitter.Module.ShuffleSplitter import ShuffleSplitter

def test_shuffleSplitt_500000():
    count= 3333
    splitter = ShuffleSplitter(1, 42, 0.3)

    start = timer()
    ret = splitter._split(count)
    end = timer()
    print('---------------')
    print(f'time in {end - start}s')

    test = ret[0][0]
    train = ret[0][1]

    ret = np.concatenate([test, train])
    sort = np.sort(ret)

    for i in range(count):
        assert sort[i] == i

def test_shuffleSplitt_3():
    count= 3
    splitter = ShuffleSplitter(1, 42, 0.3)

    start = timer()
    ret = splitter._split(count)
    end = timer()
    print('---------------')
    print(f'time in {end - start}s')

    test = ret[0][0]
    train = ret[0][1]

    ret = np.concatenate([test, train])
    sort = np.sort(ret)

    for i in range(count):
        assert sort[i] == i

def test_shuffleSplitt_3333():
    count= 3333
    splitter = ShuffleSplitter(1, 42, 0.3)

    start = timer()
    ret = splitter._split(count)
    end = timer()
    print('---------------')
    print(f'time in {end - start}s')

    test = ret[0][0]
    train = ret[0][1]

    ret = np.concatenate([test, train])
    sort = np.sort(ret)

    for i in range(count):
        assert sort[i] == i

def test_shuffleSplitt_1mio():
    count= 1 * 1000 * 1000
    splitter = ShuffleSplitter(1, 42, 0.3)

    start = timer()
    ret = splitter._split(count)
    end = timer()
    print('---------------')
    print(f'time in {end - start}s')

    test = ret[0][0]
    train = ret[0][1]

    ret = np.concatenate([test, train])
    sort = np.sort(ret)

    for i in range(count):
        assert sort[i] == i
