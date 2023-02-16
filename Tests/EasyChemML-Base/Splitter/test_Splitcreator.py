from EasyChemML.Splitter.Splitcreator import Split


def test_split_getTrainIterator_Batchsize128():
    test_split = Split(train=list(range(1024)), test=list(range(1024, 2056)))
    batch_size = 128

    for batch in test_split.getTrainIterator(batch_size):
        start = 0
        end = batch_size
        assert batch == list(range(start, end))
        start += batch_size
        end = start + batch_size

def test_split_getTrainIterator_Batchsize1():
    test_split = Split(train=list(range(1024)), test=list(range(1024, 2056)))
    batch_size = 1

    for batch in test_split.getTrainIterator(batch_size):
        start = 0
        end = batch_size
        assert batch == list(range(start, end))
        start += batch_size
        end = start + batch_size

def test_split_getTrainIterator_Batchsize2LARGE():
    test_split = Split(train=list(range(1024)), test=list(range(1024, 2056)))
    batch_size = 5000

    for batch in test_split.getTrainIterator(batch_size):
        assert batch == list(range(0, 1024))

def test_split_getTestIterator_Batchsize128():
    test_split = Split(train=list(range(1024)), test=list(range(1024, 2056)))
    batch_size = 128

    for batch in test_split.getTestIterator(batch_size):
        start = 1024
        end = start + batch_size
        assert batch == list(range(start, end))
        start += batch_size
        end = start + batch_size

def test_split_getTestIterator_Batchsize1():
    test_split = Split(train=list(range(1024)), test=list(range(1024, 2056)))
    batch_size = 1

    for batch in test_split.getTestIterator(batch_size):
        start = 1024
        end = start + batch_size
        assert batch == list(range(start, end))
        start += batch_size
        end = start + batch_size

def test_split_getTestIterator_Batchsize2LARGE():
    test_split = Split(train=list(range(1024)), test=list(range(1024, 2056)))
    batch_size = 5000

    for batch in test_split.getTestIterator(batch_size):
        assert batch == list(range(1024, 2056))