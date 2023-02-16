import numpy as np
import os

from tqdm import tqdm

from EasyChemML.DataImport.Module.XLSX import XLSX
from EasyChemML.Environment import Environment
from EasyChemML.Utilities.DataUtilities.BatchPartition import BatchPartition, BatchPartitionMode
from EasyChemML.DataImport.DataImporter import DataImporter
from EasyChemML.Utilities.DataUtilities.BatchTable import BatchAccess
from EasyChemML.Utilities.DataUtilities.BatchDatatypHolder import BatchDatatypHolder
from EasyChemML.Utilities.DataUtilities.BatchDatatyp import BatchDatatyp, BatchDatatypClass

from EasyChemML.Splitter.Splitcreator import Split
from typing import List

env = Environment(TMP_path='./TMP')


class test_object():
    def __init__(self, value):
        self.val = value


def initFile(column: List[str], name: str):
    DI = DataImporter(env)

    XLSX_loader = XLSX('../../../Examples/DATASETS/Dreher_and_Doyle_input_data.xlsx', 'FullCV_01')

    dataHolder = DI.load_data_InNewBatchPartition({'test': XLSX_loader},
                                                  batchmode=BatchPartitionMode.in_memory)
    return dataHolder


def initFiles():
    DI = DataImporter(env)

    X_columns = ['Ligand', 'Additive']
    y_columns = ['Additive', 'Output']

    X_loader = XLSX('../../../Examples/DATASETS/Dreher_and_Doyle_input_data.xlsx', 'FullCV_01', columns=X_columns)
    y_loader = XLSX('../../../Examples/DATASETS/Dreher_and_Doyle_input_data.xlsx', 'FullCV_01', columns=y_columns)

    d = {'Xtest': X_loader, 'ytest': y_loader}

    dataHolder = DI.load_data_InNewBatchPartition(d, batchmode=BatchPartitionMode.in_memory)

    return dataHolder


def create_randomSplit(count: int, test=0.3):
    arr = np.arange(count)
    np.random.shuffle(arr)
    test_len = int(len(arr) * 0.3)

    split = Split(train=np.sort(arr[test_len:]), test=np.sort(arr[0:test_len]))
    return split


def create_EasyDatabase(size: int = 1000):
    batchDtyps = BatchDatatypHolder()

    batchDtyps['C1'] = BatchDatatypClass.NUMPY_FLOAT64
    batchDtyps['C2'] = BatchDatatypClass.NUMPY_FLOAT64

    bp = BatchPartition("memdisk.disk")
    bt = bp.createDatabase('easyDatabase', batchDtyps, size)

    iterator = iter(bt)
    counter = 0
    for batch in iterator:
        for i, item in enumerate(batch):
            batch['C1'][i] = counter + size
            batch['C2'][i] = counter
            counter += 1
        iterator << batch

    return bp


def create_3DDatabase(size: int = 1000):
    batchDtyps = BatchDatatypHolder()

    batchDtyps['C1'] = BatchDatatypClass.NUMPY_FLOAT64
    batchDtyps['C1'].set_shape((20,))
    batchDtyps['C2'] = BatchDatatypClass.NUMPY_FLOAT64
    batchDtyps['C2'].set_shape((30,))
    batchDtyps['C3'] = BatchDatatypClass.NUMPY_INT32
    batchDtyps['C3'].set_shape((30,))
    batchDtyps['C4'] = BatchDatatypClass.NUMPY_INT32
    batchDtyps['C4'].set_shape((30,))
    batchDtyps['C5'] = BatchDatatypClass.NUMPY_STRING
    batchDtyps['C5'].set_shape(None)

    bp = BatchPartition("memdisk.disk")
    bt = bp.createDatabase('3DDatabase', batchDtyps, size)

    iterator = iter(bt)
    counter = 0
    for batch in iterator:
        for i, item in enumerate(batch):
            batch['C1'][i] = list(range(counter + size, counter + size + 20))
            batch['C2'][i] = list(range(counter, counter + 30))
            batch['C3'][i] = list(range(counter, counter + 30))
            batch['C4'][i] = list(range(counter, counter + 30))
            batch['C5'][i] = b'Hallo'
            counter += 1
        iterator << batch

    return bp


def test_convert_2_ndarray_easy():
    bp = create_EasyDatabase(1000)
    bt = bp['easyDatabase']

    data = bt.convert_2_ndarray()

    counter = 0
    for row in data:
        assert row[0] == counter + 1000
        assert row[1] == counter
        counter+=1

def test_convert_2_ndarray_3D():
    bp = create_3DDatabase(1000)
    bt = bp['3DDatabase']

    tmp = bt['C1','C2']
    data = bt.convert_2_ndarray(columns=['C1','C2'])


    counter = 0
    for row in data:
        concat = list(range(counter + 1000, counter + 1000 + 20)) + list(range(counter, counter + 30))
        for index,item in enumerate(row):
            assert item == concat[index]
        counter+=1

def test_convert_2_ndarray_3D_colselect():
    bp = create_3DDatabase(1000)
    bt = bp['3DDatabase']

    data = bt.convert_2_ndarray(columns=['C1','C2'])

    counter = 0
    for row in data:
        concat = list(range(counter + 1000, counter + 1000 + 20)) + list(range(counter, counter + 30))
        for index, item in enumerate(row):
            assert item == concat[index]
        counter += 1


def test_create_EmptyDatabase():
    dataHolder = initFiles()
    batchDtyps = BatchDatatypHolder()

    batchDtyps['C1'] = BatchDatatypClass.NUMPY_STRING
    batchDtyps['C2'] = BatchDatatypClass.NUMPY_FLOAT64

    db = dataHolder.createDatabase('test', batchDtyps, 1000)

    iterator = iter(db)
    counter = 0
    for batch in iterator:
        for i, item in enumerate(batch):
            batch['C1'][i] = b'testvalue'
            batch['C2'][i] = counter
            counter += 1
        iterator << batch

    assert db['C1'][0] == b'testvalue'
    assert db['C2'][156] == 156.0


def test_add_PandaDataFrame_easy():
    dataHolder = initFiles()
    if 'Xtest' in dataHolder.keys() and 'ytest' in dataHolder.keys():
        assert True
    else:
        assert False


def test_get_BatchTable_easy():
    dataHolder = initFiles()

    test = dataHolder.get_BatchTable('Xtest')
    assert test['Ligand'][
               0] == b'CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P([C@@]3(C[C@@H]4C5)C[C@H](C4)C[C@H]5C3)[C@]6(C7)C[C@@H](C[C@@H]7C8)C[C@@H]8C6)C(OC)=CC=C2OC'
    assert test['Additive'][0] == b'CC1=CC(C)=NO1'

    test = dataHolder.get_BatchTable('ytest')
    assert test['Additive'][0] == b'CC1=CC(C)=NO1'
    assert test['Output'][0] == 70.41045785


def test_BatchTable_dtypes():
    dataHolder = initFiles()

    y_Table = dataHolder.get_BatchTable('ytest')
    print(f'\ny: {y_Table.getDatatypes()}')

    X_Table = dataHolder.get_BatchTable('Xtest')
    print(f'X: {X_Table.getDatatypes()}')


def test_Batchprocessing_easy():
    dataHolder = initFiles()

    y_Table = dataHolder.get_BatchTable('ytest')
    iterator = iter(y_Table)
    for batch in iterator:
        for j, item in enumerate(batch):
            if j == 0 or j == 1:
                batch[j][0] = 'Hallo'
            if j == len(batch) - 1:
                batch[j][0] = 'Hallo'

        iterator <<= batch
    test = y_Table['Additive', 'Output']
    assert y_Table['Additive'][0] == b'Hallo'
    assert y_Table[1][0] == b'Hallo'
    assert y_Table[len(y_Table) - 1][0] == b'Hallo'


def test_addObject():
    dataHolder = initFiles()
    y_Table = dataHolder.get_BatchTable('ytest')
    iterator: BatchAccess = iter(y_Table)
    batch: np.ndarray
    value = 0
    dataTypHolder: BatchDatatypHolder = y_Table.getDatatypes()
    dataTypHolder['Additive'] = BatchDatatyp(BatchDatatypClass.PYTHON_OBJECT)

    for batch in iterator:
        new_c = dataTypHolder.createAEmptyNumpyArray(len(batch))
        new_c['Output'] = batch['Output']
        for j, item in enumerate(batch):
            new_c['Additive'][j] = test_object(value)
            value += 1
        iterator <<= new_c
    assert y_Table['Additive'][1].val == 1
    assert y_Table[0][0].val == 0
    assert y_Table[0][1] == 70.41045785
    assert y_Table[1][0].val == 1
    assert y_Table[len(y_Table) - 1][0].val == 3954


def test_addObject_Xtable():
    dataHolder = initFiles()
    X_Table = dataHolder.get_BatchTable('Xtest')
    iterator: BatchAccess = iter(X_Table)
    batch: np.ndarray
    value = 0
    dataTypHolder: BatchDatatypHolder = X_Table.getDatatypes()
    dataTypHolder['Ligand'] = BatchDatatyp(BatchDatatypClass.PYTHON_OBJECT)

    for batch in iterator:
        new_c = dataTypHolder.createAEmptyNumpyArray(len(batch))
        for j, item in enumerate(batch):
            new_c['Ligand'][j] = test_object(value)
            new_c[j]['Additive'] = batch[j]['Additive']
            value += 1
        iterator <<= new_c

    assert X_Table['Ligand'][1].val == 1
    assert X_Table[0][0].val == 0
    assert X_Table[0][1] == b'CC1=CC(C)=NO1'
    assert X_Table[1][0].val == 1
    assert X_Table[len(X_Table) - 1][0].val == 3954
    assert X_Table[len(X_Table) - 1][1] == b'C1(C2=CC=CC=C2)=CC=NO1'

def test_addObject_withoutIterator_Xtable():
    dataHolder = initFiles()
    X_Table = dataHolder.get_BatchTable('Xtest')
    iterator: BatchAccess = iter(X_Table)

    value = 0
    dataTypHolder: BatchDatatypHolder = X_Table.getDatatypes()
    dataTypHolder['Ligand'] = BatchDatatyp(BatchDatatypClass.PYTHON_OBJECT)

    for batch in iterator:
        new_c = dataTypHolder.createAEmptyNumpyArray(len(batch))
        for j, item in enumerate(batch):
            new_c['Ligand'][j] = test_object(value)
            new_c[j]['Additive'] = batch[j]['Additive']
            value += 1
        iterator <<= new_c

    test = X_Table[:4]

    #print(X_Table.to_String(4))

    test[0]['Ligand'] = test_object(10)
    test[1]['Ligand'] = test_object(11)
    test[2]['Ligand'] = test_object(12)
    test[3]['Ligand'] = test_object(13)
    X_Table[:4] = test
    #print('after change')
    #print(X_Table.to_String(4))

    assert X_Table['Ligand'][1].val == 11
    assert X_Table[0][0].val == 10
    assert X_Table[0][1] == b'CC1=CC(C)=NO1'
    assert X_Table[3][1] == b'CCOC(C1=CON=C1)=O'
    assert X_Table[3][0].val == 13
    assert X_Table[1][0].val == 11
    assert X_Table[len(X_Table) - 1][0].val == 3954
    assert X_Table[len(X_Table) - 1][1] == b'C1(C2=CC=CC=C2)=CC=NO1'


def test_createNumpyArray():
    dataHolder = initFiles()
    y_Table = dataHolder.get_BatchTable('ytest')
    dataTypHolder: BatchDatatypHolder = y_Table.getDatatypes()
    newnumpy = dataTypHolder.createAEmptyNumpyArray(10)

    assert isinstance(newnumpy, np.ndarray)
    assert len(newnumpy) == 10

    test = BatchDatatypHolder()
    test.fromNUMPY_dtyp(newnumpy.dtype)

    assert test == dataTypHolder


def test_Batchprocessing_diffDtype():
    dataHolder = initFiles()
    y_Table = dataHolder.get_BatchTable('ytest')
    new_dataTyp: BatchDatatypHolder = y_Table.getDatatypes()
    iterator: BatchAccess = iter(y_Table)
    batch: np.ndarray
    value = 0

    new_dataTyp['Additive'] = BatchDatatypClass.NUMPY_FLOAT64
    for batch in iterator:
        new_c = new_dataTyp.createAEmptyNumpyArray(len(batch))
        new_c['Output'] = batch['Output']
        for j, item in enumerate(batch):
            new_c['Additive'][j] = float(value)
            value += 1
        iterator <<= (new_c)
    assert y_Table[0][0] == float(0)
    assert y_Table[1][0] == float(1)
    assert y_Table[len(y_Table) - 1][0] == float(3954.0)


def test_fromNUMPY_dtyp():
    first_dataTypHolder = BatchDatatypHolder()
    first_dataTypHolder << ('col1', BatchDatatyp(BatchDatatypClass.NUMPY_INT8, (32,)))
    first_dataTypHolder << ('col2', BatchDatatyp(BatchDatatypClass.NUMPY_INT8))
    first_dataTypHolder << ('col3', BatchDatatyp(BatchDatatypClass.NUMPY_STRING))

    secound_dataTypHolder = BatchDatatypHolder()
    secound_dataTypHolder.fromNUMPY_dtyp(first_dataTypHolder.toNUMPY_dtypes())

    assert first_dataTypHolder == secound_dataTypHolder
    assert first_dataTypHolder.toNUMPY_dtypes() == secound_dataTypHolder.toNUMPY_dtypes()

def test_addNumpyArrayToBatchPartition():
    bp = BatchPartition(os.path.join(env.TMP_path, 'memdisk.disk'), 100000)
    x = np.array([('Rex', 9, 81.0), ('Fido', 3, 27.0)],
                 dtype=[('name', 'U10'), ('age', 'i4'), ('weight', 'f4')])
    bt = bp.addNumpyArray(x, 'test')

    assert bt[0]['name'] == b'Rex'
    assert bt[0]['age'] == 9
    assert bt[0]['weight'] == 81.0
    assert bt[1]['name'] == b'Fido'
    assert bt[1]['age'] == 3
    assert bt[1]['weight'] == 27.0

def test_createSubset_reorder():
    dataHolder = initFiles()
    y_Table = dataHolder.get_BatchTable('ytest')

    split = create_randomSplit(len(y_Table))
    yreorderd_Table, new_split = dataHolder.createReorderedSubset(y_Table, split, 'y_reordered')

    # check train
    with tqdm(total=len(split.train), desc='Check train subset') as pbar:
        for i, item in enumerate(split.train):
            assert y_Table[item] == yreorderd_Table[new_split.train[i]]
            pbar.update(1)

    # print('train is checked')

    # check test
    with tqdm(total=len(split.test), desc='Check test subset') as pbar:
        for i, item in enumerate(split.test):
            assert y_Table[item] == yreorderd_Table[new_split.test[i]]
            pbar.update(1)

    # print('test is checked')
