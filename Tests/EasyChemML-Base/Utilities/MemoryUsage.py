import os, numpy as np
import time
from datetime import datetime

from EasyChemML.Environment import Environment
from EasyChemML.Utilities.DataUtilities.BatchPartition import BatchPartition, BatchPartitionMode
from EasyChemML.DataImport.DataImporter import DataImporter
from EasyChemML.Utilities.DataUtilities.BatchDatatypHolder import BatchDatatypHolder
from EasyChemML.Utilities.DataUtilities.BatchDatatyp import BatchDatatyp, BatchDatatypClass

env = Environment(TMP_path='./TMP')


class test_object():
    def __init__(self, value):
        self.val = value

def initFile():
    DI = DataImporter(env)
    test_shape = 800000000

    test_batchPartition = BatchPartition(os.path.join(env.TMP_path, 'tmp.mem'), mode=BatchPartitionMode.demand_IO)

    datatypHolder = BatchDatatypHolder()
    datatypHolder << ('A_INTEGER', BatchDatatyp(BatchDatatypClass.NUMPY_INT32))
    datatypHolder << ('B_INTEGER', BatchDatatyp(BatchDatatypClass.NUMPY_INT32))
    datatypHolder << ('C_INTEGER', BatchDatatyp(BatchDatatypClass.NUMPY_INT32))
    datatypHolder << ('D_INTEGER', BatchDatatyp(BatchDatatypClass.NUMPY_INT32))

    test_batchPartition.createDatabase(f'test', datatypHolder, shape=(test_shape,), without_compressor=True)

    batchtable = test_batchPartition['test']

    for_step = 100000000
    for i in range(0, test_shape, for_step):
        start_time = datetime.now()
        print(i)
        arr = np.full((for_step,), i, datatypHolder.toNUMPY_dtypes())
        batchtable[i-for_step:i] = arr
        time_elapsed = datetime.now() - start_time

        print('Time elapsed (hh:mm:ss) {}'.format(time_elapsed))
    test_batchPartition.flush()




# INSERT YOUR CODE





def create_EmptyDatabase():
    initFile()
    time.sleep(10000)
    print('test')

if __name__ == '__main__':
    create_EmptyDatabase()