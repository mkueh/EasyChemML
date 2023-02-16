import numpy as np

from EasyChemML.DataImport.Module.HDF5 import HDF5
from EasyChemML.Utilities.DataUtilities.BatchDatatyp import BatchDatatyp, BatchDatatypClass
from EasyChemML.Utilities.DataUtilities.BatchDatatypHolder import BatchDatatypHolder
from EasyChemML.Utilities.DataUtilities.BatchPartition import BatchPartition

zinc20_length = HDF5('/dataset/ZINC20/ZINC20_100mio_CV3.HDF5', columns=['smiles'], tableName='split_1').get_shape()[0]
step_size = 100000000
threads = 12

cv_iterations = 1
# Generate Split


# OpenZINC20 Database
ZINC20_batchPartition = BatchPartition('/dataset/ZINC20/ZINC20_100mio_CV3.HDF5', read_only=True, load_existing=True)
zinc20_table = ZINC20_batchPartition['split_1']

# Create Split Database
zinc20_split_Dtyps: BatchDatatypHolder = BatchDatatypHolder()
zinc20_split_Dtyps['ID'] = BatchDatatyp(BatchDatatypClass.NUMPY_INT64)
zinc20_split_Dtyps['SMILES'] = BatchDatatyp(BatchDatatypClass.NUMPY_STRING)

new_batchPartition = BatchPartition('ZINC20_100mio_shuffled.HDF5', 500000)
shuffled_database = new_batchPartition.createDatabase(f'shuffled', zinc20_split_Dtyps,
                                                      shape=(zinc20_length,))

tmp_arr = zinc20_split_Dtyps.createAEmptyNumpyArray(step_size)
hole_data = zinc20_table[:]

print('first 3 rows before shuffle')
print(hole_data[0])
print(hole_data[1])
print(hole_data[2])

np.random.shuffle(hole_data)

print('first 3 rows after shuffle')
print(hole_data[0])
print(hole_data[1])
print(hole_data[2])

shuffled_database[:] = hole_data
new_batchPartition.flush()
print('finish')
