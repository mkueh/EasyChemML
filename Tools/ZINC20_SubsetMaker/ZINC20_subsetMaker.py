import numpy as np

from tqdm import tqdm

from EasyChemML.DataImport.Module.HDF5 import HDF5
from EasyChemML.Utilities.DataUtilities.BatchDatatyp import BatchDatatyp, BatchDatatypClass
from EasyChemML.Utilities.DataUtilities.BatchDatatypHolder import BatchDatatypHolder
from EasyChemML.Utilities.DataUtilities.BatchPartition import BatchPartition

zinc20_length = HDF5('/dataset/ZINC20/ZINC20_MW_lower1001.HDF5', columns=['smiles'], tableName='ZINC20').get_shape()[0]
step_size = 10000
threads = 12

split_size = 100 * 1000 * 1000
cv_iterations = 1
# Generate Split


# OpenZINC20 Database
ZINC20_batchPartition = BatchPartition('/dataset/ZINC20/ZINC20_MW_lower1001.HDF5', read_only=True, load_existing=True)
zinc20_table = ZINC20_batchPartition['ZINC20']

# Create Split Database
zinc20_split_Dtyps: BatchDatatypHolder = BatchDatatypHolder()
zinc20_split_Dtyps['ID'] = BatchDatatyp(BatchDatatypClass.NUMPY_INT64)
zinc20_split_Dtyps['SMILES'] = BatchDatatyp(BatchDatatypClass.NUMPY_STRING)

new_batchPartition = BatchPartition('ZINC20_100mio_CV3.HDF5', 500000)

tmp_arr = zinc20_split_Dtyps.createAEmptyNumpyArray(step_size)
with tqdm(total=split_size * cv_iterations) as bar:
    for cv_split_i in range(cv_iterations):
        split_database = new_batchPartition.createDatabase(f'split_{cv_split_i}', zinc20_split_Dtyps,
                                                           shape=(split_size,))
        subset_indices = np.random.choice(zinc20_length, split_size, replace=False)
        sorted_subset_indices = np.sort(subset_indices)

        for current_range in range(0, split_size, step_size):
            current_set = 0
            if current_range + step_size > split_size:
                indicies = sorted_subset_indices[current_range:]
            else:
                indicies = sorted_subset_indices[current_range:current_range + step_size]

            current_INdata = zinc20_table[indicies]['SMILES']

            for enum_index, val in enumerate(current_INdata):
                tmp_arr[enum_index]['ID'] = enum_index
                tmp_arr[enum_index]['SMILES'] = val
                current_set += 1

            split_database[current_range:current_range + current_set] = tmp_arr[0:enum_index + 1]
            bar.update(enum_index + 1)

        new_batchPartition.flush()
print('finish')