from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from tqdm import tqdm

from EasyChemML.DataImport.Module.HDF5 import HDF5
from EasyChemML.Utilities.DataUtilities.BatchDatatyp import BatchDatatyp, BatchDatatypClass
from EasyChemML.Utilities.DataUtilities.BatchDatatypHolder import BatchDatatypHolder
from EasyChemML.Utilities.DataUtilities.BatchPartition import BatchPartition

zinc20_length = HDF5('/dataset/ZINC20/ZINC20_withOutStereo.HDF5', columns=['smiles'], tableName='ZINC20').get_shape()[0]
step_size = 10000
threads = 12
# Generate Split

# OpenZINC20 Database
ZINC20_batchPartition = BatchPartition('/dataset/ZINC20/ZINC20_withOutStereo.HDF5', read_only=True, load_existing=True)
zinc20_table = ZINC20_batchPartition['ZINC20']

# Create Split Database
zinc20_split_Dtyps: BatchDatatypHolder = BatchDatatypHolder()
zinc20_split_Dtyps['ID'] = BatchDatatyp(BatchDatatypClass.NUMPY_INT64)
zinc20_split_Dtyps['SMILES'] = BatchDatatyp(BatchDatatypClass.NUMPY_STRING)

new_batchPartition = BatchPartition('ZINC20_MW_lower1001.HDF5', step_size)

tmp_arr = zinc20_split_Dtyps.createAEmptyNumpyArray(step_size)

cleaned_database = new_batchPartition.createDatabase(f'ZINC20', zinc20_split_Dtyps, shape=(zinc20_length,))

add_mols = 0
with tqdm(total=zinc20_length) as bar:
    current_index = 0
    for zinc20_index in range(0, zinc20_length, step_size):
        if zinc20_index+step_size>zinc20_length:
            chunk = zinc20_table[zinc20_index:]['smiles']
        else:
            chunk = zinc20_table[zinc20_index:zinc20_index+step_size]['smiles']

        current_set = 0
        for i,item in enumerate(chunk):
            if ExactMolWt(Chem.MolFromSmiles(item)) > 1000:
                print(f'found mol with mol-weight over 1000: {tmp_arr[i]["SMILES"]}')
                continue

            tmp_arr[current_set]['ID'] = add_mols
            tmp_arr[current_set]['SMILES'] = item.decode("utf-8")
            current_set+=1

        cleaned_database[current_index:current_index+current_set] = tmp_arr[0:current_set]
        bar.update(current_set)
        add_mols += current_set
        current_index += current_set

new_batchPartition.get_struc().data_grp['ZINC20'].resize((add_mols,))
new_batchPartition.flush()
