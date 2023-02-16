from EasyChemML.DataImport.DataImporter import DataImporter
from EasyChemML.DataImport.Module.HDF5 import HDF5
from EasyChemML.Environment import Environment

env = Environment()
zinc20_hdfLoader = HDF5('/dataset/ZINC20/ZINC20_100mio_shuffled.HDF5', columns=['smiles'], tableName='shuffled')
step_size = 10000
threads = 12


di = DataImporter(env)
bp = di.load_data_InNewBatchPartition(zinc20_hdfLoader, max_chunksize=100000)
shuffled_bt = bp['shuffled']

pass