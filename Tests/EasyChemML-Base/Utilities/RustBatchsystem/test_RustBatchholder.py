from pathlib import Path

import numpy as np
from numpy.lib import recfunctions as rfn

from EasyChemML.DataImport.DataImporter import DataImporter
from EasyChemML.Environment import Environment
from EasyChemML.Utilities.DataUtilities.RustBatchsystem.pyWrapper.RustBatchSorter_Radix import RustBatchSorter_Radix
from EasyChemML.Utilities.DataUtilities.RustBatchsystem.pyWrapper.RustBatchholder import RustBatchholder, MemoryMode


def test_sort():
    env = Environment(WORKING_path='/mydata/workdir', TMP_path='/mydata/workdir/tmp')

    load_dataset = {}
    di = DataImporter(env)
    dh = di.load_data_InNewBatchPartition(load_dataset)

    dt = np.dtype([('a', 'i4', 200)])
    arr = np.random.randint(20000, size=(1000, 200))
    arr = rfn.unstructured_to_structured(arr, dt)
    dh.addNumpyArray(arr, 'test_array')

    rustholderPath = Path(env.TMP_path).joinpath('RustBatch')
    rb = RustBatchholder(str(rustholderPath), 10000)
    sorter = RustBatchSorter_Radix(str(rustholderPath))
    rb.transferToRust(dh, 'test_array', memMode=MemoryMode.DirectIO)
    rusttable = rb.getRustBatchTable('test_array')


    sorter.sort(rb, 'test_array')

    rb.transferToBatchtable('test_array', dh, 'test_array_new')
    #print(dh['test_array_new'].to_String())

    rb.clean()
