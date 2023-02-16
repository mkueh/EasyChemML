import sys
from pathlib import Path
import os

import numpy as np

from pyRustBatchsystem import BatchHolder_py
from pyRustBatchsystem import BatchSorter_Radix_py

programm_path, file = os.path.split(sys.argv[0])
tmp_path = os.path.join(programm_path, 'pyRustBatchsystem/TMP')

bh = BatchHolder_py(tmp_path)
bh.create_new_table('test', [100,100])

bt_f64_test = bh.get_batchtable_f64('test')
random_arr = np.random.rand(100,100)

bt_f64_test.add_chunk(random_arr)
reload_arr = bt_f64_test.load_chunk(0)

for i,row in enumerate(random_arr):
    for j,col in enumerate(row):
        assert col == reload_arr[i,j]

bh.clean()

