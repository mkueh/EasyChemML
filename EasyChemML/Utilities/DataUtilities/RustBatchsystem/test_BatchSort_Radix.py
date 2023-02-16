import sys
from pathlib import Path
import os

import numpy as np

from pyRustBatchsystem import BatchHolder_py
from pyRustBatchsystem import BatchSorter_Radix_py
from pyRustBatchsystem import BatchListFunctions_duplicates_py

programm_path, file = os.path.split(sys.argv[1])
tmp_path = os.path.join(programm_path, os.path.join('pyRustBatchsystem','TMP'))

bh = BatchHolder_py(tmp_path)
bh.create_new_table('test', [40])

bt_i8_test = bh.get_batchtable_i8('test')

for i in range(10):
    random_arr = np.random.randint(0, 2, size=(10, 10), dtype='int8')
    bt_i8_test.add_chunk(random_arr)

#reload_arr = bt_i8_test.load_chunk(0)
#for i, row in enumerate(random_arr):
#    for j, col in enumerate(row):
#        assert col == reload_arr[i, j]

sorter = BatchSorter_Radix_py(os.path.join(tmp_path, 'sort_tmp'))

sorter.sort_i8(bt_i8_test)
bt_i8_test.print_arr()

reload_arr = bt_i8_test.load_chunk(0)
random_arr = np.sort(random_arr)

batchlist_duplicates = BatchListFunctions_duplicates_py()
out = batchlist_duplicates.count_duplicates_on_sorted_list_i8(bt_i8_test, True)
#for i, row in enumerate(random_arr):
#    print(reload_arr[i])

bh.clean()
