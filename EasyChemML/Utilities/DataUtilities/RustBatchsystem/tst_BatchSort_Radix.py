import sys
from pathlib import Path
import os

import numpy as np

from pyRustBatchsystem import BatchHolder_py
from pyRustBatchsystem import BatchSorter_Radix_py
from pyRustBatchsystem import BatchListFunctions_duplicates_py

programm_path, file = os.path.split(sys.argv[0])
tmp_path = os.path.join(programm_path, os.path.join('pyRustBatchsystem', 'TMP'))
os.mkdir(tmp_path)

bh = BatchHolder_py(tmp_path)
bh.create_new_table('test', [3], 'InMemory')

bt_i8_test = bh.get_batchtable_i8('test')



for i in range(10):
    random_arr = np.random.randint(0, 1, size=(3, 10), dtype='int8')
    bt_i8_test.add_chunk(random_arr)

random_arr = np.random.randint(2, 4, size=(2, 10), dtype='int8')
bt_i8_test.add_chunk(random_arr)


sorter = BatchSorter_Radix_py(os.path.join(tmp_path, 'sort_tmp'))

bt_i8_test.print_arr()
sorter.sort_i8(bt_i8_test)
bt_i8_test.print_arr()

reload_arr = bt_i8_test.load_chunk(10)
random_arr = np.sort(random_arr)

batchlist_duplicates = BatchListFunctions_duplicates_py()
out = batchlist_duplicates.count_duplicates_on_sorted_list_i8(bt_i8_test, True)

bh.clean()
