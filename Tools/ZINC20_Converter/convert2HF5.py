import glob, lz4.frame, pickle, numpy as np

import pandas as pd

from EasyChemML.Utilities.DataUtilities.BatchDatatypHolder import BatchDatatypHolder
from EasyChemML.Utilities.DataUtilities.BatchPartition import BatchPartition


def decompress_object(data, lz4_args={}):
    raw_pickle = lz4.frame.decompress(data, **lz4_args)
    return pickle.loads(raw_pickle)


def decompress_object_from_file(file_path, lz4_args={}):
    f = open(file_path, 'rb')
    data = f.read()
    f.close()
    return decompress_object(data, lz4_args)


def get_size(folder_path: str) -> int:
    counter = 0
    glob_search = folder_path + '*.count'
    fingerprints_countfiles = glob.glob(glob_search)

    for i, fingerprints_countfile in enumerate(fingerprints_countfiles):
        count = -1
        with open(fingerprints_countfile, "r") as file:
            raw = file.read()
            count = int(raw)
        counter += count
        print(f'found countfile for {fingerprints_countfile} | {i + 1} of {len(fingerprints_countfiles)}')
    print(f'total count {counter}')

    return counter


def get_dataTypes(folder_path: str):
    glob_search = folder_path + '*.npick'
    fingerprints_files = glob.glob(glob_search)
    data: np.ndarray = decompress_object_from_file(file_path=fingerprints_files[0])
    return data.dtype


def preprocessing_data(data: np.ndarray) -> np.ndarray:
    # remove stereo

    for datapoint in data:
        smiles = datapoint[0]
        datapoint[0] = smiles.replace('@', '')

    df = pd.DataFrame(data)
    df = df.duplicated(subset=['smiles'], keep='first').tolist()

    idx = []
    for i, val in enumerate(df):
        if val:
            idx.append(i)
    data = np.delete(data, idx)
    return data


if __name__ == '__main__':
    new_batchPartition = BatchPartition('ZINC20_withOutStereo.HDF5', 10000000)
    path = '/dataset/ZINC20/dataset/'

    size = get_size(path)
    dtypes = get_dataTypes(path)
    batchDtyps: BatchDatatypHolder = BatchDatatypHolder()
    batchDtyps.fromNUMPY_dtyp(dtypes)
    del batchDtyps['X']

    new_batchPartition.createDatabase('ZINC20', batchDtyps, size)
    table = new_batchPartition['ZINC20']

    fingerprints_files = glob.glob(path + '*.npick')
    current_index = 0
    for i, fingerprint_file in enumerate(fingerprints_files[0:10]):
        print(f'decompress {i + 1} of {len(fingerprints_files)} | process {fingerprint_file}')
        data: np.ndarray = decompress_object_from_file(file_path=fingerprint_file)
        dtpyes = batchDtyps.toH5PY_dtypes()
        data = data[:][batchDtyps.getColumns()]
        data = data.astype(dtpyes)
        data = preprocessing_data(data)
        print(f'write {len(data)} mols to the HF5-File, total written {current_index}')
        print(f' ---- content test: {data[0]}')
        table[current_index: current_index + len(data)] = data[:]
        current_index += len(data)
    print(f'finished! after loading {current_index} datapoints')
    new_batchPartition.get_struc().data_grp['ZINC20'].resize((current_index,))
