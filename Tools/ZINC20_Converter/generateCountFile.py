import glob, os, lz4.frame, pickle


def decompress_object(data, lz4_args={}):
    raw_pickle = lz4.frame.decompress(data, **lz4_args)
    return pickle.loads(raw_pickle)

def decompress_object_from_file(file_path, lz4_args={}):
    f = open(file_path, 'rb')
    data = f.read()
    f.close()
    return decompress_object(data, lz4_args)

def create_countFiles(folder_path:str) -> int:
    counter = 0
    fingerprints_files = glob.glob(os.path.join(folder_path,'*.npick'))
    fingerprints_files_counts = glob.glob(os.path.join(folder_path, '*.count'))

    for i, fingerprint_file in enumerate(fingerprints_files):
        filename = fingerprint_file[:-6]
        filename += '.count'

        if filename in fingerprints_files_counts:
            print(f'found a count file for {fingerprint_file}')
            continue

        data = decompress_object_from_file(file_path = fingerprint_file)
        counter += len(data)

        with open(filename, "w") as file:
            file.write(f'{len(data)}')
        print(f'count in {fingerprint_file} {len(data)} mols | {i+1} of {len(fingerprints_files)}')

    return counter

if __name__ == '__main__':
    create_countFiles('dataset')
