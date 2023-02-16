import requests, pandas as pd, io, numpy as np, os, shutil, glob
import lz4.frame, pickle
from typing import List
from joblib import Parallel, delayed

URL_list_Path = 'ZINC20-URL.txt'
output_path = 'dataset'

def compress_object(object, lz4_args={}):
    pickled_object = pickle.dumps(object, protocol=pickle.HIGHEST_PROTOCOL)
    return lz4.frame.compress(pickled_object, **lz4_args)

def compress_object_to_file(object, file_path, lz4_args={}):
        data = compress_object(object, lz4_args)

        f = open(file_path, 'wb')
        f.write(data)
        f.flush()
        f.close()

def file_2_list(path:str):
    out = []
    with open(path, 'r') as f:
        for line in f:
            out.append(line.rstrip())
    return out

def resume_download(path:str):
    globstatment = path + '/*.npick'
    npickFiles = glob.glob(globstatment)

    downloaded = []
    for npick in npickFiles:
        downloaded.append(npick[-10:-6])

    return downloaded

def delete_downloadedURLS(downloaded_keys:List[str], urls:List[str]):
    if len(downloaded_keys) == 0:
        return urls

    delete_indices = []
    for i, url in enumerate(urls):
        tranch_name = url[-8:-4]
        if tranch_name in downloaded_keys:
            delete_indices.append(i)

    np_array = np.array(urls)
    np_array = np.delete(np_array, delete_indices)

    return np_array

def download_tranch(i:int, url:str):
        tranch_name = url[-8:-4]
        print(f'load tranch {tranch_name} | loaded {i+1} of {len(urls)+1}')

        r = requests.get(url)
        data = io.StringIO(r.text)
        dataframe = pd.read_csv(data, delimiter='\t')
        np_array = np.rec.fromrecords(dataframe, names=dataframe.columns.tolist())
        compress_object_to_file(np_array, os.path.join(output_path,f'{tranch_name}.npick'))

if __name__ == '__main__':
    print(f'look for {URL_list_Path}')
    if not os.path.exists(URL_list_Path):
        print(f'could not found {URL_list_Path}')
        raise Exception()

    urls = file_2_list(URL_list_Path)

    print(f'create {output_path} folder')
    if os.path.exists(output_path):
        downloaded = resume_download(output_path)
        urls = delete_downloadedURLS(downloaded, urls)
    else:
        os.mkdir(output_path)



    Parallel(n_jobs=4)(delayed(download_tranch)(i, urls[i]) for i in range(len(urls)))


    print('finished')
