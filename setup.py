import glob, os, subprocess, sys, shutil, datetime, hashlib, pickle, json, platform
from pathlib import Path

import setuptools

def removeOLDCython(path:str):
    build_path = os.path.join(path,'build')

    if os.path.exists(build_path):
        shutil.rmtree(build_path)

    files_cpp = [os.path.abspath(f) for f in glob.glob(path + '/**/*.cpp', recursive=True )]
    files_pyd = [os.path.abspath(f) for f in glob.glob(path + '/**/*.pyd', recursive=True)]
    files_so = [os.path.abspath(f) for f in glob.glob(path + '/**/*.so', recursive=True)]

    all_files = files_cpp + files_pyd + files_so

    for file in all_files:
        print(f'remove {file}')
        os.remove(file)

def hash_MD5_File(path:str):
    BUF_SIZE = 65536  # lets read stuff in 64kb chunks!

    md5 = hashlib.md5()

    with open(path, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            md5.update(data)

    return md5.hexdigest()

def create_version_data(path):
    excution_of_setup = datetime.datetime.now()
    excution_of_setup = excution_of_setup.strftime("%d/%m/%Y %H:%M:%S")
    print("date and time of excution_of_setup=", excution_of_setup)

    if os.path.exists("Version.json"):
        os.remove("Version.json")

    files_py = [os.path.abspath(f) for f in glob.glob(path + '/**/*.py', recursive=True)]
    files_pyx = [os.path.abspath(f) for f in glob.glob(path + '/**/*.pyx', recursive=True)]
    files_hjson = [os.path.abspath(f) for f in glob.glob(path + '/**/*.hjson', recursive=True)]
    working_path = os.path.join(path, 'workingdir')
    files_workingDir = set([os.path.abspath(f) for f in glob.glob(working_path + '/**/*.*', recursive=True)])
    Test_path = os.path.join(path, 'Test')
    files_Test = set([os.path.abspath(f) for f in glob.glob(Test_path + '/**/*.*', recursive=True)])

    file_list = list(set(files_py + files_pyx + files_hjson) - files_Test - files_workingDir)
    hash_list = []

    for file in file_list:
        hash = hash_MD5_File(file)
        hash_list.append(hash_MD5_File(file))
        print(f'File: {file} has HASH: {hash}')

    hash_list.sort()

    p = pickle.dumps(hash_list, -1)
    hash = hashlib.md5(p).hexdigest()

    output_json = {'time': excution_of_setup, 'hash': hash}
    with open("Version.json", "w") as write_file:
        json.dump(output_json, write_file)

def compile_Cython():
    print('Compile all Cython setup.py files')
    files = [os.path.abspath(f) for f in glob.glob(path + '/**/setup.py', recursive=True)]

    mfile = os.path.normpath(sys.argv[0])
    for i, f in enumerate(files):
        if str(f) == mfile:
            del files[i]

    for f in files:
        print(str(sys.executable) + ' ' + f)
        folder = os.path.normpath(f + os.sep + os.pardir)
        print('##### remove old Cython files #####')
        removeOLDCython(folder)
        print('##### compile Cython files #####')
        subprocess.call(str(sys.executable) + ' ' + f + ' build_ext --inplace', shell=True, cwd=os.path.dirname(f))

def compile_Rust():
    print('Compile all setup.rpy files')
    files = [os.path.abspath(f) for f in glob.glob(path + '/**/setup_rust.py', recursive=True)]

    mfile = os.path.normpath(sys.argv[0])
    for i, f in enumerate(files):
        if str(f) == mfile:
            del files[i]

    for f in files:
        print(str(sys.executable) + ' ' + f)
        folder = os.path.normpath(f + os.sep + os.pardir)
        print('##### remove old rust files #####')
        removeOLDCython(folder)
        print('##### compile rust files #####')
        subprocess.call(str(sys.executable) + ' ' + f + ' build_rust', shell=True, cwd=os.path.dirname(f))

path = Path(__file__).parent.absolute()
print(path)
path = str(path)

print('### remove all pyc data ###')
files_pyc = [os.path.abspath(f) for f in glob.glob(path+'/**/*.pyc', recursive=True)]

for f in files_pyc:
    print('remove: ' + f)
    os.remove(f)

compile_Cython()
compile_Rust()

print('##### create Version.json #####')
create_version_data(path)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

os_system_name = platform.system()

if os_system_name == 'Windows':
    install_requires = [
        "torch",
        "torchvision",
        "torchaudio",
        "setuptools>=42",
        "tokenizers",
        "wheel",
        "numpy",
        "multiprocess",
        "dataimport>=0.1.0",
        "tqdm",
        "enlighten",
        "pandas",
        "joblib",
        "catboost",
        "sklearn",
        "Cython==0.29.23",
        "bitarray",
        "sortedcontainers",
        "Pympler",
        "rdkit",
        "h5py",
        "hdf5plugin",
        "lz4",
        "deepchem",
        "hjson",
        "hyperopt>=0.2.5",
        "pyyaml",
        "deepdiff",
        'numcodecs',
        'PTable',
        'tables',
        'pytest',
        'xarray',
        'xlrd==1.2.0',
        'openpyxl'
    ]
else:
    install_requires = [
        "torch",
        "torchvision",
        "torchaudio",
        "setuptools>=42",
        "multiprocess",
        "tokenizers",
        "wheel",
        "numpy",
        "dataimport>=0.1.0",
        "tqdm",
        "pandas",
        "joblib",
        "catboost",
        "sklearn",
        "Cython==0.29.23",
        "bitarray",
        "sortedcontainers",
        "Pympler",
        "enlighten",
        "h5py",
        "hdf5plugin",
        "lz4",
        "deepchem",
        "hjson",
        "hyperopt>=0.2.5",
        "pyyaml",
        "rdkit",
        "deepdiff",
        'numcodecs',
        'PTable',
        'tables',
        'pytest',
        'xarray',
        'xlrd==1.2.0',
        'openpyxl'
    ]

#init_pyx_files = [f for f in glob.glob(path+'/**/__init__.py', recursive=True)]
setuptools.setup(
    name="EasyChemML",
    version="0.2.1.6",
    author="Marius Kuehnemund",
    author_email="m_kueh11@wwu.de",
    description="One framework to rule them all",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    packages=setuptools.find_packages(),
    python_requires=">=3.9.7",
    install_requires=install_requires
)
