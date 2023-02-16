from pathlib import Path

from setuptools import setup
import shutil, os

from setuptools_rust import Binding, RustExtension

setup(
    name="pyRustBatchsystem",
    version="1.0",
    packages=['pyRustBatchsystem'],
    rust_extensions=[
        RustExtension(
            "pyRustBatchsystem.pyRustBatchsystem",
            binding=Binding.PyO3,
            debug=False)],
    zip_safe=False,

)

src_path_glob = Path(os.path.dirname(__file__)).joinpath('build')
src_paths = list(src_path_glob.glob('**/*pyd'))
src_paths.extend(list(src_path_glob.glob('**/*so')))
src_paths.extend(list(src_path_glob.glob('**/*py')))

dst_path = Path(os.path.dirname(__file__)).joinpath('pyRustBatchsystem')

if len(src_paths) == 0:
    raise Exception('No compiled Rust-Extension found in setup directory')
else:
    print('found Rust-Extension Files')

for src_path in src_paths:
    print(f'copy from {src_path} to {dst_path}')
    shutil.copy(src_path, dst_path)


if os.path.exists('pyRustBatchsystem.egg-info'):
    shutil.rmtree('pyRustBatchsystem.egg-info')

if os.path.exists('build'):
    shutil.rmtree('build')

if os.path.exists('target'):
    shutil.rmtree('target')
