import threading
import time

from RemoteRunner import SSHConfig, RemoteRunner, SlurmDirectives, SlurmParams

test_single = True
test_sequential = False
test_multithread = False

ssh_config = SSHConfig(
    'catalyst.uni-muenster.de',
    60001,
    'gbircan',
)

slurm_directives = SlurmDirectives(
    nodes='1',
    mem='100M',
    ntasks_per_node='1',
    partition=SlurmParams.Partition.CPU,
    mail_type=SlurmParams.MailType.ALL,
    mail_user='gbircan@uni-muenster.de'
)

rr = RemoteRunner(
    ssh_config=ssh_config,
    base_dir='/home/gbircan/remoterunner'
)

# Single Test
if test_single:
    begin = time.time()

    rr.run(
        job_name='remoterunner-test',
        slurm_directives=slurm_directives,
        slurm_modules=['GCC'],
        container_path='/home/gbircan/singularity/easyChemMLGithub.sif',
        script_path='train.py',
        resource_paths=['../_DATASETS/Dreher_and_Doyle_input_data.xlsx'],
        attach=False,
        download=False
        # resource_paths=['../_DATASETS/Tox_Karmaus.xlsx', '100mbfile'],
    )

    end = time.time()
    print(f'Elapsed time: {end - begin}s')

# Sequential Test
if test_sequential:
    configurations = [
        {'resource_paths': ['../_DATASETS/Dreher_and_Doyle_input_data.xlsx'], 'script_path': 'train.py'},
        {'resource_paths': ['../_DATASETS/Tox_Karmaus.xlsx'], 'script_path': 'train_tox.py'},
        {'resource_paths': ['../_DATASETS/Dreher_and_Doyle_input_data.xlsx'], 'script_path': 'train_testsize20.py'},
        {'resource_paths': ['../_DATASETS/Tox_Karmaus.xlsx'], 'script_path': 'train_tox_testsize20.py'}
    ]

    startFor = time.time()
    for index, config in enumerate(configurations):
        start = time.time()
        rr.run(
            job_name=f'rr_test_{index}',
            slurm_directives=slurm_directives,
            container_path='/home/gbircan/singularity/easyChemMLGithub.sif',
            script_path=config['script_path'],
            resource_paths=config['resource_paths']
        )
        end = time.time()
        print(f'[{time.time()}] Execution {index} terminated after {end - start}s')
    endFor = time.time()
    print('========')
    print(f'[{time.time()}] Complete Execution terminated after {endFor - startFor}s')

# Multithread Test
if test_multithread:
    configurations = [
        {'resource_paths': ['../_DATASETS/Dreher_and_Doyle_input_data.xlsx'], 'script_path': 'train.py'},
        {'resource_paths': ['../_DATASETS/Tox_Karmaus.xlsx'], 'script_path': 'train_tox.py'},
        {'resource_paths': ['../_DATASETS/Dreher_and_Doyle_input_data.xlsx'], 'script_path': 'train_testsize20.py'},
        {'resource_paths': ['../_DATASETS/Tox_Karmaus.xlsx'], 'script_path': 'train_tox_testsize20.py'},
        {'resource_paths': ['../_DATASETS/Dreher_and_Doyle_input_data.xlsx'], 'script_path': 'train.py'},
        {'resource_paths': ['../_DATASETS/Tox_Karmaus.xlsx'], 'script_path': 'train_tox.py'},
        {'resource_paths': ['../_DATASETS/Dreher_and_Doyle_input_data.xlsx'], 'script_path': 'train_testsize20.py'},
        {'resource_paths': ['../_DATASETS/Tox_Karmaus.xlsx'], 'script_path': 'train_tox_testsize20.py'},
    ]

    class myThread(threading.Thread):
        def __init__(self, index, config):
            threading.Thread.__init__(self)
            self.index = index
            self.config = config

        def run(self):
            rr = RemoteRunner(
                ssh_config=ssh_config,
                base_dir='/home/gbircan/remoterunner'
            )
            start = time.time()
            rr.run(
                job_name=f'rr_test_{self.index}',
                slurm_directives=slurm_directives,
                container_path='/home/gbircan/singularity/easyChemMLGithub.sif',
                script_path=self.config['script_path'],
                resource_paths=self.config['resource_paths']
            )
            end = time.time()
            print(f'[{time.time()}] [Thread {self.index}] Terminated in {end - start}s')


    for index, config in enumerate(configurations):
        thread = myThread(index, config)
        thread.start()