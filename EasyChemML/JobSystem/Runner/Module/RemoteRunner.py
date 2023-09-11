import os
import tempfile
from enum import Enum
import datetime
import uuid
import paramiko
import time
import getpass

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Helper Functions
printProgressCount = 0
def progress(size, sent):
    global printProgressCount
    percentage = int((size/sent) * 100)
    if printProgressCount % 10 == 0 and printProgressCount != 100:
        print("\r", f"Transferred: {percentage}%", end="")
    if percentage == 100: print("\r", f"Transferred: 100%")
    printProgressCount += 1

def log(message):
    print('[RemoteRunner] ' + message)

def logRemote(message):
    print('Remote > ' + message, end="")

class SSHConfig(object):
    def __init__(self, hostname: str, port: int, username: str, password: str = ''):
        self.hostname: str = hostname
        self.port: int = port
        self.username: str = username
        self.password: str = password

class SlurmParams():
    # Available Partitions defined by https://confluence.uni-muenster.de/pages/viewpage.action?pageId=27755336
    class Partition(Enum):
        NORMAL = 'normal'
        LONG = 'long'
        EXPRESS = 'express'
        BIG_SMP = 'bigsmp'
        LARGE_SMP = 'largesmp'
        REQUEUE = 'requeue*'
        GPU_V100 = 'gpuv100'
        VIS_GPU = 'vis-gpu'
        VIS = 'vis'
        BROADWELL = 'broadwell'
        ZEN2 = 'zen2-128C-496G'
        GPU_2080 = 'gpu2080'
        GPU_EXPRESS = 'gpuexpress'
        GPU_TITANRTX = 'gputitanrtx'
        GPU_3090 = 'gpu3090'
        GPU_A100 = 'gpua100'
        GPU_HGX = 'gpuhgx'
        CPU = 'CPU'
    class MailType(Enum):
        NONE='NONE'
        BEGIN='BEGIN'
        END='END'
        FAIL='FAIL'
        REQUEUE='REQUEUE'
        ALL='ALL'
        INVALID_DEPEND='INVALID_DEPEND'
        STAGE_OUT='STAGE_OUT'
        TIME_LIMIT='TIME_LIMIT'
        TIME_LIMIT_90='TIME_LIMIT_90'
        TIME_LIMIT_80='TIME_LIMIT_80'
        TIME_LIMIT_50='TIME_LIMIT_50'
        ARRAY_TASKS='ARRAY_TASKS'

class SlurmDirectives():
    def __init__(
            self,
            nodes: str = "",
            ntasks_per_node: str = "",
            cpus_per_task: str = "",
            partition: SlurmParams.Partition | str = "",
            time: str = "",
            mail_type: SlurmParams.MailType | str = "",
            mail_user: str = "",
            output: str = "",
            error: str = "",
            gres: str = "",
            mem: str = "",
    ):
        self.nodes: str = nodes
        self.ntasks_per_node: str = ntasks_per_node
        self.cpus_per_task: str =  cpus_per_task
        self.partition: SlurmParams.Partition | str = partition
        self.time: str = time
        self.mail_type: SlurmParams.MailType | str = mail_type
        self.mail_user: str = mail_user
        self.output: str = output
        self.error: str = error
        self.gres: str = gres
        self.mem: str = mem

    def returnDefinedDirectives(self) -> dict:
        defined_directives = {}
        for attribute_name in self.__dict__:
            attribute_value = self.__dict__[attribute_name]
            if attribute_value:
                defined_directives[attribute_name] = \
                    attribute_value.value if isinstance(attribute_value, Enum) else attribute_value
        return defined_directives

class RemoteRunner():
    def __init__(
            self,
            ssh_config: SSHConfig,
            base_dir: str,
    ):

        print(f'''{bcolors.OKBLUE}
    ______________  _____       ____                       __       ____                             
   / ____/ ____/  |/  / /      / __ \___  ____ ___  ____  / /____  / __ \__  ______  ____  ___  _____
  / __/ / /   / /|_/ / /      / /_/ / _ \/ __ `__ \/ __ \/ __/ _ \/ /_/ / / / / __ \/ __ \/ _ \/ ___/
 / /___/ /___/ /  / / /___   / _, _/  __/ / / / / / /_/ / /_/  __/ _, _/ /_/ / / / / / / /  __/ /    
/_____/\____/_/  /_/_____/  /_/ |_|\___/_/ /_/ /_/\____/\__/\___/_/ |_|\__,_/_/ /_/_/ /_/\___/_/                 
        {bcolors.ENDC}''')

        self.ssh_config: SSHConfig = ssh_config
        self.base_dir: str = base_dir

        self.resources_dirname: str = 'resources'
        self.exit_msg: str = '[ECML RemoteRunner] Exit 1'

        # Initialize SSH Client
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Attempt to connect to Remote
        self._connect()

        # Establish SFTP Session and create or enter base dir
        self.sftp = self.ssh.open_sftp()
        self._enterOrCreateDir(self.base_dir, self.sftp)

    def _connect(self):
        try:
            if not self.ssh_config.password:
                self.ssh_config.password = getpass.getpass('[RemoteRunner] Please enter your SSH user password: ')

            log(f'Attempting to connect to "{self.ssh_config.hostname}:{self.ssh_config.port}"...')
            self.ssh.connect(
                hostname=self.ssh_config.hostname,
                port=self.ssh_config.port,
                username=self.ssh_config.username,
                password=self.ssh_config.password,
                timeout=10
            )
        except Exception as e:
            log(f'Could not connect with "{self.ssh_config.hostname}:{self.ssh_config.port}". Error: "{e}"')
            quit()

    def _enterOrCreateDir(self, path, sftp):
        # Try to enter the directory, if it does not exist, create new one
        try:
            sftp.chdir(path)
        except IOError as e:
            log(f'Creating new directory under "{path}"')
            sftp.mkdir(path)
            sftp.chdir(path)

    def run(
            self,
            job_name: str,
            slurm_directives: SlurmDirectives,
            container_path: str,
            script_path: str,
            resource_paths: list[str] = None,
            slurm_modules: list[str] = None,
            attach: bool = False,
    ):
        # Generate unique execution identifier used for data storage and further referencing
        execution_id = f'{datetime.datetime.now().strftime("%d-%m-%Y_%H%M%S")}#{uuid.uuid4().hex}'
        execution_sftp = self.ssh.open_sftp()

        log(f'Started execution with id: {execution_id}\n=======================================')

        with tempfile.TemporaryDirectory('-ecml_remoterunner') as tmpdirname:
            log(f'Created Temporary Directory: "{tmpdirname}"')

            slurm_script_path = self._createSlurmFile(
                tmpdirname,
                self._prepareSlurmScript(
                    slurm_directives=slurm_directives,
                    container_path=container_path,
                    slurm_modules=slurm_modules,
                    job_name=job_name
                )
            )

            self._transferFiles(
                script_path=script_path,
                resource_paths=resource_paths,
                dir_name=execution_id,
                slurm_script_path=slurm_script_path,
                sftp=execution_sftp
            )

        # Submit job to slurm queue
        command = f'cd {os.path.join(self.base_dir, execution_id)}; sbatch --parsable job.sh'
        log(f'Executing command: {command}')
        stdin, stdout, stderr = self.ssh.exec_command(command)

        # Check if errors occured
        error = stderr.read().decode()
        if error:
            raise Exception(error)

        # Parse Job Number
        job_nr = ""
        for line in iter(stdout.readline, ""):
                job_nr = line.split(";")[0].strip()

        log(f'{bcolors.BOLD}{bcolors.OKGREEN}Job submitted, Job Nr is: {job_nr}{bcolors.ENDC}')

        log(f'To download directory, use:  {bcolors.BOLD+bcolors.OKCYAN}scp -P {self.ssh_config.port} -r {self.ssh_config.username}@{self.ssh_config.hostname}:{os.path.join(self.base_dir, execution_id)} <local directory>{bcolors.ENDC}')

        if not attach:
            return f'Execution {execution_id} with job ID {job_nr} successful'

        # Poll if output file exists
        job_started = False
        log('Waiting for job output file creation.')
        while not job_started:
            command = f'cd {os.path.join(self.base_dir, execution_id)}; [ -f "slurm-{job_nr}.out" ] && echo "File exists" || echo "File does not exist"'
            stdin, stdout, stderr = self.ssh.exec_command(command)
            for line in iter(stdout.readline, ""):
                if line.strip() == 'File exists':
                    job_started = True
            if not job_started: time.sleep(2)

        log('Job output file created. Capturing job stdout:\n===============')

        # Attach to job output
        command = f'cd {os.path.join(self.base_dir, execution_id)}; tail -f -n +1 slurm-{job_nr}.out'
        stdin, stdout, stderr = self.ssh.exec_command(command, get_pty=False)

        for line in iter(stdout.readline, ""):
            if line.strip() == self.exit_msg:
                log('Job finished, terminating...')
                quit()
            logRemote(line)

    def _transferFiles(self, sftp, dir_name: str, script_path: str, slurm_script_path: str, resource_paths: list[str] = None):
        self._enterOrCreateDir(os.path.join(self.base_dir, dir_name), sftp)

        log('Transferring execution script:')
        sftp.put(script_path, 'script.py', callback=progress)
        log ('Transferring slurm script:')
        sftp.put(slurm_script_path, 'job.sh', callback=progress)

        if resource_paths is not None:
            self._enterOrCreateDir(os.path.join(self.base_dir, dir_name, self.resources_dirname), sftp)
            for path in resource_paths:
                log(f'Transferring resource "{path}":')
                sftp.put(path, os.path.basename(path), callback=progress)

    def _prepareSlurmScript(
            self,
            slurm_directives: SlurmDirectives,
            container_path: str,
            job_name: str,
            slurm_modules: list[str] = None,
    ) -> list[str]:
        script_lines = ['#!/bin/bash -l']

        directives_list = slurm_directives.returnDefinedDirectives()

        directives_list['job_name'] = job_name

        for key, value in directives_list.items():
            script_lines.append(f'#SBATCH --{key.replace("_", "-")}={value}')

        if slurm_modules is not None:
            for module in slurm_modules:
                script_lines.append(f'module load {module}')

        script_lines.append(f'singularity exec {container_path} python script.py')

        script_lines.append(f'echo "{self.exit_msg}"')

        return script_lines

    def _createSlurmFile(self, dir: str, script_lines: list[str]) -> str:
        filename = 'job.sh'
        fullpath = os.path.join(dir, filename)

        with open(fullpath, 'w') as file:
            for line in script_lines:
                file.write(line + '\n')

        log(f'Saved Slurm script under {fullpath}')

        return fullpath

    def __del__(self):
        self.ssh.close()
        print('[RemoteRunner]: Terminated')
