import copy
import logging
import os
import shutil
import sys
import time
from pathlib import Path

from EasyChemML.JobSystem.CheckpointSystem.CheckpointSystem import CheckpointSystem


def check_relativePath(path: str) -> bool:
    return not os.path.isabs(path)


class Environment:
    TMP_path: str
    WORKING_path: str
    CHECKPOINT_path: str

    CheckpointSystem: CheckpointSystem

    def __init__(self, WORKING_path: str = None, TMP_path: str = None, CHECKPOINT_path: str = None):
        if WORKING_path is None:
            self.WORKING_path = self._generate_WORKING_path()
        else:
            self.WORKING_path = WORKING_path

        if TMP_path is None:
            self.TMP_path = self._generate_TMP_path()
        else:
            self.TMP_path = TMP_path

        if CHECKPOINT_path is None:
            self.CHECKPOINT_path = self._generate_CHECKPOINT_path()
        else:
            self.CHECKPOINT_path = CHECKPOINT_path

        # Make Paths Absolute
        if check_relativePath(self.WORKING_path):
            self.WORKING_path = str(Path(self.WORKING_path).resolve())

        if check_relativePath(self.TMP_path):
            self.TMP_path = str(Path(self.TMP_path).resolve())

        if check_relativePath(self.CHECKPOINT_path):
            self.CHECKPOINT_path = str(Path(self.CHECKPOINT_path).resolve())

        # Create Folder
        if not os.path.exists(self.WORKING_path):
            Path(self.WORKING_path).mkdir(parents=True)

        if not os.path.exists(self.TMP_path):
            Path(self.TMP_path).mkdir(parents=True)

        if not os.path.exists(self.CHECKPOINT_path):
            Path(self.CHECKPOINT_path).mkdir(parents=True)

        # Prepare Environment
        self._createLogging()
        self._changeCPUAffinity()
        self.CheckpointSystem = CheckpointSystem(self)

    def _changeCPUAffinity(self):
        if not os.name == 'nt':  # not possible for windows
            affinity_mask = range(os.cpu_count())
            os.sched_setaffinity(os.getpid(), affinity_mask)

    def clean(self):
        time.sleep(1)
        try:
            shutil.rmtree(self.TMP_path)
        except:
            print('Cleanup failed')

    def _createLogging(self):
        from importlib import reload
        reload(logging)

        logging.basicConfig(filename=os.path.join(self.WORKING_path, 'MainNode.log'), level=logging.INFO)
        logging.info(' ###################################################################################')
        logging.info('    ______           _______     _______ _    _ ______ __  __        __  __ _       ')
        logging.info('   |  ____|   /\    / ____\ \   / / ____| |  | |  ____|  \/  |      |  \/  | |      ')
        logging.info('   | |__     /  \  | (___  \ \_/ / |    | |__| | |__  | \  / |______| \  / | |      ')
        logging.info('   |  __|   / /\ \  \___ \  \   /| |    |  __  |  __| | |\/| |______| |\/| | |      ')
        logging.info('   | |____ / ____ \ ____) |  | | | |____| |  | | |____| |  | |      | |  | | |____  ')
        logging.info('   |______/_/    \_\_____/   |_|  \_____|_|  |_|______|_|  |_|      |_|  |_|______| ')
        logging.info('####################################################################################')

        print(' ###################################################################################')
        print('    ______           _______     _______ _    _ ______ __  __        __  __ _       ')
        print('   |  ____|   /\    / ____\ \   / / ____| |  | |  ____|  \/  |      |  \/  | |      ')
        print('   | |__     /  \  | (___  \ \_/ / |    | |__| | |__  | \  / |______| \  / | |      ')
        print('   |  __|   / /\ \  \___ \  \   /| |    |  __  |  __| | |\/| |______| |\/| | |      ')
        print('   | |____ / ____ \ ____) |  | | | |____| |  | | |____| |  | |      | |  | | |____  ')
        print('   |______/_/    \_\_____/   |_|  \_____|_|  |_|______|_|  |_|      |_|  |_|______| ')
        print('####################################################################################')

    def _generate_TMP_path(self):
        program_path, file = os.path.split(sys.argv[0])
        tmp_path = os.path.join(program_path, 'TMP')

        if os.path.exists(tmp_path):
            print('remove TMP')
            try:
                shutil.rmtree(tmp_path)
            except:
                print('delete tmp folder failed')
            time.sleep(0.1)

        try:
            os.mkdir(tmp_path)
        except:
            print('tmp folder already exists')
        return tmp_path

    def _generate_WORKING_path(self):
        program_path, file = os.path.split(sys.argv[0])
        return os.path.join(program_path)

    def _generate_CHECKPOINT_path(self):
        program_path, file = os.path.split(sys.argv[0])
        checkpoint_path = os.path.join(program_path, 'Checkpoints')
        return checkpoint_path


class EasyProjectEnvironment(Environment):

    def __init__(self, projectFolder_path: str):
        projectFolder_path = Path(projectFolder_path);
        if not projectFolder_path.is_absolute():
            projectFolder_path = projectFolder_path.resolve()

        workPath = copy.copy(projectFolder_path)
        tmpPath = copy.copy(workPath).joinpath('TMP')
        checkpointPath = copy.copy(workPath).joinpath('CHECKPOINT')

        super().__init__(str(workPath), str(tmpPath), str(checkpointPath))
