import math
import os
from typing import Dict, Any, TYPE_CHECKING, Tuple

from EasyChemML.Utilities.FormatUtilities.HjsonLoader import HjsonLoader
from EasyChemML.JobSystem.CheckpointSystem.Checkpoint import Checkpoint
from EasyChemML.JobSystem.CheckpointSystem.CheckpointGroup import CheckpointGroup
from EasyChemML.Utilities.CompressUtilities.lz4_compressor import lz4_compressor

if TYPE_CHECKING:
    from EasyChemML.Environment import Environment


class CheckpointSystem:
    __saveFile: str = 'CheckpointSystemSTATE.hjson'
    __lz4: lz4_compressor = None
    __extention_data: str = '.CPDATA'
    __extention_meta: str = '.CPMETA'
    __env: 'Environment' = None

    checkpointGroups: Dict[str, CheckpointGroup] = {}

    def __init__(self, env: 'Environment'):
        self.__env: 'Environment' = env
        self.__lz4 = lz4_compressor()
        self.checkpointGroups: Dict[str, CheckpointGroup] = {}
        self.__initCheckpointSystem()

    def saveCheckpointSystem(self):
        state = {}
        state['extention_data'] = self.__extention_data
        state['extention_meta'] = self.__extention_meta

        groups = {}
        for group_name in self.checkpointGroups:
            group = self.checkpointGroups[group_name]
            groups[group.jobname] = group.getstate()

        state['checkpointgroups'] = groups

        HjsonLoader.dump_Hjson(state, os.path.join(self.__env.CHECKPOINT_path, self.__saveFile))

    def updateCheckpointGroup(self, jobName:str, maximum_checkpoints:int = 3):
        """
        update or create a Checkpoint group

        Args:
            jobName: Name of the checkpointgroup. If the group is to be used by the job runner, the name must be the same as the job name.
            maximum_checkpoints: Maximum number of checkpoints that are stored persistently

        """
        if jobName not in self.checkpointGroups:
            self.checkpointGroups[jobName] = CheckpointGroup(jobName, maximum_checkpoints)
        else:
            self.checkpointGroups[jobName].maximum_checkpoints = maximum_checkpoints

        self.saveCheckpointSystem()

    def createCheckpoint(self, saved_data: Dict[str, Any], addMetaData: Dict[str, Any], Filename: str,
                         jobName: str):
        if jobName not in self.checkpointGroups:
            self.checkpointGroups[jobName] = CheckpointGroup(jobName)

        data_path = os.path.join(self.__env.CHECKPOINT_path, Filename + self.__extention_data)
        meta_path = os.path.join(self.__env.CHECKPOINT_path, Filename + self.__extention_meta)

        self.__lz4.compress_object_to_file(saved_data, data_path)
        self.__lz4.compress_object_to_file(addMetaData, meta_path)

        self.checkpointGroups[jobName].append(Checkpoint(meta_path, data_path))
        self.saveCheckpointSystem()

    def loadDataOfCheckpoint(self, checkpoint: Checkpoint) -> Tuple[Any, Any]:
        meta_data = self.__lz4.decompress_object_from_file(checkpoint.meta_path)
        data_data = self.__lz4.decompress_object_from_file(checkpoint.data_path)
        return meta_data, data_data

    def checkForCheckpoints(self, jobName: str) -> bool:
        if jobName in self.checkpointGroups:
            return self.checkpointGroups[jobName].checkForCheckpoints()
        return False

    def loadLastCheckpoint(self, jobName: str) -> Checkpoint:
        if self.checkForCheckpoints(jobName):
            return self.checkpointGroups[jobName].getNewest()
        return None

    def __initCheckpointSystem(self):
        if os.path.exists(os.path.join(self.__env.CHECKPOINT_path, self.__saveFile)):
            loaded_state: Dict[str, Any] = HjsonLoader.load_Hjson(os.path.join(self.__env.CHECKPOINT_path, self.__saveFile))

            self.__extention_meta = loaded_state['extention_meta']
            self.__extention_data = loaded_state['extention_data']

            for group_name in loaded_state['checkpointgroups']:
                group = loaded_state['checkpointgroups'][group_name]
                self.checkpointGroups[group_name] = CheckpointGroup('', -1)
                self.checkpointGroups[group_name].setstate(group)

    @staticmethod
    def checkCheckpointNeeded(absoluteIteration: int, checkpoints_afterIterations: int):
        if checkpoints_afterIterations > 0:
            relativIteration: int = absoluteIteration % checkpoints_afterIterations

            if relativIteration == 0:
                return True
            else:
                return False
        else:
            return False

    @staticmethod
    def calcBatchcountPerEpoch(batchSize:int, arrSize:int):
        return math.ceil(arrSize/batchSize)
