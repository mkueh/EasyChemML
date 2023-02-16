from typing import List, Dict, Any

from EasyChemML.JobSystem.CheckpointSystem.Checkpoint import Checkpoint


class CheckpointGroup:
    checkpoints: List[Checkpoint]
    maximum_checkpoints: int
    jobname: str

    def __init__(self, jobname: str, maximum_checkpoints: int = 3):
        """

        Args:
            jobname: jobname of the grouped checkpoints
            maximum_checkpoints: Number of checkpoints that are saved. Older checkpoints are overwritten when new ones are added. If the value is -1, no checkpoints are deleted.
        """
        self.jobname = jobname
        self.checkpoints = []
        self.maximum_checkpoints = maximum_checkpoints

    def getNewest(self) -> Checkpoint:
        return self.checkpoints[-1]

    def sort_byTime(self):
        self.checkpoints.sort(key=lambda x: x.timestamp)

    def checkForCheckpoints(self) -> bool:
        if len(self.checkpoints) > 0:
            return True
        else:
            return False

    def append(self, checkpoint: Checkpoint) -> bool:
        for check in self.checkpoints:
            if check == checkpoint:
                raise Exception('Checkpoints needs unique filenames')

        if len(self.checkpoints) >= self.maximum_checkpoints and not self.maximum_checkpoints == -1:
            self.checkpoints.sort(key=lambda x: x.timestamp)
            removed = self.checkpoints.pop(0)
            self._removeCheckpoint(removed)

        self.checkpoints.append(checkpoint)
        return True

    def _removeCheckpoint(self, checkpoint: Checkpoint):
        checkpoint.remove()

    def getstate(self):
        state = {}
        state['maximum_checkpoints'] = self.maximum_checkpoints
        state['jobname'] = self.jobname

        checkpoints = []
        for checkpoint in self.checkpoints:
            checkpoints.append(checkpoint.getstate())

        state['checkpoints'] = checkpoints
        return state

    def setstate(self, state:Dict[str, Any]):
        self.maximum_checkpoints = state['maximum_checkpoints']
        self.jobname = state['jobname']

        for checkpoint in state['checkpoints']:
            c = Checkpoint('','')
            c.setstate(checkpoint)
            self.checkpoints.append(c)
