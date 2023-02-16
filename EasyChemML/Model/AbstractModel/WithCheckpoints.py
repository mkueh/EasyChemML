from abc import ABC, abstractmethod
from typing import Dict, Any


class WithCheckpoints(ABC):

    @abstractmethod
    def getCurrentState(self):
        pass

    @abstractmethod
    def setCurrentState(self, saved_data: Dict[str, Any]):
        pass

    @abstractmethod
    def getCheckpointsAfterIterations(self) -> int:
        pass