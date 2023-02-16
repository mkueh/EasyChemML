from abc import ABC, abstractmethod


class WithEpochs(ABC):

    @abstractmethod
    def getEpochs(self) -> int:
        pass
