from abc import ABC, abstractmethod

class WithBatches(ABC):

    @abstractmethod
    def getBatchsize(self) -> int:
        pass

