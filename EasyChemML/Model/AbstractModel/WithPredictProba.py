from abc import ABC, abstractmethod
from typing import Dict

class WithPredictProba(ABC):

    @staticmethod
    @abstractmethod
    def hasPredicte_proba() -> bool:
        return True
