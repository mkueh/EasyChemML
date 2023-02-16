from Utilities.Dataset import Dataset
from Preprocessing.Preprocessing import typs
from abc import ABC, abstractmethod

class Abstract_Preprocessing(ABC):

    def __init__(self, param={}):
        pass

    @staticmethod
    @abstractmethod
    def convert(self, dataset: Dataset, n_jobs: int, typ: typs):
        raise Exception('not implemented')

    @staticmethod
    @abstractmethod
    def getItemname():
        raise Exception('not implemented')