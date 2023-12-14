from abc import abstractmethod, ABC
from typing import List, Union

from EasyChemML.Encoder.impl_EvoFP.Utilities.SMART_Fingerprint import SMART_Fingerprint
from EasyChemML.Utilities.Dataset import Dataset
from EasyChemML.Utilities.SharedDataset import SharedDataset


class Abstract_Fitnessfunction(ABC):

    @abstractmethod
    def get_datasets(self) -> Union[SharedDataset, List[SharedDataset]]:
        pass

    @abstractmethod
    def calc_fitness(self, one: SMART_Fingerprint, working_path, n_jobs=1) -> (
            float, int):
        pass
