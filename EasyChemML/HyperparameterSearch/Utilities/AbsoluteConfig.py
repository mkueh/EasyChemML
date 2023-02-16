from typing import Type, Dict

from EasyChemML.JobSystem.Utilities.Config import Config
from EasyChemML.Model.AbstractModel.Abstract_Model import Abstract_Model


class AbsoluteConfig():

    algorithm: Type[Abstract_Model]
    explicit_algorithm_para: Dict

    def __init__(self, algorithm: Type[Abstract_Model], explicit_algorithm_para: Dict):
        self.algorithm = algorithm
        self.explicit_algorithm_para = explicit_algorithm_para

    def toConfig(self) -> Config:
        return Config(self.algorithm, self.explicit_algorithm_para)