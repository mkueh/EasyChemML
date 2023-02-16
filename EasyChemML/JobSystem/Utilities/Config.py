from EasyChemML.Model.AbstractModel.Abstract_Model import Abstract_Model

from typing import Dict, Type


class Config(object):
    algorithm: Type[Abstract_Model]
    algorithm_para: Dict

    def __init__(self, algorithm: Type[Abstract_Model], implicit_algorithm_para: Dict):
        self.algorithm = algorithm
        self.algorithm_para = implicit_algorithm_para
