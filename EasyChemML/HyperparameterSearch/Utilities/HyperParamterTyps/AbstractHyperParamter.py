from abc import abstractmethod
from enum import Enum, auto
from typing import List
import numpy as np


class AbstractHyperParamter:

    def __init__(self):
        pass

    @abstractmethod
    def toExplicitParameters(self):
        pass
