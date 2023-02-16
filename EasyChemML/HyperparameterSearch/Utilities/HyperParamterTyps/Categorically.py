from .AbstractHyperParamter import AbstractHyperParamter
from typing import List


class Categorically(AbstractHyperParamter):
    items: List

    def __init__(self, items: List):
        super().__init__()
        self.items = items

    def toExplicitParameters(self):
        return self.items
