from .AbstractHyperParamter import AbstractHyperParamter
import numpy as np


class IntRange(AbstractHyperParamter):
    start: int
    stop: int
    step: int
    count: int

    def __init__(self, start: int = 0, stop: int = 0, step: int = 1, count: int = None):
        super().__init__()
        self.start = start
        self.stop = stop
        self.step = step
        self.count = count

    def toExplicitParameters(self):
        arr = []

        if not self.count is None:
            raise Exception(
                'It is not possible to find an explicit representation for this hyperparameter if count is not set.')

        return list(np.arange(start=self.start, stop=self.stop, step=self.step))
