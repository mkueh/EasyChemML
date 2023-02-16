from .AbstractHyperParamter import AbstractHyperParamter
import numpy as np


class FloatRange(AbstractHyperParamter):
    start: float
    stop: float
    step: float
    count: int

    def __init__(self, start: float = 0.0, stop: float = 0.0, step: float = 1, count: int = None):
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

        return list(np.arange(self.start, self.stop, self.step))
