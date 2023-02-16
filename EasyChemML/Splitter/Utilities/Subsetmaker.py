from typing import Union
#from numba import jit
import numpy as np


class Subsetmaker:

    @staticmethod
    #@jit(nopython=True)  numpy 1.22 is not supported
    def generateInverseSubsetWithRange(baseSubset: Union[list, np.ndarray], rangeEnd: int) -> np.ndarray:
        length_ofInverseSubset = rangeEnd - len(baseSubset)
        inverseSubset = np.zeros((length_ofInverseSubset,), np.int64)

        baseSubset_pos = 0
        inverseSubset_pos = 0
        for current_val in range(rangeEnd):
            if len(baseSubset) > baseSubset_pos and current_val == baseSubset[baseSubset_pos]:
                baseSubset_pos += 1
            else:
                inverseSubset[inverseSubset_pos] = current_val
                inverseSubset_pos += 1

        return inverseSubset
