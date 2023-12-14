from EasyChemML.Encoder.impl_EvoFP.Utilities.SMART_Fingerprint import SMART_Fingerprint
from EasyChemML.Encoder.impl_EvoFP.Fitnessfunction._IsFullRelevant_Fitness.CIsFullRelevant_Fitness import CIsFullRelevant_Fitness
from EasyChemML.Utilities.Dataset import Dataset
from EasyChemML.Utilities.ParallelUtilities.Shared_PythonList import Shared_PythonList
from typing import List
import numpy as np

from EasyChemML.Utilities.SharedDataset import SharedDataset

"""Gives a Score for how relevant is the Pattern in the whole Fingerprint"""


class IsFullRelevant_Fitness():

    __bitfeature = 0
    __cclass:CIsFullRelevant_Fitness

    def __init__(self, train_dataset:Dataset, feature_typ, MManager, **kwargs):
        self.__bitfeature = feature_typ
        self.__cclass = CIsFullRelevant_Fitness(feature_typ)
        self.__identical_ratio = kwargs['identical_ratio']

    def get_datasets(self) -> SharedDataset:
        return None

    def calc_fitness(self, one: SMART_Fingerprint, train_dataset:SharedDataset,
                     working_path, n_jobs=1) -> (float, List[float]):
        #t = time.process_time()
        #tmp = self.__cclass.calc_fitness(one, train_dataset.getFeature(), n_jobs)
        ratio_collision = self.__cclass.calc_fitness(one, n_jobs)
        return (1 - abs(ratio_collision-self.__identical_ratio), 0)

        #elapsed_time = time.process_time() - t
        #print('fitness calc needs: ' + str(elapsed_time) + 's')




"""
C1
A1  A2  A3
0   0   1   MOL1
0   1   0   MOL2
0   1   1   MOL3
1   0   0   MOL4
1   0   1   MOL5
1   1   0   MOL6
1   1   1   MOL7
1   1   0   MOLx

C2
A1  A2  A3
0   0   1   MOL1
0   1   0   MOL2
0   1   1   MOL3
1   0   0   MOL4
1   0   1   MOL5
1   1   0   MOL6
1   1   1   MOL7
1   1   0   MOLx

C3
A1  A2  A3
0   0   1   MOL1
0   1   0   MOL2
0   1   1   MOL3
1   0   0   MOL4
1   0   1   MOL5
1   1   0   MOL6
1   1   1   MOL7
1   1   0   MOLx

4/8 = 0.5
3/8 = 0.375
0.4375

1/2 1
1/2 2
0/2 3
0/2 4
1/2 5
1/2 6
2/2 7
1/2 8
"""