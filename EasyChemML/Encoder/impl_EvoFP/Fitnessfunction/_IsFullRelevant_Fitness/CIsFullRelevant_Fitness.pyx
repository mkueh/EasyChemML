# distutils: language=c++
# cython: language_level=3

from EasyChemML.Utilities.Dataset import Dataset
from typing import List
from rdkit import Chem
from bitarray import bitarray
from EasyChemML.Encoder.impl_EvoFP.Utilities.SMART_Pattern import SMART_pattern
from EasyChemML.Encoder.impl_EvoFP.Utilities.SMART_Fingerprint import SMART_Fingerprint
from libcpp.vector cimport vector
from libcpp cimport bool
import collections, math, numpy as np


cdef class CIsFullRelevant_Fitness():
    cdef bool __bitfeature

    def __init__(self, bool bitfeature):
        self.__bitfeature = bitfeature

    def calc_fitness(self, one: SMART_Fingerprint, train_dataset: List[Chem.rdchem.Mol], n_jobs=1):
        collisionCounter = self.__calc_collision(one, train_dataset, self.__bitfeature)

        metric = float(collisionCounter) / math.pow(len(train_dataset)*len(train_dataset[0]),2)

        return metric

    def __calc_collision(self, one: SMART_Fingerprint, train_dataset: List[Chem.rdchem.Mol], bitfeature):
        collisionCounter = 0
        train_dataset_length = len(train_dataset)

        for s_index,s_mol in enumerate(train_dataset):
            for compare_index in range(s_index+1, train_dataset_length):
                for col_index in range(len(train_dataset[0])):
                    fp_s_index = one.getFingerprintMOL(train_dataset[s_index][col_index], bitfeature, True)
                    fp_compare_index = one.getFingerprintMOL(train_dataset[compare_index][col_index], bitfeature, True)

                    for fp_index in range(len(fp_s_index)):
                        if fp_s_index[fp_index] == fp_compare_index[fp_index]:
                            collisionCounter += 1

        return (collisionCounter * 2) + (len(train_dataset)*len(train_dataset[0]))

