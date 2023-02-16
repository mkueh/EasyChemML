# distutils: language=c++
# cython: language_level=3

import copy
from enum import Enum

from rdkit import Chem, DataStructs
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import MACCSkeys, AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.DataStructs.cDataStructs import ExplicitBitVect

from typing import List, TYPE_CHECKING, Dict
import numpy as np

from EasyChemML.Utilities.DataUtilities.BatchDatatyp import BatchDatatypClass

if TYPE_CHECKING:
    from EasyChemML.Encoder.FingerprintEncoder import FingerprintTyp


class FingerprintGenerator_Mode(Enum):
    PLAIN_FP = 0
    NONZERO_INDICES = 1


class FingerprintGenerator:
    __mode: FingerprintGenerator_Mode
    __default_NoneToke:int
    __offsetForNonZeroIndices:int

    def __init__(self, mode: FingerprintGenerator_Mode = FingerprintGenerator_Mode.PLAIN_FP, default_NoneToke:int = -1, offsetForNonZeroIndices:int = 0):
        self.__mode = mode
        self.__default_NoneToke = default_NoneToke
        self.__offsetForNonZeroIndices = offsetForNonZeroIndices

    def getMinimumDtyp(self, fingerprints: List['FingerprintTyp']):
        if self.__mode == FingerprintGenerator_Mode.NONZERO_INDICES:
            return BatchDatatypClass.NUMPY_INT32

        minimum_lvl = 0
        for fp in fingerprints:
            if minimum_lvl < BatchDatatypClass.get_dtype_lvl(fp[1]):
                minimum_lvl = BatchDatatypClass.get_dtype_lvl(fp[1])

        return BatchDatatypClass.get_by_lvl(minimum_lvl)

    def getFullShape(self, fingerprints: List['FingerprintTyp'], fingerprintArgs: List[Dict]):
        shapes = []
        for index_fpArray, fp in enumerate(fingerprints):
            if fp[2] == 'var':
                shapes.append(fingerprintArgs[index_fpArray]['length'])
            else:
                shapes.append(int(fp[2]))
        return shapes

    def generateArrOfFingerprints(self, one_row, fingerprints: List['FingerprintTyp'], fingerprintArgs: List[Dict]):
        minimum_dtype = self.getMinimumDtyp(fingerprints)
        col_shape = np.sum(self.getFullShape(fingerprints, fingerprintArgs))
        outputList = np.zeros(shape=(len(one_row), col_shape), dtype=minimum_dtype.to_numpy())

        for index_column, column in enumerate(one_row):
            tmp_args = copy.deepcopy(fingerprintArgs)
            col_tmp = []

            for i, fp_name in enumerate(fingerprints):
                if 'length' in tmp_args[i]:
                    length = tmp_args[i]['length']
                    del tmp_args[i]['length']
                else:
                    if not fp_name[0] == 'maccs':
                        raise Exception('Length not setten in args nr. ' + str(i) + f' | fp_name: {fingerprints[i]}')

                if fp_name[0] == 'rdkit':
                    tmp = self.__generateFingerprints_RDKit(column, length, tmp_args[i], minimum_dtype)
                elif fp_name[0] == 'morgan_circular':
                    tmp = self.__generateFingerprints_Morgan_Circular(column, length, tmp_args[i], minimum_dtype)
                elif fp_name[0] == 'morgan_circular_count':
                    tmp = self.__generateFingerprints_Morgan_Circular_Count(column, length, tmp_args[i], minimum_dtype)
                elif fp_name[0] == 'avalon':
                    tmp = self.__generateFingerprints_Avalon(column, length, tmp_args[i], minimum_dtype)
                elif fp_name[0] == 'layerdfingerprint':
                    tmp = self.__generateFingerprints_LayerdFingerprint(column, length, tmp_args[i], minimum_dtype)
                elif fp_name[0] == 'maccs':
                    tmp = self.__generateFingerprints_MACCS_keys(column, tmp_args[i], minimum_dtype)
                elif fp_name[0] == 'atom_pairs':
                    tmp = self.__generateFingerprints_Atom_Pairs(column, length, tmp_args[i], minimum_dtype)
                elif fp_name[0] == 'topological_torsions':
                    tmp = self.__generateFingerprints_Topological_Torsions(column, length, tmp_args[i], minimum_dtype)
                else:
                    raise Exception(f'The selected fingerprintname is not supported {fp_name[0]}')

                col_tmp.append(tmp)

            concat_arr = np.concatenate(col_tmp, axis=None)

            if not self.__offsetForNonZeroIndices == 0 and self.__mode == FingerprintGenerator_Mode.NONZERO_INDICES:
                concat_arr = concat_arr + self.__offsetForNonZeroIndices

            filled_arr = np.concatenate((concat_arr, [self.__default_NoneToke] * (col_shape - len(concat_arr))), axis=None)

            outputList[index_column] = filled_arr

        return outputList

    def __generate_Array(self, fp: ExplicitBitVect, minimum_dtype: BatchDatatypClass):
        arr = np.zeros((0,), dtype=minimum_dtype.to_numpy())
        DataStructs.ConvertToNumpyArray(fp, arr)

        if self.__mode == FingerprintGenerator_Mode.PLAIN_FP:
            return arr
        elif self.__mode == FingerprintGenerator_Mode.NONZERO_INDICES:
            return np.nonzero(arr)
        else:
            raise Exception(f'FingerprintGenerator_Mode {self.__mode} is unknown')

    def __getEmptyBitVector(self, length: int, minimum_dtype: BatchDatatypClass):
        arr = np.zeros((length,), dtype=minimum_dtype.to_numpy())
        return arr

    def __generateFingerprints_RDKit(self, data, length, args, minimum_dtype: BatchDatatypClass):
        if data == 'NA':
            return self.__getEmptyBitVector(length, minimum_dtype)
        fp = Chem.RDKFingerprint(mol=data, fpSize=length, **args)
        return self.__generate_Array(fp, minimum_dtype)

    def __generateFingerprints_Atom_Pairs(self, data, length, args, minimum_dtype: BatchDatatypClass):
        if data == 'NA':
            return self.__getEmptyBitVector(length, minimum_dtype)
        return self.__generate_Array(rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(data, nBits=length, **args),
                                     minimum_dtype)

    def __generateFingerprints_Topological_Torsions(self, data, length, args, minimum_dtype: BatchDatatypClass):
        if data == 'NA':
            return self.__getEmptyBitVector(length, minimum_dtype)
        return self.__generate_Array(rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect
                                     (data, nBits=length, **args), minimum_dtype)

    def __generateFingerprints_MACCS_keys(self, data, args, minimum_dtype: BatchDatatypClass):
        if data == 'NA':
            return self.__getEmptyBitVector(167, minimum_dtype)
        return self.__generate_Array(MACCSkeys.GenMACCSKeys(data, **args), minimum_dtype)

    def __generateFingerprints_Morgan_Circular(self, data, length, args, minimum_dtype: BatchDatatypClass):
        if data == 'NA':
            return self.__getEmptyBitVector(length, minimum_dtype)
        return self.__generate_Array(AllChem.GetMorganFingerprintAsBitVect(data, nBits=length, **args), minimum_dtype)

    def __generateFingerprints_Morgan_Circular_Count(self, data, length, args, minimum_dtype: BatchDatatypClass):
        if data == 'NA':
            return self.__getEmptyBitVector(length, minimum_dtype)
        return self.__generate_Array(rdMolDescriptors.GetHashedMorganFingerprint(data, nBits=length, **args),
                                     minimum_dtype)

    def __generateFingerprints_Avalon(self, data, length, args, minimum_dtype: BatchDatatypClass):
        if data == 'NA':
            return self.__getEmptyBitVector(length, minimum_dtype)
        return self.__generate_Array(pyAvalonTools.GetAvalonFP(data, nBits=length, **args), minimum_dtype)

    def __generateFingerprints_LayerdFingerprint(self, data, length, args, minimum_dtype: BatchDatatypClass):
        if data == 'NA':
            return self.__getEmptyBitVector(length, minimum_dtype)
        return self.__generate_Array(Chem.LayeredFingerprint(data, fpSize=length, **args), minimum_dtype)
