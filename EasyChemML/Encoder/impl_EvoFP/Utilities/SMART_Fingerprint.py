import pickle
import random, hashlib, datetime

import lz4.frame, pickle

from EasyChemML.Utilities.CompressUtilities.lz4_compressor import lz4_compressor
from .SMART_Pattern import SMART_pattern
from typing import List
from bitarray import bitarray
from rdkit import Chem
import numpy as np


class SMART_Fingerprint(object):
    __slots__ = ['pattern', 'create_time']

    def __init__(self, pattern: List[SMART_pattern]):
        self.pattern = pattern
        self.create_time = datetime.datetime.now()

    def __len__(self):
        return len(self.pattern)

    def __getitem__(self, item):
        return self.pattern[item]

    def __setitem__(self, key, value):
        self.pattern[key] = value

    def __str__(self):
        st = ''
        st += 'Fingerprint has structure: \n'
        for p in self.pattern:
            st += str(p) + '\n'
        return st

    def __add__(self, other):
        if isinstance(other, SMART_Fingerprint):
            pattern_1 = self.pattern
            pattern_2 = other.pattern
            new_patternList = []
            new_patternList.extend(pattern_1)
            new_patternList.extend(pattern_2)
            return SMART_Fingerprint(new_patternList)
        else:
            raise Exception('can only add other SMART_Fingerprint')

    def __eq__(self, other):
        if isinstance(other, SMART_Fingerprint):
            if self.__equalList(self.pattern, other.pattern):
                return True
            return False

        return self == other

    def __equalList(self, a, b):
        tmp = [x for x in a if x in b]
        if len(tmp) == len(a):
            return True
        return False

    """
    prozentuale Übereinstimmung
    """

    def __cmp__(self, other):
        raise Exception('Not Implemented yet')
        pass  # TODO implementieren

    def id(self):
        self.pattern.sort(key=lambda x: str(x), reverse=True)

        tmp_str = ''
        for smart in self.pattern:
            tmp_str = tmp_str + str(smart)

        return hashlib.md5(tmp_str.encode()).hexdigest()

    def shuffelpattern(self):
        random.shuffle(self.pattern)

    def getFingerprintMOL(self, rdkitmols, bit_feature, asBitarray=False):
        if isinstance(rdkitmols, Chem.rdchem.Mol) or isinstance(rdkitmols, str):
            return self._getFingerprintMOL_MOL(rdkitmols, bit_feature, asBitarray)
        else:
            return self._getFingerprintMOL_LIST(rdkitmols, bit_feature, asBitarray)

    def _getFingerprintMOL_MOL(self, rdkitmols: Chem.Mol, bit_feature, asBitarray=False):
        if bit_feature:
            if asBitarray:
                fingerprint = bitarray(len(self.pattern))
            else:
                fingerprint = [None] * len(self.pattern)

            for j, pat in enumerate(self.pattern):
                match = pat.getMatchesWithMol(rdkitmols, bit_feature)
                fingerprint[j] = match

        else:
            fingerprint = []
            for j, pat in enumerate(self.pattern):
                matches = pat.getMatchesWithMol(rdkitmols, bit_feature)
                fingerprint.append(matches)
        return fingerprint

    def _getFingerprintMOL_LIST(self, rdkitmols: np.ndarray, bit_feature, asBitarray=False):
        if bit_feature:
            molCount = len(rdkitmols)
            if asBitarray:
                fingerprint = bitarray(len(self.pattern) * molCount)
            else:
                fingerprint = [None] * len(self.pattern) * molCount

            for i, mol in enumerate(rdkitmols):
                for j, pat in enumerate(self.pattern):
                    match = pat.getMatchesWithMol(mol, bit_feature)
                    fingerprint[i * len(self.pattern) + j] = match

        else:
            fingerprint = []
            for i, mol in enumerate(rdkitmols):
                for j, pat in enumerate(self.pattern):
                    matches = pat.getMatchesWithMol(mol, bit_feature)
                    fingerprint.append(matches)
        return fingerprint

    def getFingerpintof2DArr(self, arr, bit_feature, asBitarray=False):
        out = []
        for entry in arr:
            out.append(self.getFingerprintMOL(entry, bit_feature, asBitarray))
        return out

    def getPatternsAsString(self):
        strlist = []

        for p in self.pattern:
            strlist.append(str(p))

        return strlist

    @staticmethod
    def save(SMART_fingerprint: 'SMART_Fingerprint', path: str):
        compressor = lz4_compressor()
        compressor.compress_object_to_file(SMART_fingerprint, path)

    @staticmethod
    def load(path: str) -> 'SMART_Fingerprint':
        compressor = lz4_compressor()
        return compressor.decompress_object_from_file(path)
