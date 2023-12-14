from rdkit import Chem
import hashlib, datetime
from typing import Dict

from EasyChemML.Encoder.impl_EvoFP.EVOFingerprint_Enum import FeatureTyp


class SMART_pattern(object):

    __slots__ = ['__atomics', '__bonds', '__rdkitMOL', '__id', '__create_time', 'createInfo']

    def __init__(self, bonds, atomics=None, createInfo:Dict=None):
        if atomics is None:
            self.__atomics = []
        else:
            self.__atomics = atomics
        self.__bonds = bonds
        self.__id = self.__calcID()
        self.__create_time = datetime.datetime.now()
        self.createInfo = createInfo
        self.__rdkitMOL = -1

    def __str__(self):
        #TODO TEST
        string = ''
        for i, a in enumerate(self.__atomics):
            if i < len(self.__atomics) - 1:
                string = string + str(a) + self.__bonds[i]
            else:
                string = string + str(a)
        return string

    def __len__(self):
        return len(self.__atomics)


    def __eq__(self, other):
        if isinstance(other, SMART_pattern):
            if self.__equalList(self.__atomics, other.getAtomics()) and self.__equalList(self.__bonds, other.getBounds()):
                return True
            return False
        raise Exception('equal with no SMART_Pattern object is not implemented yet')

    def __equalList(self, a, b):
        tmp = [x for x in a if x in b]
        if len(tmp) == len(a):
            return True
        return False

    def addAtomics(self, atomic, recalcID: bool = False):
        self.__atomics.append(atomic)
        self.__rdkitMOL = -1
        if recalcID:
            self.__id = self.__calcID()
            self.__create_time = datetime.datetime.now()

    def toMol(self):
        if self.__rdkitMOL == -1:
            self.__rdkitMOL = Chem.MolFromSmarts(str(self))
        return self.__rdkitMOL

    def getMatchesWithMol(self, rdkitmol:Chem.Mol, bitfeature: FeatureTyp):
        if isinstance(rdkitmol, str):
            return 0
        else:
            matches = len(rdkitmol.GetSubstructMatches(self.toMol()))
            if bitfeature == FeatureTyp.match_feature:
                if matches == 0:
                    return 0
                else:
                    return 1
            elif bitfeature == FeatureTyp.count_feature:
                return matches
            else:
                raise Exception('FeatureTyp not implemented')

    def __calcID(self):
        return hashlib.md5(str(self).encode()).hexdigest()

    """
    1 is valid
    0 is not valid
    """

    def checkValidSmart(self):
        prim = Chem.MolFromSmarts(str(self))
        if not prim is None:
            return 1
        else:
            return 0

    def recalcID(self):
        self.__id = self.__calcID()
        self.__create_time = datetime.datetime.now()

    def getcreateTime(self):
        return self.__create_time

    def getID(self):
        return self.__id

    def getAtomics(self):
        return self.__atomics

    def getAtomicAt(self, index):
        return self.__atomics[index]

    def getBondAt(self, index):
        return self.__bonds[index]

    def getAtomicCount(self):
        return len(self.__atomics)

    def getBondCount(self):
        return len(self.__bonds)

    def getBounds(self):
        return self.__bonds

    def getgeneticSlice(self, start, end, withEndBond=False):
        atomic_subset = self.__atomics[start:end]
        if withEndBond:
            bonds_subset = self.__bonds[start:end]
        else:
            bonds_subset = self.__bonds[start:end - 1]
        return atomic_subset, bonds_subset
