# cython: language_level=3
# cython: c_string_type=unicode, c_string_encoding=utf8
# distutils: language = c++

from libcpp cimport bool
import random
from rdkit import Chem
from libcpp.string cimport string

from EasyChemML.Encoder.impl_EvoFP.EVOFingerprint_Enum import FeatureTyp
from EasyChemML.Encoder.impl_EvoFP.Utilities.SMART_Pattern import SMART_pattern
from EasyChemML.Encoder.impl_EvoFP.Utilities.SMART_IsColumneRelevant import SMART_IsColumneRelevant
from bitarray import bitarray

from typing import List, Union

from EasyChemML.Utilities.Dataset import Dataset
from EasyChemML.Utilities.ParallelUtilities.Shared_PythonList import Shared_PythonList
from EasyChemML.Utilities.SharedDataset import SharedDataset

cdef class SMARTGenerator:
    #https://www.daylight.com/dayhtml/doc/theory/theory.smarts.html
    #https://www.daylight.com/dayhtml/doc/theory/theory.smiles.html
    #https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/MACCSkeys.py
    #https://www.rdkit.org/docs/RDKit_Book.html
    #https://www.rdkit.org/docs/GettingStartedInPython.html#fingerprinting-and-molecular-similarity

    #Set to organic spezific values
    #atoms = [1,3,5,6,7,8,9,11,12,13,14,15,16,17,19,34,35,50,53] #Ordnungszahlen (118 = Oganesson)
    #Set to organic spezific values

    bonds = [   #Bindungen ->  atoms + bonds + atoms
             '-',           #single bond (aliphatic)
             #'/',           #directional bond "up"
             #'\ '.strip(),  #directional bond "down"
             #'/?',          #directional bond "up or unspecified"
             #'\?',          #directional bond "down or unspecified"
             '=',           #double bond
             '#',           #triple bond
             ':',           #aromatic bond
             '~',           #any bond (wildcard)
             '@']           #any ring bond

    logicBIOperations = [
                       #'&', #e1&e2 -> e1 and e2 (high precedence) (NOT SUPPORTED BY RDKIT)
                       #',', #e1,e2 -> e1 or e2
                       ';'  #e1;e2 -> e1 and e2 (low precedence)
                       ]

    logicSIOperations = [
        '!', #!e1 -> not e1
        ''] #Not not ;)

    cdef list primitives
    train_SharedDatasets:Union[SharedDataset, List[SharedDataset]]

    cdef Py_ssize_t len_primitives
    cdef Py_ssize_t len_logicBI
    cdef Py_ssize_t len_logicSI
    cdef Py_ssize_t len_bonds

    cdef bool bit_feature
    cdef bool multi_dataset

    #Stop her

    def __init__(self, train_SharedDatasets:Union[SharedDataset, List[SharedDataset]], bit_feature:FeatureTyp, multi_dataset:bool):
        """
        :param train_feature: only be used for the primitives pattern check (possible patterns)
        :param bit_feature: count hits or boolean true/false when hit
        :param multi_dataset: multi-dataset fingerprint?
        """
        self.primitives = self.__createPrimitives(train_SharedDatasets, bit_feature, multi_dataset)
        self.bit_feature = bit_feature
        self.multi_dataset = multi_dataset
        self.len_primitives = len(self.primitives)
        self.len_logicBI = len(self.logicBIOperations)
        self.len_logicSI = len(self.logicSIOperations)
        self.len_bonds = len(self.bonds)
        self.train_SharedDatasets = train_SharedDatasets

    def __getElementsOfaMol(self, mol:Chem.Mol):
        if isinstance(mol, Chem.Mol):
            elementisIn = bitarray(120)
            elementisIn.setall(False)
            for atom in mol.GetAtoms():
                ordnungsZahl = atom.GetAtomicNum()
                elementisIn[ordnungsZahl] = True
            return elementisIn
        else:
            elementisIn = bitarray(120)
            elementisIn.setall(False)
            return elementisIn

    def __checkprimitiv(self, s_item, train_feature:Shared_PythonList, train_feature_cols:List[str], bit_feature):
        s = string(b'[')
        s = s + s_item
        s = s + string(b']')

        s_mol = Chem.MolFromSmarts(s)

        if s_mol is None:
            return False

        feedback = SMART_IsColumneRelevant.calc_relevantSMART(s_mol, train_feature, bit_feature, self.multi_dataset)
        if feedback[0] > 0:
            return True, feedback[1]
        else:
            return False, None

    def __createIntervall(self, start, end):
        intervals =  []

        for i in range(start, end):
            for j in range(i, end):
                intervals.append((i,j))

        return intervals

    def __checkBounderies(self, stprop, start, end, train_feature:Shared_PythonList, train_feature_cols:List[str], bit_feature):
        relevant = []
        for i in range(start,end):
            st = stprop+'{'+str(i)+'-'+str(i)+'}'
            feedback = self.__checkprimitiv(st, train_feature, train_feature_cols, bit_feature)
            if feedback[0]:
                relevant.append((i,feedback[1]))
        if len(relevant) == 0:
            return 0 , 0
        return relevant[0][0], relevant[-1][0]+1

    def __createPropterty(self, stprop, start, end, train_feature:Shared_PythonList, train_feature_cols:List[str], bit_feature:FeatureTyp):
        prop = []
        start, end = self.__checkBounderies(stprop, start, end, train_feature, train_feature_cols, bit_feature)
        intervals = self.__createIntervall(start,end)
        for interval in intervals:
            st = stprop+'{'+str(interval[0])+'-'+str(interval[1])+'}'
            feedback = self.__checkprimitiv(st, train_feature, train_feature_cols, bit_feature)
 #           feedback = (True, feedback[1])
            if feedback[0]:
                prop.append((st, feedback[1]))
        print(f'found {stprop} pattern: {prop}')

        out = []
        for p in prop:
            out.append(p[0])
        return out

    def __atomInDataset(self, train_feature:Shared_PythonList, train_feature_cols:List[str]):
        allElementsinDataset = bitarray(120)
        allElementsinDataset.setall(False)
        for row in train_feature:
            for col_name in train_feature.getcolumns():
                if col_name in train_feature_cols:
                    bitarr:bitarray = self.__getElementsOfaMol(row[col_name])
                    allElementsinDataset = allElementsinDataset | bitarr

        intlist = []
        for i, x in enumerate(allElementsinDataset):
            if x:
                intlist.append(i)
        print('found atoms: ' + str(intlist))
        return intlist

    cpdef __createPrimitives(self, train_SharedDatasets:Union[SharedDataset, List[SharedDataset]], bit_feature:FeatureTyp, multi_dataset:bool):
        if isinstance(train_SharedDatasets, list):
            patterns = set([])
            for dataset in train_SharedDatasets:
                print(f'Check pattern for dataset : {dataset.name}')
                patterns = patterns.union(set(self.__createPrimitives(dataset, bit_feature, False)))
            print('---------------------------------------------------------')
            print(f'found {len(patterns)} patterns')
            print(patterns)
            print('---------------------------------------------------------')
            return list(patterns)
        else:
            train_feature = train_SharedDatasets.get_FeatureData()
            train_feature_cols = train_SharedDatasets.get_FeatureData_Col_Encode()
        primitives = []

        for a in self.__atomInDataset(train_feature, train_feature_cols): #create Atoms
            primitives.append('#'+str(a))

        #primitives.append('*')     #Wildecard
        #primitives.append('a')     #aromatic
        #primitives.append('A')     #aliphatic
        #for i in range(1,maxDegree+1):
            # primitives.append('<' + str(i) + '>')  # explicit atomic mass (NOT SUPPORTED BY RDKIT)
            # primitives.append('H<' + str(i) + '>')  # n attached hydrogens (NOT SUPPORTED BY RDKIT)
            # primitives.append('<' + str(i) + '>')  # explicit atomic mass (NOT SUPPORTED BY RDKIT)
            #matches atoms that have less than or equal to X explicit connections

        primitives.extend(self.__createPropterty('D', 0, 6, train_feature, train_feature_cols, bit_feature))
        primitives.extend(self.__createPropterty('h', 1, 4, train_feature, train_feature_cols, bit_feature))
        primitives.extend(self.__createPropterty('R', 1, 5, train_feature, train_feature_cols, bit_feature))
        primitives.extend(self.__createPropterty('r', 3, 21, train_feature,train_feature_cols, bit_feature))
        primitives.extend(self.__createPropterty('v', 1, 5, train_feature, train_feature_cols, bit_feature))
        primitives.extend(self.__createPropterty('X', 1, 7, train_feature, train_feature_cols, bit_feature))
        primitives.extend(self.__createPropterty('x', 0, 5, train_feature, train_feature_cols, bit_feature))
        primitives.extend(self.__createPropterty('-', 0, 3, train_feature, train_feature_cols, bit_feature))
        primitives.extend(self.__createPropterty('+', 0, 3, train_feature, train_feature_cols, bit_feature))
        primitives.extend(self.__createPropterty('z', 0, 5, train_feature, train_feature_cols, bit_feature))
        primitives.extend(self.__createPropterty('Z', 0, 5, train_feature, train_feature_cols, bit_feature))

        #@<c><n>     	chiral class <c> chirality <n>
        #@<c><n>?        chirality <c><n> or unspecified

        st = '@'
        if self.__checkprimitiv(st, train_feature, train_feature_cols, bit_feature):
            primitives.append('@')   # chirality anticlockwise

        st = '@@'
        if self.__checkprimitiv(st, train_feature, train_feature_cols, bit_feature):
            primitives.append('@@')    # chirality clockwise

        print(f'primitive count {len(primitives)}')
        print(primitives)
        #for item in self.primitives:
        #    print(item)
        return primitives

    def __createPrimitivPattern(self, int primitivCount):
        cdef string s
        cdef string tmp
        cdef unsigned long long counter = 1

        if random.random() < 0.01: #Wildecard
            return str('*'), counter

        while True:
            s = string(b'[')
            for i in range(primitivCount):
                prim = random.choice(self.primitives)
                while s.find(prim) < len(s):
                    prim = random.choice(self.primitives)
                if random.random() > 0.5: #Pattern ! or not !
                    s = s + string(b'!')
                    s = s + prim
                else:
                    s = s + prim
                if not i == primitivCount-1:
                    s = s + random.choice(self.logicBIOperations)
            s = s + string(b']')
            if self.__checkValidSmart(s):
                return s, counter

            counter = counter + 1
            s = string(b'[')


    def generateSMARTPattern(self, int MaxPrimitivVarCount, int MaxBoundCount, bit_feature:FeatureTyp, multi_dataset: bool):
        cdef unsigned long long counter = 0

        while True:
            if MaxBoundCount == 1:
                boundCount = 1
            else:
                boundCount = random.randrange(1,MaxBoundCount)
            #create bonds

            bonds = []
            for i in range(boundCount):
                bonds.append(self.bonds[random.randrange(self.len_bonds)])

            pattern = SMART_pattern(bonds = bonds, createInfo={'createTyp': 'NewGenerator'})

            #fill atomics with primitivs
            for i in range(boundCount+1):
                if MaxPrimitivVarCount == 1:
                    primCount = 1
                else:
                    primCount = random.randrange(1,MaxPrimitivVarCount)

                tmp = self.__createPrimitivPattern(primCount)
                pattern.addAtomics(tmp[0])
                counter = counter + tmp[1]
            for dataset in self.train_SharedDatasets:

                train_feature_cols = dataset.get_FeatureData_Col_Encode()
                if SMART_IsColumneRelevant.calc_relevantPattern(pattern, dataset, bit_feature, multi_dataset) > 0:
                    pattern.recalcID()
                    return pattern, counter

    cdef int __checkValidSmart(self, string SMART):
        prim = Chem.MolFromSmarts(SMART)
        if not prim is None:
            return 1
        else:
            return 0