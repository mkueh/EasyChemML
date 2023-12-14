# distutils: language=c++
# cython: language_level=3
import rdkit.Chem

from EasyChemML.Encoder.impl_EvoFP.EVOFingerprint_Enum import FeatureTyp
from EasyChemML.Encoder.impl_EvoFP.Utilities.SMART_Pattern import SMART_pattern
from libcpp.vector cimport vector
from typing import List, Union, Tuple
from rdkit import Chem

from EasyChemML.Utilities.ParallelUtilities.Shared_PythonList import Shared_PythonList

"""Gives a Score (0 or 1) if the Pattern contain some information for the prediction"""
cdef class CSMART_IsColumneRelevant():

    def __init__(self):
        pass

    @staticmethod
    def calc_relevantPattern(pattern: SMART_pattern, train_dataset: Shared_PythonList, train_feature_cols:List[str], bit_feature: FeatureTyp) -> float:
        cdef int matches
        firstMatches = {}

        for col_name in train_dataset.getcolumns():
            if col_name in train_feature_cols:
                firstMatches[col_name] = pattern.getMatchesWithMol(train_dataset[0][col_name], bit_feature)

        for mol_index,_ in enumerate(train_dataset):
            for col_name in train_dataset.getcolumns():
                if col_name in train_feature_cols:
                    matches = pattern.getMatchesWithMol(train_dataset[mol_index][col_name], bit_feature)
                    if not matches == firstMatches[col_name]:
                        return 1.0
        return 0.0

    @staticmethod
    def calc_relevantSMART(SMART:Chem.Mol, train_dataset: Shared_PythonList, train_feature_cols:List[str], bit_feature: FeatureTyp) -> Tuple[float,str]:
        cdef int matches
        cdef int firstMatch

        def getMatchesWithMol(SMART, mol, bitfeature: FeatureTyp):
            if isinstance(mol, Chem.Mol):
                matches = len(mol.GetSubstructMatches(SMART))
                if bitfeature == FeatureTyp.match_feature:
                    if matches == 0:
                        return 0
                    else:
                        return 1
                elif bitfeature == FeatureTyp.count_feature:
                    return matches
                else:
                    raise Exception('featuretyp not implemented yet')
            else:
                return -1

        for col_name in train_dataset.getcolumns():
            if col_name in train_feature_cols:
                firstMatch = getMatchesWithMol(SMART,train_dataset[0][col_name], bit_feature)
                break

        for mol_index,_ in enumerate(train_dataset):
            for col_name in train_dataset.getcolumns():
                if col_name in train_feature_cols:
                    matches = getMatchesWithMol(SMART, train_dataset[mol_index][col_name], bit_feature)
                    if matches == -1:
                        continue
                    if not matches == firstMatch:
                        return 1.0, f'FirstMol_{Chem.MolToSmiles(train_dataset[0][0])}_{firstMatch},LastMol_{Chem.MolToSmiles(train_dataset[mol_index][col_name])}_{matches}'
        return 0.0, None
