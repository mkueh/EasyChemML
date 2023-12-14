"""Gives a Score (0 or 1) if the Pattern contain some information for the prediction"""

from EasyChemML.Encoder.impl_EvoFP.Utilities.SMART_Pattern import SMART_pattern
from EasyChemML.Encoder.impl_EvoFP.Utilities.SMART_Fingerprint import SMART_Fingerprint
from typing import List, Optional, Union

from EasyChemML.Utilities.Dataset import Dataset
from EasyChemML.Utilities.ParallelUtilities.Shared_PythonList import Shared_PythonList
from EasyChemML.Utilities.SharedDataset import SharedDataset
from EasyChemML.Encoder.impl_EvoFP.Utilities._SMART_IsColumneRelevant.CSMART_IsColumneRelevant import CSMART_IsColumneRelevant
from ..EVOFingerprint_Enum import FeatureTyp


class SMART_IsColumneRelevant():

    @staticmethod
    def calc_relevantPattern(pattern: SMART_pattern, train_dataset: Union[SharedDataset, Shared_PythonList, List[SharedDataset]], bit_feature: FeatureTyp, multi_dataset: bool) -> float:
        if isinstance(train_dataset, SharedDataset):
            return CSMART_IsColumneRelevant.calc_relevantPattern(pattern, train_dataset.get_FeatureData(), train_dataset.get_FeatureData_Col(), bit_feature)
        elif isinstance(train_dataset, List):
            for dataset in train_dataset:
                if CSMART_IsColumneRelevant.calc_relevantPattern(pattern, dataset.get_FeatureData(), dataset.get_FeatureData_Col(), bit_feature):
                    return True
            return False
        else:
            return CSMART_IsColumneRelevant.calc_relevantPattern(pattern, train_dataset, train_dataset.getcolumns(), bit_feature)

    @staticmethod
    def calc_relevantSMART(smart,  train_dataset: Union[SharedDataset,Shared_PythonList,List[SharedDataset]], bit_feature: FeatureTyp, multi_dataset: bool):
        if isinstance(train_dataset, SharedDataset):
            return CSMART_IsColumneRelevant.calc_relevantSMART(smart, train_dataset.get_FeatureData(), train_dataset.get_FeatureData_Col(), bit_feature)
        elif isinstance(train_dataset, List):
            for dataset in train_dataset:
                if CSMART_IsColumneRelevant.calc_relevantSMART(smart, dataset.get_FeatureData(), dataset.get_FeatureData_Col(), bit_feature):
                    return True
            return False
        else:
            return CSMART_IsColumneRelevant.calc_relevantSMART(smart, train_dataset, train_dataset.getcolumns(), bit_feature)