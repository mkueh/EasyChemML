from EasyChemML.Utilities.SharedDataset import SharedDataset
from EasyChemML.Encoder.impl_EvoFP.Generator.SGenerator.SMARTGenerator import SMARTGenerator
from EasyChemML.Encoder.impl_EvoFP.Utilities.SMART_Fingerprint import SMART_Fingerprint
from EasyChemML.Utilities.ParallelUtilities.ParallelHelper import ParallelHelper
from EasyChemML.Utilities.ParallelUtilities.IndexQueues.IndexQueue_settings import IndexQueue_settings

from rdkit import rdBase
from typing import Union,List
import random, numpy as np

from ..EVOFingerprint_Enum import FeatureTyp

rdBase.DisableLog('rdApp.error')


class SMARTFingerprintGenerator(object):
    SG:SMARTGenerator
    multi_dataset = False


    def __init__(self, train_dataset:Union[SharedDataset, List[SharedDataset]], bit_feature:FeatureTyp, multi_dataset:bool=False):
        print('load SMART-Generator')
        if multi_dataset:
            self.SG = SMARTGenerator(train_dataset, bit_feature, multi_dataset)
        else:
            self.SG = SMARTGenerator([train_dataset], bit_feature, multi_dataset)
        self.multi_dataset = multi_dataset

    def generate_NewSMARTfps(self, fp_counts, length, bit_feature:FeatureTyp, max_primitivCounts=4, max_boundsCounts=4, n_jobs=1):
        count_pattern = fp_counts * length
        #if count_pattern > 100:
        #    print(f'Generate {count_pattern} pattern parallel')
        pattern, counts = self._generateSMART_Pattern(count_pattern, max_primitivCounts, max_boundsCounts, n_jobs=n_jobs, bit_feature=bit_feature)
        pattern_split = np.array_split(pattern, fp_counts)

        fp = []
        for i in range(fp_counts):
            fp.append(SMART_Fingerprint(list(pattern_split[i])))

        return fp, counts

    def create_SMARTfp(self, gen):
        tmp = SMART_Fingerprint(gen)
        return tmp

    def _generateSMART_Pattern(self, length, max_primitivCounts, max_boundsCounts, bit_feature:FeatureTyp, n_jobs:int=None) -> (SMART_Fingerprint, int):
        pattern = []
        false_patternCounter = 0

        IQ_settings = IndexQueue_settings(chunksize=2 ,start_index = 0, end_index = length)

        if n_jobs == 1:
            out_array = np.empty(shape=(length,), dtype=np.dtype('O'))
            for current_index in range(length):
                out_array[current_index] = self.SG.generateSMARTPattern(max_primitivCounts, max_boundsCounts, bit_feature, False)
            descendants = out_array

        else:
            parallelExecuter = ParallelHelper(n_jobs)
            descendants = parallelExecuter.execute_map_orderd_return(self._parallel_generateSMARTPattern, IQ_settings, np.dtype('O'),
                                                                     max_primitivCounts = max_primitivCounts , max_boundsCounts = max_boundsCounts, bit_feature = bit_feature, multi_dataset = self.multi_dataset)
        if len(descendants) > 1:
            random.shuffle(descendants)

        for item in descendants:
            pattern.append(item[0])
            false_patternCounter += item[1]

        return pattern, false_patternCounter

    def _parallel_generateSMARTPattern(self, max_primitivCounts, max_boundsCounts, bit_feature, multi_dataset, out_dtypes, current_chunk:int):
        out_array = np.empty(shape=(len(current_chunk),), dtype=out_dtypes)
        out_pointer = 0
        for _ in current_chunk:
            out_array[out_pointer] = self.SG.generateSMARTPattern(max_primitivCounts, max_boundsCounts, bit_feature, multi_dataset)
            out_pointer += 1
        return out_array
