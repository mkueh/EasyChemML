from typing import Union, List

from EasyChemML.Utilities.Dataset import Dataset
from EasyChemML.Utilities.SharedDataset import SharedDataset
from .SMART_FingerprintGenerator import SMARTFingerprintGenerator
from EasyChemML.Encoder.impl_EvoFP.Utilities.SMART_Fingerprint import SMART_Fingerprint
from EasyChemML.Encoder.impl_EvoFP.Utilities.SMART_Pattern import SMART_pattern
from EasyChemML.Encoder.impl_EvoFP.Utilities.SMART_IsColumneRelevant import SMART_IsColumneRelevant
from EasyChemML.Encoder.impl_EvoFP.Generator._SMARTInheritance.C_SMART_FingerprintInheritance import C_SMART_FingerprintInheritance
import random

from ..EVOFingerprint_Enum import FeatureTyp


class SMART_FingerprintInheritance():
    bonds = [  # Bindungen ->  atoms + bonds + atoms
        '-',  # single bond (aliphatic)
        # '/',           #directional bond "up"
        # '\ '.strip(),  #directional bond "down"
        # '/?',          #directional bond "up or unspecified"
        # '\?',          #directional bond "down or unspecified"
        '=',  # double bond
        '#',  # triple bond
        ':',  # aromatic bond
        '~',  # any bond (wildcard)
        '@']  # any ring bond

    __SMARTgenerator: SMARTFingerprintGenerator = -1
    __parallelExecuter = -1
    __bitfeature: FeatureTyp = False
    _multi_dataset = False
    __relevantChecker: SMART_IsColumneRelevant
    __C_SFI: C_SMART_FingerprintInheritance

    def __init__(self, SMART_generator, bitfeature: FeatureTyp, multi_dataset=False, parallelExecuter=-1):
        self.__bitfeature = bitfeature
        self.__parallelExecuter = parallelExecuter
        self.__SMARTgenerator = SMART_generator
        self.__relevantChecker = SMART_IsColumneRelevant()
        self.__C_SFI = C_SMART_FingerprintInheritance()
        self._multi_dataset = multi_dataset

    def createInheritance_Pop(self, father: SMART_Fingerprint, mother: SMART_Fingerprint, newgen_rate,
                              gen_recombinationrate, feature_data: Union[SharedDataset, List[SharedDataset]], gen_index,
                              listof_IDs, index: int,
                              gen_recombinationAttempts=10):
        c_MutationSlicer_tries = 0
        c_genRandomizer_tries = 0
        c_generateNewGen_tries = 0

        sucess = -1

        while True:

            mutation = random.random()
            new_gen = None
            if mutation <= gen_recombinationrate:
                # start = Time_measure.starttimer('starte genMutationSlicer')
                new_gen, tries = self.__genMutationSlicer(father[gen_index], mother[gen_index], feature_data,
                                                          gen_recombinationAttempts)
                if new_gen is None:
                    c_MutationSlicer_tries += tries
                    continue
                c_MutationSlicer_tries += tries
                sucess = 1
                # Time_measure.stoptimer(start, 'ende genMutationSlicer')
            elif mutation >= gen_recombinationrate + newgen_rate:
                # start = Time_measure.starttimer('starte genRandomizer')
                # Mother, father gens without mutation
                new_gen = self.__genRandomizer(father[gen_index], mother[gen_index])
                c_genRandomizer_tries += 1
                sucess = 2
                # Time_measure.stoptimer(start, 'ende genRandomizer')
            else:
                # start = Time_measure.starttimer('starte NewGen')
                # New gens
                new_gen, tries = self.__generateNewGen()
                new_gen = new_gen[0][0]
                c_generateNewGen_tries += tries
                sucess = 3
                # Time_measure.stoptimer(start, 'ende NewGen')

            if not new_gen is None:
                id = new_gen.getID()
                for created_id in listof_IDs:
                    if not created_id['ID'] is None and created_id['ID'] == id:
                        continue
                listof_IDs[index] = new_gen.getID()

                return new_gen, c_MutationSlicer_tries, c_generateNewGen_tries, c_genRandomizer_tries, sucess

    def __relevantCheckForMultiColumn(self, testset: Union[SharedDataset, List[SharedDataset]], pattern):
        return self.__relevantChecker.calc_relevantPattern(pattern, testset, self.__bitfeature, self._multi_dataset)

    def __generateNewGen(self):
        return self.__SMARTgenerator.generate_NewSMARTfps(1, 1, max_primitivCounts=4, max_boundsCounts=4,
                                                          bit_feature=self.__bitfeature)

    def __genRandomizer(self, father: SMART_pattern, mother: SMART_pattern):
        gender = random.random()
        if gender >= 0.5:
            return father
        else:
            return mother

    def __genMutationSlicer(self, father: SMART_pattern, mother: SMART_pattern,
                            feature_data: Union[SharedDataset, List[SharedDataset]], attemps):
        for i in range(attemps):
            new_slice = self.__C_SFI.C_createMutationSlice(father, mother)
            if new_slice is None:
                continue
            if self.__relevantCheckForMultiColumn(feature_data, new_slice):
                return new_slice, i + 1
        return None, i
