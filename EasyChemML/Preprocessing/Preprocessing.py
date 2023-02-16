from EasyChemML.Utilities.Dataset import Dataset
from enum import Enum
from typing import List


class typs(Enum):
    FEATURE = 1
    TARGET = 2

class Preprocessing(object):
    preproConfigs = -1

    __featurePipeline = -1
    __targetPipeline = -1

    def __init__(self):
        self.__featurePipeline = self.__createPipeline(set.Preprocessing_feature_Steps,
                                                       set.Preprocessing_feature_Steps_param)
        self.__targetPipeline = self.__createPipeline(set.Preprocessing_target_Steps,
                                                        set.Preprocessing_target_Steps_param)

    def __createPipeline(self, elements, configs):
        out = []
        for i, elem in enumerate(elements):
            preclass = self.APP_ENV.get_ModulManager().get_prepro(elem)
            out.append(preclass(configs[i]))
        return out
            

    def __runPipeline(self, dataset, pipline, n_jobs:int, typ:typs):
        for item in pipline:
            item.convert(dataset, n_jobs, typ)



    def convert(self, dataset:Dataset, columns:List[str], n_jobs, **kwargs):
        print('Start preprocessing for feature pipeline')

        self.__runPipeline(dataset, self.__featurePipeline, n_jobs, typs.FEATURE)
        print('Start preprocessing for target pipeline')
        self.__runPipeline(dataset, self.__targetPipeline, n_jobs, typs.TARGET)
        print('Preprocessing Pipeline finished')
