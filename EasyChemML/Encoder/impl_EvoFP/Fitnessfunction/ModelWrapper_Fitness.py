from enum import Enum

from EasyChemML.Encoder.impl_EvoFP.EVOFingerprint_Enum import FeatureTyp
from EasyChemML.Encoder.impl_EvoFP.Fitnessfunction.Abstract_Fitnessfunction import Abstract_Fitnessfunction
from EasyChemML.Encoder.impl_EvoFP.Utilities.SMART_Fingerprint import SMART_Fingerprint
from EasyChemML.Encoder.impl_RdkitConverter.MolRdkitConverter import MolRdkitConverter
from EasyChemML.JobSystem.Utilities.Config import Config
from EasyChemML.Metrik.MetricEnum import MetricDirection
from EasyChemML.Metrik.MetricStack import MetricStack
from EasyChemML.Metrik.Module import Abstract_Metric
from EasyChemML.Model.AbstractModel.Abstract_Model import Abstract_Model
from EasyChemML.Splitter.Module.Abstract_Splitter import Abstract_Splitter
from EasyChemML.Splitter.Module.ShuffleSplitter import ShuffleSplitter
from EasyChemML.Splitter.Splitcreator import Split

from EasyChemML.Utilities.Dataset import Dataset
from EasyChemML.Metrik.MetricsList import MetricList

import numpy as np
from typing import Dict, Union

from EasyChemML.Utilities.SharedDataset import SharedDataset


class Split_Mode(Enum):
    splitter_modul = 0
    fix_split = 1
    var_split = 2
    default = 99


class ModelWrapper_Fitness(Abstract_Fitnessfunction):
    __split: Union[Split, Abstract_Splitter]
    __splitMode: Split_Mode

    __metric: Abstract_Metric
    __fitness_metric_name: str
    __feature_typ: FeatureTyp
    __model: (Abstract_Model, Dict)
    __train_dataset: SharedDataset

    random_state: int = 42

    def __init__(self, train_dataset: Union[SharedDataset, Dataset], feature_typ: FeatureTyp, metric: Abstract_Metric,
                 model: Config, splitMode: Split_Mode = Split_Mode.default,
                 split: Union[Split, Abstract_Splitter] = None, pre_jobs: int = 1):
        self.__feature_typ = feature_typ
        self.__metric = metric

        self.__splitMode = splitMode
        self.__split = split

        if isinstance(train_dataset, Dataset):
            MolRdkitConverter().convert(datatable=train_dataset.get_FeatureData(),
                                        columns=train_dataset.get_FeatureData_Col_Encode(),
                                        n_jobs=pre_jobs)

            shared_dataset = train_dataset.to_SharedDataset(train_dataset.get_Splitset().get_outer_split(0).train)
            self.__train_dataset = shared_dataset
        else:
            self.__train_dataset = train_dataset

        if self.__splitMode == Split_Mode.splitter_modul:
            feature = self.__train_dataset.get_FeatureData()
            self.__split = self.__split.split(feature)
        elif self.__splitMode == Split_Mode.fix_split:
            pass
        elif self.__splitMode == Split_Mode.var_split:
            if isinstance(self.__split, Abstract_Splitter) and self.__split.contains_random_state():
                pass
            else:
                raise Exception('split is not a Abstract_Splitter or the Splitter dont containe a random_state')
        elif self.__splitMode == Split_Mode.default:
            splitter = ShuffleSplitter(n_splits=3, random_state=42, test_size=0.3)
            feature = self.__train_dataset.get_FeatureData()
            self.__split = splitter.split(feature)

        self.__model = model

    def get_datasets(self) -> SharedDataset:
        return self.__train_dataset

    def _get_splitting(self):
        if self.__splitMode == Split_Mode.fix_split:
            return self.__splitts
        elif self.__splitMode == Split_Mode.var_split:
            feature = self.__train_dataset.get_FeatureData()
            self.__split.random_state = self.random_state
            return self.__split.split(self.__train_dataset.get_FeatureData())
        elif self.__splitMode == Split_Mode.default:
            return self.__split

    def calc_fitness(self, one: SMART_Fingerprint, working_path, n_jobs=1) -> (
            float, int):
        feature_data = self.__train_dataset.get_FeatureData()
        target_data = self.__train_dataset.get_TargetData()

        feature = one.getFingerpintof2DArr(feature_data[:][self.__train_dataset.get_FeatureData_Col()],
                                           self.__feature_typ)
        feature = np.asarray(feature)

        metrics = MetricList()
        metricStack = MetricStack({'fitness': self.__metric})
        model = self.__model.algorithm(self.__model.algorithm_para)

        for i, (train, test) in enumerate(self._get_splitting()):
            target_dtypes = self.__train_dataset.get_TargetData().get_BatchDatatypHolder()
            train_feature = feature[train]
            train_target = target_data[train]

            target_dtypes = target_dtypes.toNUMPY_dtypes(flatMe=True)
            train_target = train_target.view(target_dtypes).reshape(train_target.shape + (-1,))

            test_feature = feature[test]
            test_target = target_data[test]

            test_target = test_target.view(target_dtypes).reshape(test_target.shape + (-1,))

            try:
                model.fit(train_feature, train_target)
            except Exception as e:
                print(f'Catboost crashed, fitness = 0')
                print(str(e))
                return 0, 0
            X = feature[test]
            y_predict = model.predict(X)
            y_true = test_target

            y_predict_proba = None
            if model.hasPredicte_proba():
                y_predict_proba = model.predicte_proba(X)

            metric = metricStack.calcMetric(y_true, y_predict, y_predict_proba)

            metrics + metric

        fitness_list = metrics.calcAverage()
        fitness = fitness_list['fitness']
        direction = metric.metric_modules['fitness'].getDirection()

        if direction == MetricDirection.lowerIsBetter:
            fitness = 0 - fitness

        return fitness, 0
