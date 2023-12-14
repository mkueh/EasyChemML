import math
from random import randrange

from EasyChemML.Encoder.impl_EvoFP.Utilities.SMART_Fingerprint import SMART_Fingerprint
from EasyChemML.Metrik.MetricStack import MetricStack
from EasyChemML.Metrik.Module.Abstract_Metric import Abstract_Metric
from EasyChemML.Splitter.Module.ShuffleSplitter import ShuffleSplitter
from EasyChemML.Utilities.Dataset import Dataset
from EasyChemML.Model.AbstractModel.Abstract_Model import Abstract_Model
from EasyChemML.Utilities.SharedDataset import SharedDataset
from .Abstract_Fitnessfunction import Abstract_Fitnessfunction

from .ModelWrapper_Fitness import ModelWrapper_Fitness, Split_Mode

import numpy as np
from typing import Dict, List, Tuple, Type

from ..EVOFingerprint_Enum import FeatureTyp
from EasyChemML.Encoder.impl_RdkitConverter.MolRdkitConverter import MolRdkitConverter


class MultiDataset_Fitness(Abstract_Fitnessfunction):
    models: Dict[str, Tuple[Type[Abstract_Model], Dict, Type[Abstract_Metric]]]
    datasets: List[SharedDataset] = []

    weighting_funct: str = 'average'
    metric_obj: MetricStack
    fitness_metric_name: str

    metrics_para: Dict = {}
    models_param: Dict = {}

    _random_state = -1
    _modelWrappers: List[ModelWrapper_Fitness]

    def __init__(self, datasets: List[Dataset], feature_typ: FeatureTyp,
                 models: Dict[str, Tuple[Type[Abstract_Model], Dict, Type[Abstract_Metric]]],
                 weightingfunction: str,
                 pre_jobs: int = 1):
        self.weighting_funct = weightingfunction
        self.models = models
        self._modelWrappers = []

        shared_datasets = []
        for dataset in datasets:
            MolRdkitConverter().convert(datatable=dataset.get_FeatureData(),
                                        columns=dataset.get_FeatureData_Col_Encode(),
                                        n_jobs=pre_jobs)

            if dataset.get_Splitset().get_outerCount() > 1:
                print('EVO-FP only can handle OuterSplit count of 1')
            split_obj = dataset.get_Splitset().get_outer_split(0)
            shared_dataset = dataset.to_SharedDataset(split_obj.train)
            shared_datasets.append(shared_dataset)
            splitter = ShuffleSplitter(n_splits=1, random_state=42, test_size=0.3)
            self._modelWrappers.append(
                ModelWrapper_Fitness(shared_dataset, feature_typ=feature_typ, metric=self.models[dataset.name][2],
                                     model=(self.models[dataset.name][0],self.models[dataset.name][1]), splitMode=Split_Mode.var_split,
                                     split=splitter))

        self.datasets = shared_datasets
        self.new_random_state()

    def calc_fitness(self, one: SMART_Fingerprint, working_path, n_jobs=1) -> (float, int):
        fitness_scores = []

        for i, payload_wrapper in enumerate(self._modelWrappers):
            payload_wrapper.random_state = self._random_state
            dataset = self.datasets[i]
            result = payload_wrapper.calc_fitness(one, working_path, n_jobs)
            fitness_scores.append(result[0])

        return fitness_scores

    def get_datasets(self) -> List[Dataset]:
        return self.datasets

    def new_random_state(self):
        self._random_state = randrange(50564789)
        print(f'change Random-state {self._random_state}')

    def get_payload_order(self):
        names = []
        for dataset in self.datasets:
            names.append(dataset.name)
        return names

    def calc_multiDataset_fitness(self, fitness_values: List[List[float]]):
        if self.weighting_funct == 'average':
            fitness_values = self._calc_fitness_average(fitness_values, False)
        elif self.weighting_funct == 'average_pow':
            fitness_values = self._calc_fitness_average(fitness_values, True)
        elif self.weighting_funct == 'norm_max_pow':
            val = self._calc_fitness_norm_with_max(fitness_values, True)
            fitness_values = val
        elif self.weighting_funct == 'norm_max':
            val = self._calc_fitness_norm_with_max(fitness_values, False)
            fitness_values = val
        elif self.weighting_funct == 'norm_asList':
            val = self._calc_fitness_norm_with_list(fitness_values, False)
            fitness_values = val
        elif self.weighting_funct == 'norm_asList_pow':
            val = self._calc_fitness_norm_with_list(fitness_values, True)
            fitness_values = val
        else:
            raise Exception(f'{self.weighting_funct} not found')

        return fitness_values

    def _calc_fitness_norm_with_list(self, fitness_values: List[List[float]], pow_values: bool = False):
        scores = []
        max_values = self._get_list_norms()

        for i, fp_fitValues in enumerate(fitness_values):
            fit_values = []
            for j, dataset_fitValue in enumerate(fp_fitValues):
                if dataset_fitValue <= 0:
                    dataset_fitValue_abs = abs(dataset_fitValue)
                    val = dataset_fitValue_abs / max_values[j]
                    fit_values.append(-1.0 * val)
                else:
                    fit_values.append(dataset_fitValue / max_values[j])

            if pow_values:
                scores.append(self._pow_average(fit_values))
            else:
                scores.append(np.average(fit_values))

        return scores

    def _get_list_norms(self):
        output = []
        if 'norm_values' in self.weighting_funct_param:
            norm_vals = self.weighting_funct_param['norm_values']

            for dataset in self.datasets:
                if dataset.name in norm_vals:
                    output.append(norm_vals[dataset.name])
                else:
                    raise Exception(f'norm_values not contain a value form dataset {dataset.name}')
        else:
            raise Exception('weightingfunction_param not contain norm_values')
        return output

    def _calc_fitness_norm_with_max(self, fitness_values: List[List[float]], pow_values: bool = False):
        scores = []
        max_values = self._find_max(fitness_values)

        for i, fp_fitValues in enumerate(fitness_values):
            fit_values = []
            for j, dataset_fitValue in enumerate(fp_fitValues):
                if dataset_fitValue <= 0:
                    dataset_fitValue_abs = abs(dataset_fitValue)
                    val = dataset_fitValue_abs / max_values[j]
                    fit_values.append(-1.0 * val)
                else:
                    fit_values.append(dataset_fitValue / max_values[j])

            if pow_values:
                scores.append(self._pow_average(fit_values))
            else:
                scores.append(np.average(fit_values))

        return scores

    def _calc_fitness_average(self, fitness_values: List[List[float]], pow_values=False):
        scores = []
        for i, fp_fitValues in enumerate(fitness_values):
            fit_values = []
            for j, dataset_fitValue in enumerate(fp_fitValues):
                fit_values.append(dataset_fitValue)

            if pow_values:
                scores.append(self._pow_average(fit_values))
            else:
                scores.append(np.average(fit_values))
        return scores

    def _pow_average(self, scores: List[float]):
        for i, _ in enumerate(scores):
            if scores[i] <= 0:
                scores[i] = -1.0 * math.pow(scores[i], 2)
            else:
                scores[i] = math.pow(scores[i], 2)
        return np.average(scores)

    def _find_max(self, fitness_values: List[List[float]]):
        max_vals = []
        for i, _ in enumerate(fitness_values[0]):
            max_val = fitness_values[0][i]
            for j in range(len(fitness_values)):
                if fitness_values[j][i] > max_val:
                    max_val = fitness_values[j][i]

            if max_val <= 0:
                max_val = 1
            max_vals.append(max_val)
        return max_vals
