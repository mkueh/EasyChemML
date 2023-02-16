import copy
from typing import List, Dict, Union
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.pyll.base import scope

from EasyChemML.JobSystem.Utilities.Config import Config
from EasyChemML.Environment import Environment
from .Abstract_Hyperparametersearch import Abstract_Hyperparametersearch
from EasyChemML.JobSystem.Runner.Module.LocalRunner import LocalRunner
from ..Utilities.ConfigStack import ConfigStack
from ..Utilities.HyperParamterTyps import IntRange, FloatRange, Categorically, AbstractHyperParamter
from ...JobSystem.JobFactory.Module.HyperparameterJob import HyperparameterJob
from ...JobSystem.Runner.Module import Abstract_Runner
from ...Metrik.MetricEnum import MetricDirection, MetricOutputType

import numpy as np

from ...Metrik.MetricStack import MetricStack
from ...Metrik.MetricsList import MetricList
from ...Metrik.Module.Abstract_Metric import Abstract_Metric
from ...Utilities.Dataset import Dataset


class Hyperopt(Abstract_Hyperparametersearch):
    _env: Environment

    def __init__(self, env: Environment):
        self._env = env

    def performe_HyperparamterSearch(self, configs: Union[ConfigStack, Config], runner: LocalRunner, dataset: Dataset,
                                     evalMetric: Abstract_Metric, maxEvalStepsPerConfig) -> Config:
        if not isinstance(configs, ConfigStack):
            if isinstance(configs, Config):
                configs = ConfigStack([configs])
            else:
                raise Exception('Parameter configs is not from Typ ConfigStack or Config')

        if not evalMetric.getMetric_Outputtype() == MetricOutputType.singleValue:
            raise Exception('The chosen Metric returns multiple values. This is not supported yet')

        config_bests = []
        bests_metricList = MetricList()
        for config in configs:
            print(f'Optimise configuration of algorithm {config.algorithm}')
            parameter = self._convertHyperparamert2Hyperopt(config, runner, dataset, evalMetric)
            trials = Trials()

            parameter['hyperopt_Bypass'] = (trials, config, runner, dataset, evalMetric)
            best = fmin(self._minimize2Funktion, space=parameter, algo=tpe.suggest, max_evals=maxEvalStepsPerConfig,
                        trials=trials)
            print(f'Best config for these run is: {best}')
            bests_metricList + trials.best_trial['result']['metric']
            config_bests.append((best,config))

        best_result_index = bests_metricList.getbestMetric('evalMetric', True)
        overall_best_config = config_bests[best_result_index]
        overall_best_config = self._generateAbsolutBestConfig(overall_best_config[1], overall_best_config[0])
        print(f'-----------------------------------------')
        print(f'Overall best is algo: {overall_best_config.algorithm}\nparameter: {overall_best_config.algorithm_para} ')
        print(f'-----------------------------------------')
        return overall_best_config

    def _minimize2Funktion(self, parameter: Dict):
        config: Config
        runner: Abstract_Runner
        dataset: Dataset
        evalMetric: Abstract_Metric
        trials: Trials()

        trials, config, runner, dataset, evalMetric = parameter['hyperopt_Bypass']
        del parameter['hyperopt_Bypass']

        jobs = []
        counter = 0
        print(f'Try parameter {parameter}')
        for out_index in range(dataset.get_Splitset().get_outerCount()):
            for in_index in range(dataset.get_Splitset().get_innerCount(out_index)):
                jobs.append(HyperparameterJob(
                    f'Round_{len(trials)}|HyperparameterJob_{counter}|OuterIndex_{out_index}|InnerIndex_{in_index}',
                    config.algorithm, parameter, dataset.get_FeatureData(),
                    dataset.get_FeatureData_Col(), dataset.get_TargetData(),
                    dataset.get_TargetData_Col(), MetricStack({'evalMetric': evalMetric}),
                    dataset.get_Splitset().get_inner_split_absolut(out_index, in_index)))
                counter += 1
        finished_jobs = runner.run_Jobs(jobs)
        metrics = MetricList()

        for i, job in enumerate(finished_jobs):
            metrics + job.evalMetric

        average = metrics.calcAverage()
        print(f'Average Result Metric: {average}')
        averageDictKey = average.keys_asList()[0]

        if average.getModule(averageDictKey).getDirection() == MetricDirection.oneIsBest:
            return {'loss': 1 - average[averageDictKey], 'status': STATUS_OK, 'metric': average}
        elif average.getModule(averageDictKey).getDirection() == MetricDirection.lowerIsBetter:
            return {'loss': average[averageDictKey], 'status': STATUS_OK, 'metric': average}
        elif average.getModule(averageDictKey).getDirection() == MetricDirection.higherIsBetter:
            return {'loss': average[averageDictKey] * -1, 'status': STATUS_OK, 'metric': average}
        else:
            raise Exception(f'{average.getModule(averageDictKey)} is not supported by Hyperopt')

    def _convertHyperparamert2Hyperopt(self, config: Config, runner, dataset: Dataset, evalMetric: Abstract_Metric):
        converted_para = {}
        parameters = config.algorithm_para
        for para in parameters:
            entry = parameters[para]
            if isinstance(entry, IntRange):
                converted_para[para] = scope.int(hp.quniform(para, int(entry.start), int(entry.stop), 1))
            elif isinstance(entry, FloatRange):
                converted_para[para] = hp.uniform(para, entry.start, entry.stop)
            elif isinstance(entry, Categorically):
                converted_para[para] = hp.choice(para, entry.items)
            else:
                converted_para[para] = parameters[para]
        return converted_para

    def _generateAbsolutBestConfig(self, hyperConfig: Config, optimizedParameters: Dict) -> Config:
        config = copy.deepcopy(hyperConfig)

        for key in config.algorithm_para:
            if key in optimizedParameters:
                entry = config.algorithm_para[key]
                if isinstance(entry, IntRange):
                    config.algorithm_para[key] = int(optimizedParameters[key])
                elif isinstance(entry, FloatRange):
                    config.algorithm_para[key] = float(optimizedParameters[key])
                elif isinstance(entry, Categorically):
                    config.algorithm_para[key] = optimizedParameters[key]
                else:
                    config.algorithm_para[key] = optimizedParameters[key]

        return config
