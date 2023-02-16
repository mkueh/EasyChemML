import copy
from typing import List, Any, Tuple, Union

from EasyChemML.JobSystem.Utilities.Config import Config
from EasyChemML.Environment import Environment
from .Abstract_Hyperparametersearch import Abstract_Hyperparametersearch
from EasyChemML.JobSystem.Runner.Module.LocalRunner import LocalRunner
from ..Utilities.AbsolutConfigsExporter import AbsolutConfigExporter
from ..Utilities.AbsoluteConfig import AbsoluteConfig
from ..Utilities.ConfigStack import ConfigStack
from ..Utilities.HyperParamterTyps import AbstractHyperParamter
from EasyChemML.JobSystem.JobFactory.JobFactory import Job_Factory
from EasyChemML.JobSystem.JobFactory.Module.Jobs.ModelTrainEvalJob import ModelTrainEvalJob
from ...Metrik.MetricEnum import MetricDirection, MetricOutputType
from ...Metrik.MetricStack import MetricStack
from ...Metrik.Module.Abstract_Metric import Abstract_Metric
from ...Utilities.Dataset import Dataset


class GridSearch(Abstract_Hyperparametersearch):
    __env: Environment
    __result_export_path: str

    def __init__(self, env: Environment, result_export_path:str=None):
        """

        Args:
            env: EasyChemML Environment
            result_export_path: if this parameter is not none, the Gridsearch create a result CSV file
        """
        super().__init__(env)
        self.__env = env
        self.__result_export_path = result_export_path

    def performe_HyperparamterSearch(self, configs: Union[ConfigStack,Config], runner: LocalRunner, dataset: Dataset,
                                     metric: Abstract_Metric) -> Config:
        print('Grid search will be initialized')

        if metric.getMetric_Outputtype() == MetricOutputType.singleValue:
            metricStack = MetricStack({'optimise_metric': metric})
        else:
            raise Exception('The chosen Metrik returns multiple values. This is not supported yet')

        absolut_configs: List[AbsoluteConfig] = []
        for config in configs:
            absolut_configs.extend(self.__generateAbsoluteConfigs(config))

        calculation_jobs = self.__generate_ModelTrainEvalJobs(absolut_configs, dataset, metricStack)

        print(f'Run {len(calculation_jobs)} Jobs with {runner}')
        finished_jobs = runner.run_Jobs(calculation_jobs)
        print('Grid search finished')

        if self.__result_export_path is not None:
            exporter = AbsolutConfigExporter(self.__env)
            exporter.export_FinishedJobsAsCSV(calculation_jobs, self.__result_export_path)

        index_ofBest, best_job = self.__getBestResults(finished_jobs)

        print(f'find best combinate ... index {index_ofBest}')
        return Config(best_job.algorithm, best_job.explicit_algorithm_para)

    def getGeneratedJobs(self, configs: List[Config]) -> List[AbsoluteConfig]:
        absolut_configs: List[AbsoluteConfig] = []
        for config in configs:
            absolut_configs.extend(self.__generateAbsoluteConfigs(config))
        return absolut_configs

    def __generate_ModelTrainEvalJobs(self, absoluteConfigs: List[AbsoluteConfig], dataset: Dataset,
                                      metricStack: MetricStack) -> List[ModelTrainEvalJob]:
        jobs = []
        for out_index in range(dataset.get_Splitset().get_outerCount()):
            for in_index in range(dataset.get_Splitset().get_innerCount(out_index)):
                for ab_config in absoluteConfigs:
                    job_factory = Job_Factory(self.__env)
                    job: ModelTrainEvalJob = job_factory.create_ModelTrainEvalJob(dataset.name, dataset,
                                                                                  ab_config.toConfig(),
                                                                                  metricStack,
                                                                                  dataset.get_Splitset().get_inner_split(
                                                                                      out_index, in_index))
                    jobs.append(job)
        return jobs

    def __generateAbsoluteConfigs(self, config: Config) -> List[AbsoluteConfig]:
        parameters = config.algorithm_para

        paraSet = {}
        for parameter in parameters:
            if isinstance(parameters[parameter], AbstractHyperParamter):
                val: AbstractHyperParamter = parameters[parameter]
                paraSet[parameter] = val.toExplicitParameters()
            else:
                paraSet[parameter] = [parameters[parameter]]

        param_permutation = [{}, ]
        for k, v in paraSet.items():
            new_values = len(v)
            current_exp_len = len(param_permutation)
            for _ in range(new_values - 1):
                param_permutation.extend(copy.deepcopy(param_permutation[:current_exp_len]))
            for validx in range(len(v)):
                for exp in param_permutation[validx * current_exp_len:(validx + 1) * current_exp_len]:
                    exp[k] = v[validx]

        absoluteConfigs = []
        for para in param_permutation:
            new_absolut = AbsoluteConfig(config.algorithm, para)
            absoluteConfigs.append(new_absolut)
        return absoluteConfigs

    def __getBestResults(self, jobs: List[ModelTrainEvalJob]) -> Tuple[int, ModelTrainEvalJob]:
        best_result: Any
        indexOfBest: int = -1
        fin_job: ModelTrainEvalJob = None
        for i, fin_job in enumerate(jobs):
            if i == 0:
                indexOfBest = i
                best_result = fin_job.result_metric_TEST['optimise_metric']
            else:
                direct = fin_job.result_metric_TEST.metric_modules['optimise_metric'].getDirection()
                val = fin_job.result_metric_TEST['optimise_metric']
                if direct == MetricDirection.oneIsBest:
                    if abs(1 - val) < abs(1 - best_result):
                        indexOfBest = i
                        best_result = val
                elif direct == MetricDirection.lowerIsBetter:
                    if val < best_result:
                        indexOfBest = i
                        best_result = val
                elif direct == MetricDirection.higherIsBetter:
                    if val > best_result:
                        indexOfBest = i
                        best_result = val
                else:
                    raise Exception('MetricDirection not found')

        return indexOfBest, fin_job
