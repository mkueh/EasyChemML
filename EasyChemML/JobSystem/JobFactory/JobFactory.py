from EasyChemML.Environment import Environment
from EasyChemML.JobSystem.Utilities.Config import Config
from EasyChemML.Metrik.MetricStack import MetricStack
from EasyChemML.JobSystem.JobFactory.Module.Jobs import ModelTrainEvalJob, ModelTrainJob, ModelEvalJob
from EasyChemML.Model.AbstractModel import Abstract_Model
from EasyChemML.Splitter.Splitcreator import Split
from EasyChemML.Utilities.Dataset import Dataset


class Job_Factory:
    environment: Environment

    def __init__(self, environment: Environment):
        self.environment = environment

    def create_ModelTrainEvalJob(self, job_name: str, dataset: Dataset, model_config: Config,
                                 metric: MetricStack, split: Split) -> ModelTrainEvalJob:
        new_job = ModelTrainEvalJob(job_name, model_config.algorithm, model_config.algorithm_para,
                                    dataset.get_FeatureData(),
                                    dataset.get_FeatureData_Col(),
                                    dataset.get_TargetData(), dataset.get_TargetData_Col(), metric, split)
        return new_job

    def create_ModelTrainJob(self, job_name: str, dataset: Dataset, model_config: Config,
                             split) -> ModelTrainJob:
        new_job = ModelTrainJob(job_name, model_config.algorithm, model_config.algorithm_para,
                                dataset.get_FeatureData(), dataset.get_FeatureData_Col(),
                                dataset.get_TargetData(), dataset.get_TargetData_Col(), split)
        return new_job

    def create_ModelEvalJob(self, job_name: str, dataset: Dataset, trained_Model: Abstract_Model, metric: MetricStack):
        new_job = ModelEvalJob(job_name, trained_Model, dataset.get_FeatureData(), dataset.get_FeatureData_Col(),
                               dataset.get_TargetData(), dataset.get_TargetData_Col(), metric)
        return new_job
