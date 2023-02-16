from typing import List
from typing import TYPE_CHECKING

from EasyChemML.Environment import Environment
from EasyChemML.JobSystem.JobFactory.Module.Jobs.ModelTrainEvalJob import ModelTrainEvalJob
from EasyChemML.Utilities.ExcelUtilities.CSV.CSVExporter import CSVExporter

if TYPE_CHECKING:
    pass

class AbsolutConfigExporter:
    __env: Environment

    def __init__(self, env: Environment):
        self.__env = env

    def export_FinishedJobsAsCSV(self, finishedJobs:List[ModelTrainEvalJob], filename:str):
        arr_algo = self.__FinishedJobs__AlgorithmArray(finishedJobs)
        arr_para = self.__FinishedJobs__ExplicitParaArray(finishedJobs)
        arr_train = self.__FinishedJobs__TrainMetrikArray(finishedJobs)
        arr_test = self.__FinishedJobs__TestMetrikArray(finishedJobs)
        print_arr = [arr_algo, arr_para, arr_train, arr_test]

        arr_columns = ['Algorithm', 'Parameter', 'Train Metric Values', 'Test Metric Values']

        CSVExporter.exportToCSV(print_arr, arr_columns, self.__env.WORKING_path, filename)


    def __FinishedJobs__AlgorithmArray(self, finishedJobs:List[ModelTrainEvalJob]):
        out = []

        for job in finishedJobs:
            out.append(str(job.algorithm))

        return out

    def __FinishedJobs__ExplicitParaArray(self, finishedJobs: List[ModelTrainEvalJob]):
        out = []

        for job in finishedJobs:
            out.append(str(job.explicit_algorithm_para))

        return out

    def __FinishedJobs__TestMetrikArray(self, finishedJobs: List[ModelTrainEvalJob]):
        out = []

        for job in finishedJobs:
            out.append(str(job.result_metric_TEST['optimise_metric']))

        return out

    def __FinishedJobs__TrainMetrikArray(self, finishedJobs: List[ModelTrainEvalJob]):
        out = []

        for job in finishedJobs:
            out.append(str(job.result_metric_TRAIN['optimise_metric']))

        return out