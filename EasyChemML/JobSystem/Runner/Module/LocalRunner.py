import logging, datetime

from EasyChemML.Environment import Environment
from EasyChemML.JobSystem.JobFactory.Module.Jobs import ModelTrainEvalJob, ModelPredictJob, ModelTrainJob, ModelEvalJob
from EasyChemML.JobSystem.JobFactory.Module.HyperparameterJob import HyperparameterJob
from EasyChemML.JobSystem.Runner.Module.Abstract_Runner import Abstract_Runner
from EasyChemML.JobSystem.Runner.Module._localRunner.RunnerHyperparameterJob import RunnerHyperparamterJob
from EasyChemML.JobSystem.Runner.Module._localRunner.RunnerModelPredictJob import RunnerModelPredictJob
from EasyChemML.JobSystem.Runner.Module._localRunner.RunnerModelTrainJob import RunnerModelTrainJob

from EasyChemML.Utilities.ExcelUtilities.CSV.CSVExporter import CSVExporter


class LocalRunner(Abstract_Runner):
    environment: Environment

    def __init__(self, environment: Environment):  # TODO param
        self.environment = environment

    def run_Job(self, job):
        return self._run_job(job)

    def run_Jobs(self, jobs):
        tmp = []
        for job in jobs:
            tmp.append(self._run_job(job))
        return tmp

    def _run_job(self, job):
        print(f'run job: {job.job_name}')
        if isinstance(job, HyperparameterJob):
            runner: RunnerHyperparamterJob = RunnerHyperparamterJob(self.environment)
            return runner.run_job(job)
        elif isinstance(job, ModelTrainEvalJob):
            return self._run_ModelTrainEvalJob(job)
        elif isinstance(job, ModelTrainJob):
            runner: RunnerModelTrainJob = RunnerModelTrainJob(self.environment)
            return runner.run_job(job)
        elif isinstance(job, ModelPredictJob):
            runner: RunnerModelPredictJob = RunnerModelPredictJob(self.environment)
            return runner.run_job(job)
        elif isinstance(job, ModelEvalJob):
            return self._run_ModelEvalJob(job)
        else:
            raise Exception('unkown jobclass')

    def _run_HyperparameterJob(self, job: HyperparameterJob):
        raise Exception('not implemented yet')

    def _run_ModelTrainEvalJob(self, job: ModelTrainEvalJob):
        clf = job.algorithm(job.explicit_algorithm_para)
        metric = job.targetMetric

        logging.info(str(job.algorithm) + " is started")
        logging.info('Time: ' + str(datetime.datetime.now()))

        X_train = job.X.convert_2_ndarray(job.split.train, job.X_cols)
        y_train = job.y.convert_2_ndarray(job.split.train, job.y_cols)
        X_test = job.X.convert_2_ndarray(job.split.test, job.X_cols)
        y_test = job.y.convert_2_ndarray(job.split.test, job.y_cols)

        clf.fit(X=X_train, y=y_train)

        # Testset Prediction
        y_test_predict = clf.predict(X_test)
        y_predict_proba = None
        if clf.hasPredicte_proba():
            y_predict_proba = clf.predicte_proba(X_test)
        result_metric_TEST = metric.calcMetric(y_test, y_test_predict, y_predict_proba)

        # Trainset Prediction
        X_train_predict = clf.predict(X_train)
        X_train_predict_proba = None
        if clf.hasPredicte_proba():
            X_train_predict_proba = clf.predicte_proba(X_train)
        result_metric_TRAIN = metric.calcMetric(y_train, X_train_predict, X_train_predict_proba)

        # Print Testset prediction
        columns_testP = ['True_values', 'Predicted_values']
        columns_testP.extend(result_metric_TEST.keys())
        arrays = [y_test, y_test_predict]
        for metric in result_metric_TEST.keys():
            arrays.append([result_metric_TEST[metric]])

        CSVExporter.exportToCSV(arrays=arrays,
                           columns_a=columns_testP,
                           path=self.environment.WORKING_path,
                           CSV_filename=('Outer_Step_' + job.job_name + '_Testprediction'))

        # Print Trainset prediction
        columns_trainP = ['True_values', 'Predicted_values']
        columns_trainP.extend(result_metric_TRAIN.keys())
        arrays = [y_train, X_train_predict]
        for metric in result_metric_TRAIN.keys():
            arrays.append([result_metric_TRAIN[metric]])

        CSVExporter.exportToCSV(arrays=arrays,
                           columns_a=columns_trainP,
                           path=self.environment.WORKING_path,
                           CSV_filename=('Outer_Step_' + job.job_name + '_Trainprediction'))

        job.set_resultMetric(result_metric_TEST, result_metric_TRAIN)
        job.set_trained_Model(clf)
        return job

    def _run_ModelEvalJob(self, job: ModelEvalJob):
        metric = job.result_metric

        logging.info(str(job.job_name) + " is started")
        logging.info('Time: ' + str(datetime.datetime.now()))

        X = job.X.convert_2_ndarray(columns=job.X_cols)
        y = job.X.convert_2_ndarray(columns=job.y_cols)

        clf = job.trained_Model
        # Testset Prediction
        y_predict = clf.predict(X)
        y_predict_proba = None
        if clf.hasPredicte_proba():
            y_predict_proba = clf.predicte_proba(X)
        result_metric = metric.calcMetric(y, y_predict, y_predict_proba)

        # Print Testset prediction
        columns_testP = ['True_values', 'Predicted_values']
        columns_testP.extend(result_metric.keys())
        arrays = [y, y_predict]
        for metric in result_metric.keys():
            arrays.append([result_metric[metric]])

        CSVExporter.exportToCSV(arrays=arrays,
                           columns_a=columns_testP,
                           path=self.environment.WORKING_path,
                           CSV_filename=('Outer_Step_' + job.job_name + '_Testprediction'))

        job.set_resultMetric(result_metric)
        job.set_trained_Model(clf)
        return job
