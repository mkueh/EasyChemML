import datetime
import logging
from typing import Any

import numpy as np
import tqdm as tqdm

from EasyChemML.Environment import Environment
from EasyChemML.JobSystem.JobFactory.Module.HyperparameterJob import HyperparameterJob
from EasyChemML.Utilities.DataUtilities.BatchDatatypHolder import BatchDatatypHolder
from EasyChemML.Utilities.DataUtilities.BatchTable import BatchAccess


class RunnerHyperparamterJob:
    environment: Environment
    _writeBack_lastIndex: int

    def __init__(self, environment: Environment):
        self.environment = environment

    def run_job(self, job: HyperparameterJob) -> HyperparameterJob:
        return self._naiv_run(job)

    def _naiv_run(self, job: HyperparameterJob) -> HyperparameterJob:
        clf = job.algorithm(job.explicit_algorithm_para)
        metric = job.evalMetric

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

        job.set_resultMetric(result_metric_TEST)
        return job

