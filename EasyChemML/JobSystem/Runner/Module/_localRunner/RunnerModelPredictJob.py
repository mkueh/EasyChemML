import datetime
import logging
from typing import Any

import numpy as np
import tqdm as tqdm

from EasyChemML.Environment import Environment
from EasyChemML.JobSystem.JobFactory.Module.Jobs import ModelPredictJob
from EasyChemML.Utilities.DataUtilities.BatchDatatypHolder import BatchDatatypHolder
from EasyChemML.Utilities.DataUtilities.BatchTable import BatchAccess


class RunnerModelPredictJob:
    environment: Environment
    _writeBack_lastIndex: int

    def __init__(self, environment: Environment):
        self.environment = environment

    def run_job(self, job: ModelPredictJob) -> ModelPredictJob:
        if job.processInBatches:
            return self._run_ModelPredictJob(job)
        else:
            return self._naiv_run(job)

    def _writeBack(self, predicted_vals: Any, job: ModelPredictJob):
        if isinstance(predicted_vals, np.ndarray):
            pass
        else:
            predicted_vals = np.array(predicted_vals)

        if job.writeInBatchSystem is None or job.writeInBatchSystem is False:
            job.set_predicted_vals(predicted_vals)
        else:
            bp, key = job.writeInBatchSystem

            if key in bp:
                bp[key][self._writeBack_lastIndex:len(predicted_vals)] = predicted_vals
                self._writeBack_lastIndex += len(predicted_vals)
            else:
                dataTypHolder = BatchDatatypHolder().fromNUMPY_dtyp(predicted_vals.dtype)
                bp.createDatabase(key, dataTypHolder, (len(job.X),))
                bp[key][0:len(predicted_vals)] = predicted_vals
                self._writeBack_lastIndex = len(predicted_vals)
                job.set_predicted_vals(bp[key])

    def _naiv_run(self, job: ModelPredictJob) -> ModelPredictJob:
        features = job.X.convert_2_ndarray(job.X_indices, job.X_cols)
        clf = job.trained_Model

        predicted_vals = clf.predict(features)

        if not job.skipSavePrediction:
            self._writeBack(predicted_vals, job)

        return job

    def _run_ModelPredictJob(self, job: ModelPredictJob) -> ModelPredictJob:
        logging.info(str(job.trained_Model) + " is started")
        logging.info('Time: ' + str(datetime.datetime.now()))

        if job.X_indices is None:
            iterator: BatchAccess = iter(job.X)
            batch: np.ndarray

            with tqdm(total=len(iterator)) as pbar:
                status_count = 0

                for batch in iterator:
                    clf = job.trained_Model
                    features = batch[job.X_cols]
                    features = job.X.convert_2_ndarray(features, job.X_cols)
                    predicted_values = clf.predict(features)

                    if not job.skipSavePrediction:
                        self._writeBack(predicted_values, job)

                    status_count += 1
                    pbar.n = status_count
                    pbar.refresh()

        return job
