import logging, datetime
from typing import Union

from EasyChemML.Environment import Environment
from EasyChemML.JobSystem.JobFactory.Module.Jobs import ModelTrainJob

from EasyChemML.Model.AbstractModel import Abstract_Model, WithEpochs, WithBatches, WithCheckpoints
from EasyChemML.JobSystem.CheckpointSystem.CheckpointSystem import CheckpointSystem


class RunnerModelTrainJob:

    def __init__(self, environment: Environment):
        self.environment = environment

    def run_job(self, job: ModelTrainJob) -> ModelTrainJob:
        clf_class = job.algorithm

        if not issubclass(clf_class, (WithEpochs, WithBatches, WithCheckpoints)):
            return self._runNaiv_ModelTrainJob(job)
        elif issubclass(clf_class, (WithEpochs, WithBatches, WithCheckpoints)):
            return self._runEpochBatches_ModelTrainJob(job)
        else:
            raise Exception('not implemented yet')

    def _runNaiv_ModelTrainJob(self, job: ModelTrainJob) -> ModelTrainJob:
        clf: Abstract_Model = job.algorithm(job.explicit_algorithm_para)

        logging.info(str(job.algorithm) + " is started")
        logging.info('Time: ' + str(datetime.datetime.now()))

        X_train = job.X.convert_2_ndarray(job.split.train, job.X_cols)
        y_train = job.y.convert_2_ndarray(job.split.train, job.y_cols)

        clf.fit(X=X_train, y=y_train)

        job.set_result_trained_Model(clf)
        return job

    def _runEpochBatches_ModelTrainJob(self, job: ModelTrainJob) -> ModelTrainJob:
        clf: Union[Abstract_Model, WithBatches, WithEpochs, WithCheckpoints] = job.algorithm(
            **job.explicit_algorithm_para)
        checksystem = self.environment.CheckpointSystem

        batchSize = clf.getBatchsize()
        epochs = clf.getEpochs()
        checkpoints_afterIterations = clf.getCheckpointsAfterIterations()
        batchcountPerEpoch = CheckpointSystem.calcBatchcountPerEpoch(batchSize, len(job.split.train))
        current_epoch = 0
        current_batch = 0

        if checksystem.checkForCheckpoints(job.job_name):
            loaded_checkpoint = checksystem.loadLastCheckpoint(job.job_name)
            print(f'Found a Checkpoint {loaded_checkpoint} for the jobname: {job.job_name}')
            meta_data, model_data = checksystem.loadDataOfCheckpoint(loaded_checkpoint)
            current_epoch = meta_data['Epoch']
            current_batch = meta_data['Batch'] + 1
            print(f'Resume at Epoch: {current_epoch} and Batch: {current_batch}')
            clf.setCurrentState(model_data)

        logging.info(str(job.algorithm) + " is started")
        logging.info('Time: ' + str(datetime.datetime.now()))

        absolute_iteration: int = (current_epoch * batchcountPerEpoch) + current_batch
        for epochStep in range(current_epoch, epochs):
            for batchStep, trainBatchIndices in enumerate(job.split.getTrainIterator(batchSize, skip=current_batch), start=current_batch):
                X_train = job.X.convert_2_ndarray(trainBatchIndices, job.X_cols)
                y_train = job.X.convert_2_ndarray(trainBatchIndices, job.y_cols)

                loss = clf.fit(X=X_train, y=y_train)
                print(f'current: Epoch {epochStep} | Batch {batchStep} | loss: {loss}')
                absolute_iteration += 1

                if checksystem.checkCheckpointNeeded(absolute_iteration, checkpoints_afterIterations):
                    metaData = {}
                    metaData['Epoch'] = epochStep
                    metaData['Batch'] = batchStep
                    metaData['Jobname'] = job.job_name
                    metaData['CurrentLoss'] = loss

                    checkpointName = job.job_name + f'_AbsIteration{absolute_iteration}'
                    checksystem.createCheckpoint(clf.getCurrentState(), metaData, checkpointName, job.job_name)
            current_batch = 0

        job.set_result_trained_Model(clf)
        return job
