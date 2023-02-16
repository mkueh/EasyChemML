from datetime import datetime

from EasyChemML.DataImport.DataImporter import DataImporter
from EasyChemML.DataImport.Module.XLSX import XLSX
from EasyChemML.Encoder.FingerprintEncoder import FingerprintEncoder, FingerprintHolder, FingerprintTyp
from EasyChemML.Encoder.impl_RdkitConverter.MolRdkitConverter import MolRdkitConverter
from EasyChemML.Environment import Environment
from EasyChemML.JobSystem.JobFactory.JobFactory import Job_Factory
from EasyChemML.JobSystem.JobFactory.Module.Jobs import ModelTrainJob, ModelEvalJob
from EasyChemML.JobSystem.Utilities.Config import Config
from EasyChemML.Metrik.MetricStack import MetricStack
from EasyChemML.Metrik.Module.AccuracyScore import AccuracyScore
from EasyChemML.Metrik.Module.F1_Score import F1_Score
from EasyChemML.Metrik.Module.RocAucScore import RocAucScore
from EasyChemML.Metrik.Module.RecallScore import RecallScore
from EasyChemML.Metrik.Module.PrecisionScore import PrecisionScore
from EasyChemML.Model.CatBoost_c import CatBoost_c
from EasyChemML.JobSystem.Runner.Module.LocalRunner import LocalRunner
from EasyChemML.Splitter.Module.AllTrainSplitter import AllTrainSplitter
from EasyChemML.Splitter.Splitcreator import Splitcreator
from EasyChemML.Utilities.Dataset import Dataset

env = Environment()
n_jobs = 72
# ------------------------------------------------------------------------------------------------------------------
start_time = datetime.now()
load_dataset = {}
load_dataset['train'] = XLSX('HIV_FKB.xlsx', sheet_name='ext_train_set',
                             columns=['x', 'y'])
load_dataset['test'] = XLSX('HIV_FKB.xlsx', sheet_name='test_set',
                            columns=['x', 'y'])

di = DataImporter(env)
dh = di.load_data_InNewBatchPartition(load_dataset)

time_elapsed = datetime.now() - start_time
print('LOADING DATA TAKES (hh:mm:ss.ms) {}'.format(time_elapsed), flush=True)
# ------------------------------------------------------------------------------------------------------------------

print('--- convert SMILES to mol-object', flush=True)
start_time = datetime.now()

mr_converter = MolRdkitConverter()
mr_converter.convert(dh['train'], ['x'], n_jobs)
mr_converter.convert(dh['test'], ['x'], n_jobs)

time_elapsed = datetime.now() - start_time
print('convert SMILES to mol-object TAKES (hh:mm:ss.ms) {}'.format(time_elapsed), flush=True)

# ------------------------------------------------------------------------------------------------------------------

print('--- convert mol-objects to fingerprints', flush=True)
start_time = datetime.now()

fingerprints = [FingerprintHolder(FingerprintTyp.ECFP, {'length': 1024, 'radius': 2})]

ecfp_encoder = FingerprintEncoder()
ecfp_encoder.convert(datatable=dh['train'], columns=['x'], n_jobs=n_jobs, fingerprints=fingerprints)
ecfp_encoder.convert(datatable=dh['test'], columns=['x'], n_jobs=n_jobs, fingerprints=fingerprints)

time_elapsed = datetime.now() - start_time
print('convert mol-objects to fingerprints TAKES (hh:mm:ss.ms) {}'.format(time_elapsed), flush=True)

# -------------------------------------------------------------------------------------------------------------------

print('--- TRAIN Catboost_r with train', flush=True)
start_time = datetime.now()

splitter_outer = AllTrainSplitter()
split_creator = Splitcreator()
splitset = split_creator.generate_split(dh['train'], splitter_outer)

train_dataset = Dataset(dh['train'], name='train', feature_col=['x'], target_col=['y'],
                        split=splitset, env=env)

job_factory = Job_Factory(env)
job_runner = LocalRunner(env)

catboost_model = Config(CatBoost_c, {'verbose': False, 'thread_count': n_jobs})
job: ModelTrainJob = job_factory.create_ModelTrainJob('HIV_STD_ECFP_4_1024', train_dataset, catboost_model,
                                                      train_dataset.get_Splitset().
                                                      get_outer_split(0))
job_runner.run_Job(job)

job.result_trained_Model.save_model('HIV_STD_ECFP_4_1024_trained.model')
time_elapsed = datetime.now() - start_time
print('TRAIN Catboost_r with train TAKES (hh:mm:ss.ms) {}'.format(time_elapsed), flush=True)

# -------------------------------------------------------------------------------------------------------------------

print('--- Predicting Test', flush=True)
start_time = datetime.now()

accuracy = AccuracyScore()
f1score = F1_Score()
recall = RecallScore()
auc_roc = RocAucScore()
precision = PrecisionScore()

metricStack_c = MetricStack(
    {'accuracy': accuracy, 'f1': f1score, 'recall': recall, 'auc_roc': auc_roc, 'precision':precision})

loaded_model = CatBoost_c()
loaded_model.load_model('HIV_STD_ECFP_4_1024_trained.model')
job_prediction = ModelEvalJob('HIV_STD_ECFP_4_1024_test', loaded_model, dh['test'], ['x'], ['y'], metricStack_c)
results = job_runner.run_Job(job_prediction)

time_elapsed = datetime.now() - start_time
print('Predicting Test TAKES (hh:mm:ss.ms) {}'.format(time_elapsed), flush=True)
