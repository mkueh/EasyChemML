from datetime import datetime

from EasyChemML.DataImport.DataImporter import DataImporter
from EasyChemML.DataImport.Module.XLSX import XLSX
from EasyChemML.Encoder import MolRdkitConverter, MFF
from EasyChemML.Encoder.RandomFeatureEncoder import RandomFeatureEncoder
from EasyChemML.Environment import Environment
from EasyChemML.JobSystem.JobFactory.JobFactory import Job_Factory
from EasyChemML.JobSystem.JobFactory.Module.Jobs import ModelTrainJob, ModelEvalJob
from EasyChemML.JobSystem.Utilities.Config import Config
from EasyChemML.Metrik.MetricStack import MetricStack
from EasyChemML.Metrik.Module.R2_Score import R2_Score
from EasyChemML.JobSystem.Runner.Module.LocalRunner import LocalRunner
from EasyChemML.Model.CatBoost_r import CatBoost_r
from EasyChemML.Preprocessing.Module.SklearnPCA import SklearnPCA
from EasyChemML.Splitter.Module.ShuffleSplitter import ShuffleSplitter
from EasyChemML.Splitter.Splitcreator import Splitcreator
from EasyChemML.Utilities.Dataset import Dataset

env = Environment()
n_jobs = 72
# ------------------------------------------------------------------------------------------------------------------
start_time = datetime.now()
load_dataset = {}
load_dataset['doyle'] = XLSX('../_DATASETS/Dreher_and_Doyle_input_data.xlsx', sheet_name='FullCV_01',
                             columns=['Ligand', 'Additive', 'Base', 'Aryl halide', 'Output'])

di = DataImporter(env)
dh = di.load_data_InNewBatchPartition(load_dataset)

time_elapsed = datetime.now() - start_time
print('LOADING DATA TAKES (hh:mm:ss.ms) {}'.format(time_elapsed), flush=True)
# ------------------------------------------------------------------------------------------------------------------

print('Start MolRdkitConverter')
molRdkit_converter = MolRdkitConverter()
molRdkit_converter.convert(dh['doyle'], columns=['Ligand', 'Additive', 'Base', 'Aryl halide'], n_jobs=10)

print('Start FingerprintEncoder')
# fingerprints = [FingerprintHolder(FingerprintTyp.ECFC, {'length': 2048, 'radius': 4})]
# ecfp_encoder = FingerprintEncoder()
# ecfp_encoder.convert(X=bp['bp_dataset'], columns=['SMILES_FP'], n_jobs=64, fingerprints=fingerprints)

mff_encoder = MFF()
mff_encoder.convert(datatable=dh['doyle'], columns=['Ligand', 'Additive', 'Base', 'Aryl halide'], fp_length=256,
                    n_jobs=64)

# -------------------------------------------------------------------------------------------------------------------

print('--- pca on mff vector', flush=True)
start_time = datetime.now()

pca = SklearnPCA()
pca.convert(dh['doyle'], ['Ligand', 'Additive', 'Base', 'Aryl halide'], n_jobs)

time_elapsed = datetime.now() - start_time
print('convert SMILES to mol-object TAKES (hh:mm:ss.ms) {}'.format(time_elapsed), flush=True)
print(dh['doyle'].to_String(rows=10))

# -------------------------------------------------------------------------------------------------------------------

print('--- TRAIN Catboost_r with train', flush=True)
start_time = datetime.now()

split_creator = Splitcreator()
splitter_outer = ShuffleSplitter(1, 42, test_size=0.1)
splitset = split_creator.generate_split(dh['doyle'], splitter_outer)

train_dataset = Dataset(dh['doyle'], name='train', feature_col=['Ligand', 'Additive', 'Base', 'Aryl halide'], target_col=['Output'],
                        split=splitset, env=env)

job_factory = Job_Factory(env)
job_runner = LocalRunner(env)

catboost_model = Config(CatBoost_r, {'verbose': False, 'thread_count': n_jobs})
job: ModelTrainJob = job_factory.create_ModelTrainJob('DOYLE_STD_ECFP_4_1024', train_dataset, catboost_model,
                                                      train_dataset.get_Splitset().
                                                      get_outer_split(0))
job_runner.run_Job(job)

job.trained_Model.save_model('DOYLE_STD_ECFP_4_1024_trained.model')
time_elapsed = datetime.now() - start_time
print('TRAIN Catboost_r with train TAKES (hh:mm:ss.ms) {}'.format(time_elapsed), flush=True)

# -------------------------------------------------------------------------------------------------------------------

print('--- Predicting Test', flush=True)
start_time = datetime.now()

metricStack_r = MetricStack(
    {'r2': R2_Score()})

loaded_model = CatBoost_r()
loaded_model.load_model('DOYLE_STD_ECFP_4_1024_trained.model')
job_prediction = ModelEvalJob('DOYLE_STD_ECFP_4_1024_test', loaded_model, dh['doyle'], ['Ligand', 'Additive', 'Base', 'Aryl halide'], dh['doyle'],
                              ['Output'], metricStack_r)
results = job_runner.run_Job(job_prediction)
print(results.result_metric)
time_elapsed = datetime.now() - start_time
print('Predicting Test TAKES (hh:mm:ss.ms) {}'.format(time_elapsed), flush=True)
