from EasyChemML.DataImport.DataImporter import DataImporter
from EasyChemML.DataImport.Module.XLSX import XLSX
from EasyChemML.Encoder.MFF import MFF
from EasyChemML.Encoder.impl_RdkitConverter.MolRdkitConverter import MolRdkitConverter
from EasyChemML.Environment import EasyProjectEnvironment
from EasyChemML.JobSystem.JobFactory.JobFactory import Job_Factory
from EasyChemML.JobSystem.JobFactory.Module.Jobs import ModelTrainJob, ModelTrainEvalJob, ModelPredictJob
from EasyChemML.JobSystem.Utilities.Config import Config
from EasyChemML.Metrik.MetricStack import MetricStack
from EasyChemML.Metrik.Module.MeanAbsoluteError import MeanAbsoluteError
from EasyChemML.Metrik.Module.R2_Score import R2_Score
from EasyChemML.Model.CatBoost_r import CatBoost_r

from EasyChemML.JobSystem.Runner.Module.LocalRunner import LocalRunner
from EasyChemML.Splitter.Module.RangeSplitter import RangeSplitter
from EasyChemML.Splitter.Module.ShuffleSplitter import ShuffleSplitter
from EasyChemML.Splitter.Splitcreator import Splitcreator
from EasyChemML.Utilities.Dataset import Dataset

env = EasyProjectEnvironment('TestFolder')
step_size = 100000
threads = 30

zinc20_hdfLoader = {'dreher_dataset': XLSX('Examples/_DATASETS/Dreher_and_Doyle_input_data.xlsx', sheet_name='FullCV_01')}
di = DataImporter(env)
bp = di.load_data_InNewBatchPartition(zinc20_hdfLoader, max_chunksize=100000)

print('Start MolRdkitConverter')
molRdkit_converter = MolRdkitConverter()
molRdkit_converter.convert(bp['dreher_dataset'], columns=['Ligand', 'Additive', 'Base', 'Aryl halide'], n_jobs=10)

print('Start FingerprintEncoder')
# fingerprints = [FingerprintHolder(FingerprintTyp.ECFC, {'length': 2048, 'radius': 4})]
# ecfp_encoder = FingerprintEncoder()
# ecfp_encoder.convert(X=bp['bp_dataset'], columns=['SMILES_FP'], n_jobs=64, fingerprints=fingerprints)

mff_encoder = MFF()
mff_encoder.convert(datatable=bp['dreher_dataset'], columns=['Ligand', 'Additive', 'Base', 'Aryl halide'], fp_length=16,
                    n_jobs=64)
# ----------------------------------- Training --------------------------------------
# ----------------------------------- Training --------------------------------------

split_creator = Splitcreator()
splitter_boilingpoint = ShuffleSplitter(1, 42, test_size=0.1)
splitset_boilingpoint = split_creator.generate_split(bp['dreher_dataset'], splitter_boilingpoint)

dataset_boilingpoint = Dataset(bp['dreher_dataset'],
                               name='dreher_dataset',
                               feature_col=['Ligand', 'Additive', 'Base', 'Aryl halide'],
                               target_col=['Output'],
                               split=splitset_boilingpoint, env=env)

job_factory = Job_Factory(env)
job_runner = LocalRunner(env)

r2score = R2_Score()
mae = MeanAbsoluteError()
metricStack_r = MetricStack({'r2': r2score, 'mae': mae})

catboost_r = Config(
    CatBoost_r,
    {'verbose': 50,
     'thread_count': 64,
     'allow_writing_files': False,
     'iterations': 100,
     'depth': 4}
)

job: ModelTrainEvalJob = job_factory.create_ModelTrainEvalJob(
    'dataset_boilingpoint_0',
    dataset_boilingpoint,
    catboost_r,
    metricStack_r,
    dataset_boilingpoint.get_Splitset().
    get_outer_split(0)
)

job_runner.run_Job(job)

print(f'Test_lipo: {job.result_metric_TEST}')
print(f'Train_lipo: {job.result_metric_TRAIN}')

job.trained_Model.save_model('model_reaxys.catb')

job_factory = Job_Factory(env)
job_runner = LocalRunner(env)

loaded_model = CatBoost_r()
loaded_model.load_model(path='model_reaxys.catb')

X_indices = list(range(len(bp['dreher_dataset'])))
job_predict = ModelPredictJob(job_name='Predict', trained_Model=loaded_model, X=bp['dreher_dataset'],
                              X_cols=['Ligand', 'Additive', 'Base', 'Aryl halide'])
job_runner.run_Job(job_predict)

for val in job_predict.predicted_vals:
    print(str(val))

env.clean()
