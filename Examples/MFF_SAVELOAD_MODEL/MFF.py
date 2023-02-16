from EasyChemML.DataImport.DataImporter import DataImporter
from EasyChemML.DataImport.Module.XLSX import XLSX
from EasyChemML.Environment import Environment

from EasyChemML.Encoder.MFF import MFF
from EasyChemML.Encoder.impl_RdkitConverter.MolRdkitConverter import MolRdkitConverter
from EasyChemML.JobSystem.JobFactory.JobFactory import Job_Factory
from EasyChemML.JobSystem.JobFactory.Module.Jobs import ModelPredictJob, ModelTrainEvalJob
from EasyChemML.JobSystem.Runner.Module.LocalRunner import LocalRunner
from EasyChemML.JobSystem.Utilities.Config import Config
from EasyChemML.Metrik.MetricStack import MetricStack
from EasyChemML.Metrik.Module.MeanAbsoluteError import MeanAbsoluteError
from EasyChemML.Metrik.Module.R2_Score import R2_Score
from EasyChemML.Model.CatBoost_r import CatBoost_r

from EasyChemML.Splitter.Module.RangeSplitter import RangeSplitter
from EasyChemML.Splitter.Splitcreator import Splitcreator
from EasyChemML.Utilities.Dataset import Dataset

env = Environment()

# fp = SMART_Fingerprint.load('8-Count.fp')

load_dataset = {}
load_dataset['dreher'] = XLSX('../_DATASETS/Dreher_and_Doyle_input_data.xlsx', sheet_name='Test3',
                              columns=['Ligand', 'Additive', 'Base', 'Aryl halide', 'Output'], range=(0, 1500))

di = DataImporter(env)
dh = di.load_data_InNewBatchPartition(load_dataset)

splitter_outer = RangeSplitter(0, 150)
split_creator = Splitcreator()
splitset_multi = split_creator.generate_split(dh['dreher'], splitter_outer)

mol_converter = MolRdkitConverter()
mol_converter.convert(dh['dreher'], ['Ligand', 'Additive', 'Base', 'Aryl halide'], 12)

mff_converter = MFF()
mff_converter.convert(dh['dreher'], ['Ligand', 'Additive', 'Base', 'Aryl halide'], 1, 64, return_nonZero_indices=True)

dreher_dataset = Dataset(dh['dreher'], name='dreher_Test3', feature_col=['Ligand', 'Additive', 'Base', 'Aryl halide'],
                         split=splitset_multi)

job_factory = Job_Factory(env)
job_runner = LocalRunner(env)

r2score = R2_Score()
mae = MeanAbsoluteError()
metricStack = MetricStack({'r2': r2score, 'mae':mae})

catboost_model = Config(CatBoost_r, {'verbose': 1, 'thread_count': 12, 'n_estimators': 5})
job: ModelTrainEvalJob = job_factory.create_job_modelCalculation('test_train',dreher_dataset, catboost_model, metricStack,
                                                                 dreher_dataset.get_Splitset().
                                                                 get_outer_split(0))
job_runner.run_Job(job)

print(f'TEST: {job.result_metric_TEST}')
print(f'TRAIN: {job.result_metric_TRAIN}')

#job.trained_Model.save_model('saved_model.model')
loaded_model = CatBoost_r()
loaded_model.load_model('saved_model.model')
job_prediction = ModelPredictJob('Test_predicition', loaded_model, dh['dreher'], ['Ligand', 'Additive', 'Base', 'Aryl halide'],
                                 dreher_dataset.get_Splitset().get_outer_split(0).test)

print('test')
