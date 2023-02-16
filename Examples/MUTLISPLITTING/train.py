from EasyChemML.DataImport.DataImporter import DataImporter
from EasyChemML.DataImport.Module.XLSX import XLSX
from EasyChemML.Encoder.MFF import MFF
from EasyChemML.Encoder.impl_RdkitConverter.MolRdkitConverter import MolRdkitConverter
from EasyChemML.Environment import Environment
from EasyChemML.JobSystem.JobFactory.JobFactory import Job_Factory
from EasyChemML.JobSystem.JobFactory.Module.Jobs import ModelTrainJob, ModelTrainEvalJob, ModelEvalJob
from EasyChemML.JobSystem.Utilities.Config import Config
from EasyChemML.Metrik.MetricStack import MetricStack
from EasyChemML.Metrik.Module.MeanAbsoluteError import MeanAbsoluteError
from EasyChemML.Metrik.Module.R2_Score import R2_Score
from EasyChemML.Model.CatBoost_r import CatBoost_r
from EasyChemML.Metrik.Module.MeanSquaredError import MeanSquaredError
from EasyChemML.Metrik.Module.ExplainedVarianceScore import ExplainedVarianceScore
from EasyChemML.Metrik.Module.MaxError import MaxError

from EasyChemML.JobSystem.Runner.Module.LocalRunner import LocalRunner
from EasyChemML.Splitter.Module.RangeSplitter import RangeSplitter
from EasyChemML.Splitter.Module.AllTestSplitter import AllTestSplitter
from EasyChemML.Splitter.Module.AllTrainSplitter import AllTrainSplitter
from EasyChemML.Splitter.Splitcreator import Splitcreator
from EasyChemML.Utilities.Dataset import Dataset

env = Environment(WORKING_path_addRelativ='Output')

karmaus_dataset_karmaus = 'karmaus_dataset_karmaus'
karmaus_dataset_random = 'karmaus_dataset_random'
drug_tox_test = "drug_tox_test"

dataloader = {
    karmaus_dataset_karmaus: XLSX('../_DATASETS/Tox_Karmaus.xlsx', sheet_name='Karmaus'),
    karmaus_dataset_random: XLSX('../_DATASETS/Tox_Karmaus.xlsx', sheet_name='Random'),
    drug_tox_test: XLSX('../_DATASETS/Tox_Karmaus.xlsx', sheet_name="Drugs")
}

di = DataImporter(env)
bp = di.load_data_InNewBatchPartition(dataloader, max_chunksize=100000)

print('Start MolRdkitConverter')
molRdkit_converter = MolRdkitConverter()
molRdkit_converter.convert(bp[karmaus_dataset_karmaus], columns=['SMILES'], n_jobs=10)
molRdkit_converter.convert(bp[karmaus_dataset_random], columns=['SMILES'], n_jobs=10)
molRdkit_converter.convert(bp[drug_tox_test], columns=['SMILES Drug'], n_jobs=1)


print('Start FingerprintEncoder')
mff_encoder = MFF()
mff_encoder.convert(datatable=bp[karmaus_dataset_karmaus], columns=['SMILES'], fp_length=1024, n_jobs=20)
mff_encoder.convert(datatable=bp[karmaus_dataset_random], columns=['SMILES'], fp_length=1024, n_jobs=20)
mff_encoder.convert(datatable=bp[drug_tox_test], columns=['SMILES Drug'], fp_length=1024, n_jobs=4)

# ----------------------------------- Training --------------------------------------
split_creator = Splitcreator()
spliter_karmaus_5fold1 = RangeSplitter(0, 3517)
spliter_karmaus_5fold2 = RangeSplitter(3517, 7033)
spliter_karmaus_5fold3 = RangeSplitter(7033, 10549)
spliter_karmaus_5fold4 = RangeSplitter(10549, 14065)
spliter_karmaus_5fold5 = RangeSplitter(14065, 17581)

spliter_karmaus_external_test = RangeSplitter(15289, 17580)

spliter_test_drugs = AllTestSplitter()
spliter_train_drugs = AllTrainSplitter()


#Dataset Splits
splitset_5fold1 = split_creator.generate_split(bp[karmaus_dataset_random], spliter_karmaus_5fold1)
splitset_5fold2 = split_creator.generate_split(bp[karmaus_dataset_random], spliter_karmaus_5fold2)
splitset_5fold3 = split_creator.generate_split(bp[karmaus_dataset_random], spliter_karmaus_5fold3)
splitset_5fold4 = split_creator.generate_split(bp[karmaus_dataset_random], spliter_karmaus_5fold4)
splitset_5fold5 = split_creator.generate_split(bp[karmaus_dataset_random], spliter_karmaus_5fold5)

splitset_karmaus_external_test = split_creator.generate_split(bp[karmaus_dataset_karmaus], spliter_karmaus_external_test)

splitset_drugs_train = split_creator.generate_split(bp[karmaus_dataset_random], spliter_train_drugs)
splitset_drugs_test = split_creator.generate_split(bp[karmaus_dataset_random], spliter_test_drugs)

#Datasets
dataset_5fold1 = Dataset(bp[karmaus_dataset_random],name='5fold1',feature_col=['SMILES'],target_col=['LD50'],split=splitset_5fold1, env=env)
dataset_5fold2 = Dataset(bp[karmaus_dataset_random],name='5fold2',feature_col=['SMILES'],target_col=['LD50'],split=splitset_5fold2, env=env)
dataset_5fold3 = Dataset(bp[karmaus_dataset_random],name='5fold3',feature_col=['SMILES'],target_col=['LD50'],split=splitset_5fold3, env=env)
dataset_5fold4 = Dataset(bp[karmaus_dataset_random],name='5fold4',feature_col=['SMILES'],target_col=['LD50'],split=splitset_5fold4, env=env)
dataset_5fold5 = Dataset(bp[karmaus_dataset_random],name='5fold5',feature_col=['SMILES'],target_col=['LD50'],split=splitset_5fold5, env=env)

dataset_karmaus_external_test = Dataset(bp[karmaus_dataset_random],name='karmaus',feature_col=['SMILES'],target_col=['LD50'],split=splitset_karmaus_external_test, env=env)

dataset_drugs_train = Dataset(bp[karmaus_dataset_random],name='full_train_drugs',feature_col=['SMILES'],target_col=['LD50'],split=splitset_drugs_train, env=env)


#Runner
job_factory = Job_Factory(env)
job_runner = LocalRunner(env)

r2score = R2_Score()
mae = MeanAbsoluteError()
mse = MeanSquaredError()
evs = ExplainedVarianceScore()
maxe = MaxError()
metricStack_r = MetricStack({'r2': r2score, 'mae': mae, "mse":mse, "evs":evs, "maxe": maxe})

catboost_r = Config(
    CatBoost_r,
    {'verbose': 10,
     'thread_count': 40,
     'allow_writing_files': False,
     'iterations': 50000,
     'depth': 12}
)

job1: ModelTrainEvalJob = job_factory.create_ModelTrainEvalJob(
    'dataset_5fold1',
    dataset_5fold1,
    catboost_r,
    metricStack_r,
    dataset_5fold1.get_Split().get_outer_split(0)
)

job2: ModelTrainEvalJob = job_factory.create_ModelTrainEvalJob(
    'dataset_5fold2',
    dataset_5fold2,
    catboost_r,
    metricStack_r,
    dataset_5fold2.get_Split().get_outer_split(0)
)

job3: ModelTrainEvalJob = job_factory.create_ModelTrainEvalJob(
    'dataset_5fold3',
    dataset_5fold3,
    catboost_r,
    metricStack_r,
    dataset_5fold3.get_Split().get_outer_split(0)
)

job4: ModelTrainEvalJob = job_factory.create_ModelTrainEvalJob(
    'dataset_5fold4',
    dataset_5fold4,
    catboost_r,
    metricStack_r,
    dataset_5fold4.get_Split().get_outer_split(0)
)

job5: ModelTrainEvalJob = job_factory.create_ModelTrainEvalJob(
    'dataset_5fold5',
    dataset_5fold5,
    catboost_r,
    metricStack_r,
    dataset_5fold5.get_Split().get_outer_split(0)
)

job6: ModelTrainEvalJob = job_factory.create_ModelTrainEvalJob(
    'dataset_karmaus',
    dataset_karmaus_external_test,
    catboost_r,
    metricStack_r,
    dataset_karmaus_external_test.get_Split().get_outer_split(0)
)

job7: ModelTrainJob = job_factory.create_ModelTrainJob(
    'full_train_drugs',
    dataset_drugs_train,
    catboost_r,
    dataset_drugs_train.get_Split().get_outer_split(0)
)

job_runner.run_Jobs([job1,job2,job3,job4,job5,job6,job7])

job7.trained_Model.save_model('Full_Tox.catb')

job_predict = ModelEvalJob(job_name='Test_on_drugs', trained_Model=job7.trained_Model, datatable=bp[drug_tox_test], features_cols=['SMILES Drug'], target_cols =['Tox_Fake'], metric =metricStack_r)
job_runner.run_Job(job_predict)

env.clean()

print("Woop")
