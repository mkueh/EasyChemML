from EasyChemML.DataImport.DataImporter import DataImporter
from EasyChemML.DataImport.Module.XLSX import XLSX
from EasyChemML.Encoder.impl_RdkitConverter.MolRdkitConverter import MolRdkitConverter
from EasyChemML.Environment import Environment

from EasyChemML.Encoder.DEPR_EvoFP import EvoFP
from EasyChemML.Encoder.impl_EvoFP.EVOFingerprint_Enum import FeatureTyp
from EasyChemML.JobSystem.JobFactory.JobFactory import Job_Factory
from EasyChemML.JobSystem.JobFactory.Module.Jobs import ModelTrainEvalJob
from EasyChemML.JobSystem.Utilities.Config import Config
from EasyChemML.Metrik.MetricStack import MetricStack

from EasyChemML.Metrik.Module.R2_Score import R2_Score
from EasyChemML.JobSystem.Runner.Module.LocalRunner import LocalRunner

from EasyChemML.Splitter.Module.RangeSplitter import RangeSplitter
from EasyChemML.Splitter.Splitcreator import Splitcreator
from EasyChemML.Utilities.Dataset import Dataset

from EasyChemML.Model.CatBoost_r import CatBoost_r

feature_typ = FeatureTyp.match_feature
thread_count = 50


env = Environment()

load_dataset = {}
load_dataset['qm9'] = XLSX('/dataset/Evo_Eval_Datasets/QM9_Homo_Lumo_Gaps.xlsx', sheet_name='homo-lumo-qm9',
                           columns=['SMILES', 'Output'])
load_dataset['doyle_Test1'] = XLSX('/dataset/Evo_Eval_Datasets/Dreher_and_Doyle_input_data.xlsx', sheet_name='Test1',
                           columns=['Ligand','Additive','Base','Aryl halide','Output'])
load_dataset['doyle_Test2'] = XLSX('/dataset/Evo_Eval_Datasets/Dreher_and_Doyle_input_data.xlsx', sheet_name='Test2',
                           columns=['Ligand','Additive','Base','Aryl halide','Output'])
load_dataset['doyle_Test3'] = XLSX('/dataset/Evo_Eval_Datasets/Dreher_and_Doyle_input_data.xlsx', sheet_name='Test3',
                           columns=['Ligand','Additive','Base','Aryl halide','Output'])
load_dataset['doyle_Test4'] = XLSX('/dataset/Evo_Eval_Datasets/Dreher_and_Doyle_input_data.xlsx', sheet_name='Test4',
                           columns=['Ligand','Additive','Base','Aryl halide','Output'])
load_dataset['doyle_Plates1-3'] = XLSX('/dataset/Evo_Eval_Datasets/Dreher_and_Doyle_input_data.xlsx', sheet_name='Plates1-3',
                           columns=['Ligand','Additive','Base','Aryl halide','Output'])
load_dataset['doyle_Plate2_new'] = XLSX('/dataset/Evo_Eval_Datasets/Dreher_and_Doyle_input_data.xlsx', sheet_name='Plate2_new',
                           columns=['Ligand','Additive','Base','Aryl halide','Output'])


di = DataImporter(env)
dh = di.load_data_InNewBatchPartition(load_dataset)
job_factory = Job_Factory(env)
job_runner = LocalRunner(env)

MolRdkitConverter().convert(datatable=dh['qm9'], columns=['SMILES'], n_jobs=thread_count)
MolRdkitConverter().convert(datatable=dh['doyle_Test1'], columns=['Ligand', 'Additive', 'Base', 'Aryl halide'], n_jobs=thread_count)
MolRdkitConverter().convert(datatable=dh['doyle_Test2'], columns=['Ligand', 'Additive', 'Base', 'Aryl halide'], n_jobs=thread_count)
MolRdkitConverter().convert(datatable=dh['doyle_Test3'], columns=['Ligand', 'Additive', 'Base', 'Aryl halide'], n_jobs=thread_count)
MolRdkitConverter().convert(datatable=dh['doyle_Test4'], columns=['Ligand', 'Additive', 'Base', 'Aryl halide'], n_jobs=thread_count)
MolRdkitConverter().convert(datatable=dh['doyle_Plates1-3'], columns=['Ligand', 'Additive', 'Base', 'Aryl halide'], n_jobs=thread_count)
MolRdkitConverter().convert(datatable=dh['doyle_Plate2_new'], columns=['Ligand', 'Additive', 'Base', 'Aryl halide'], n_jobs=thread_count)

print('--------------------------------- MOL CONVERTER FINISHED -----------------------------------------------')

evo_fp_modul = EvoFP()
smart_fp = evo_fp_modul.load_fingerprint('1024-count.fp')
evo_fp_modul.convert(smart_fp=smart_fp, datatable=dh['qm9'], columns=['SMILES'], n_jobs=thread_count,
                     feature_typ=feature_typ)

print('--------------------------------- QM9 FINISHED -----------------------------------------------')

evo_fp_modul.convert(smart_fp=smart_fp, datatable=dh['doyle_Test1'], columns=['Ligand', 'Additive', 'Base', 'Aryl halide'], n_jobs=thread_count,
                     feature_typ=feature_typ)

print('--------------------------------- DOYLE TEST1 FINISHED -----------------------------------------------')

evo_fp_modul.convert(smart_fp=smart_fp, datatable=dh['doyle_Test2'], columns=['Ligand', 'Additive', 'Base', 'Aryl halide'], n_jobs=thread_count,
                     feature_typ=feature_typ)

print('--------------------------------- DOYLE TEST2 FINISHED -----------------------------------------------')

evo_fp_modul.convert(smart_fp=smart_fp, datatable=dh['doyle_Test3'], columns=['Ligand', 'Additive', 'Base', 'Aryl halide'], n_jobs=thread_count,
                     feature_typ=feature_typ)

print('--------------------------------- DOYLE TEST3 FINISHED -----------------------------------------------')

evo_fp_modul.convert(smart_fp=smart_fp, datatable=dh['doyle_Test4'], columns=['Ligand', 'Additive', 'Base', 'Aryl halide'], n_jobs=thread_count,
                     feature_typ=feature_typ)

print('--------------------------------- DOYLE TEST4 FINISHED -----------------------------------------------')

evo_fp_modul.convert(smart_fp=smart_fp, datatable=dh['doyle_Plates1-3'], columns=['Ligand', 'Additive', 'Base', 'Aryl halide'], n_jobs=thread_count,
                     feature_typ=feature_typ)

print('--------------------------------- DOYLE PLATES1-3 FINISHED -----------------------------------------------')

evo_fp_modul.convert(smart_fp=smart_fp, datatable=dh['doyle_Plate2_new'], columns=['Ligand', 'Additive', 'Base', 'Aryl halide'], n_jobs=thread_count,
                     feature_typ=feature_typ)

print('--------------------------------- DOYLE PLATES2 FINISHED -----------------------------------------------')

splitter_qm9_CV1 = RangeSplitter(0, 26777)
splitter_qm9_CV2 = RangeSplitter(26777, 52554)
splitter_qm9_CV3 = RangeSplitter(52554, 80331)
splitter_qm9_CV4 = RangeSplitter(80331, 107108)
splitter_qm9_CV5 = RangeSplitter(107108, 133885)

splitter_doyle_Test1 = RangeSplitter(3057, 3955)
splitter_doyle_Test2 = RangeSplitter(3055, 3955)
splitter_doyle_Test3 = RangeSplitter(3058, 3955)
splitter_doyle_Test4 = RangeSplitter(3055, 3955)
splitter_doyle_Platte1 = RangeSplitter(0, 1075)
splitter_doyle_Platte2 = RangeSplitter(1075, 2515)
splitter_doyle_Platte3 = RangeSplitter(2515, 3955)
splitter_doyle_Platte2New = RangeSplitter(1075, 2515)

split_creator = Splitcreator()

splitset_qm9_CV1 = split_creator.generate_split(dh['qm9'], splitter_qm9_CV1)
splitset_qm9_CV2 = split_creator.generate_split(dh['qm9'], splitter_qm9_CV2)
splitset_qm9_CV3 = split_creator.generate_split(dh['qm9'], splitter_qm9_CV3)
splitset_qm9_CV4 = split_creator.generate_split(dh['qm9'], splitter_qm9_CV4)
splitset_qm9_CV5 = split_creator.generate_split(dh['qm9'], splitter_qm9_CV5)

splitset_doyle_Test1 = split_creator.generate_split(dh['doyle_Test1'], splitter_qm9_CV1)
splitset_doyle_Test2 = split_creator.generate_split(dh['doyle_Test2'], splitter_qm9_CV2)
splitset_doyle_Test3 = split_creator.generate_split(dh['doyle_Test3'], splitter_qm9_CV3)
splitset_doyle_Test4 = split_creator.generate_split(dh['doyle_Test4'], splitter_qm9_CV4)
splitset_doyle_Platte1 = split_creator.generate_split(dh['doyle_Plates1-3'], splitter_qm9_CV5)
splitset_doyle_Platte2 = split_creator.generate_split(dh['doyle_Plates1-3'], splitter_qm9_CV1)
splitset_doyle_Platte3 = split_creator.generate_split(dh['doyle_Plates1-3'], splitter_qm9_CV2)
splitset_doyle_Platte2New = split_creator.generate_split(dh['doyle_Plate2_new'], splitter_qm9_CV3)

process_datasets_qm9 = []
dataset_qm9_CV1 = Dataset(dh['qm9'], name='qm9_CV1', feature_col=['SMILES'], target_col=['Output'],
                      split=splitset_qm9_CV1, env=env)
process_datasets_qm9.append(dataset_qm9_CV1)
dataset_qm9_CV2 = Dataset(dh['qm9'], name='qm9_CV2', feature_col=['SMILES'], target_col=['Output'],
                          split=splitset_qm9_CV2, env=env)
process_datasets_qm9.append(dataset_qm9_CV2)
dataset_qm9_CV3 = Dataset(dh['qm9'], name='qm9_CV3', feature_col=['SMILES'], target_col=['Output'],
                       split=splitset_qm9_CV3, env=env)
process_datasets_qm9.append(dataset_qm9_CV3)
dataset_qm9_CV4 = Dataset(dh['qm9'], name='qm9_CV4', feature_col=['SMILES'], target_col=['Output'],
                      split=splitset_qm9_CV4, env=env)
process_datasets_qm9.append(dataset_qm9_CV4)
dataset_qm9_CV5 = Dataset(dh['qm9'], name='qm9_CV5', feature_col=['SMILES'], target_col=['Output'],
                      split=splitset_qm9_CV5, env=env)
process_datasets_qm9.append(dataset_qm9_CV5)

process_datasets_doyle = []
dataset_doyle_Test1 = Dataset(dh['doyle_Test1'], name='doyle_Test1', feature_col=['Ligand','Additive','Base','Aryl halide'], target_col=['Output'],
                      split=splitset_doyle_Test1, env=env)
process_datasets_doyle.append(dataset_doyle_Test1)
dataset_doyle_Test2 = Dataset(dh['doyle_Test2'], name='doyle_Test2', feature_col=['Ligand','Additive','Base','Aryl halide'], target_col=['Output'],
                          split=splitset_doyle_Test2, env=env)
process_datasets_doyle.append(dataset_doyle_Test2)
dataset_doyle_Test3 = Dataset(dh['doyle_Test3'], name='doyle_Test3', feature_col=['Ligand','Additive','Base','Aryl halide'], target_col=['Output'],
                       split=splitset_doyle_Test3, env=env)
process_datasets_doyle.append(dataset_doyle_Test3)
dataset_doyle_Test4 = Dataset(dh['doyle_Test4'], name='doyle_Test4', feature_col=['Ligand','Additive','Base','Aryl halide'], target_col=['Output'],
                      split=splitset_doyle_Test4, env=env)
process_datasets_doyle.append(dataset_doyle_Test4)
dataset_doyle_Platte1 = Dataset(dh['doyle_Plates1-3'], name='doyle_Plates1', feature_col=['Ligand','Additive','Base','Aryl halide'], target_col=['Output'],
                      split=splitset_doyle_Platte1, env=env)
process_datasets_doyle.append(dataset_doyle_Platte1)
dataset_doyle_Platte2 = Dataset(dh['doyle_Plates1-3'], name='doyle_Plates2', feature_col=['Ligand','Additive','Base','Aryl halide'], target_col=['Output'],
                      split=splitset_doyle_Platte2, env=env)
process_datasets_doyle.append(dataset_doyle_Platte2)
dataset_doyle_Platte3 = Dataset(dh['doyle_Plates1-3'], name='doyle_Plates3', feature_col=['Ligand','Additive','Base','Aryl halide'], target_col=['Output'],
                          split=splitset_doyle_Platte3, env=env)
process_datasets_doyle.append(dataset_doyle_Platte3)
dataset_doyle_Platte2New = Dataset(dh['doyle_Plate2_new'], name='doyle_Plate2_new', feature_col=['Ligand','Additive','Base', 'Aryl halide'], target_col=['Output'],
                       split=splitset_doyle_Platte2New, env=env)
process_datasets_doyle.append(dataset_doyle_Platte2New)

for dataset_qm9 in process_datasets_qm9:
    r2score = R2_Score()
    metricStack_r = MetricStack({'r2': r2score})

    catboost_r_QM9 = Config(CatBoost_r, {'verbose': False, 'thread_count': thread_count, 'depth': 12, 'iterations': 50000, 'allow_writing_files':False})

    job: ModelTrainEvalJob = job_factory.create_ModelTrainEvalJob(dataset_qm9.name, dataset_qm9, catboost_r_QM9, metricStack_r,
                                                                  dataset_qm9.get_Splitset().get_outer_split(0))
    job_runner.run_Job(job)

    print(job.result_metric_TEST['r2'])
    print(job.result_metric_TRAIN['r2'])

for dataset_doyle in process_datasets_doyle:
    r2score = R2_Score()
    metricStack_r = MetricStack({'r2': r2score})

    catboost_r_doyle = Config(CatBoost_r, {'verbose': False, 'thread_count': thread_count, 'allow_writing_files': False})

    job: ModelTrainEvalJob = job_factory.create_ModelTrainEvalJob(dataset_doyle.name, dataset_doyle, catboost_r_doyle,
                                                                  metricStack_r,
                                                                  dataset_doyle.get_Splitset().get_outer_split(0))
    job_runner.run_Job(job)

    print(job.result_metric_TEST['r2'])
    print(job.result_metric_TRAIN['r2'])

