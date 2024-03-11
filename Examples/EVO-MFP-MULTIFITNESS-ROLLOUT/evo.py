import glob, os

from EasyChemML.DataImport.DataImporter import DataImporter
from EasyChemML.DataImport.Module.XLSX import XLSX
from EasyChemML.Encoder.impl_RdkitConverter.MolRdkitConverter import MolRdkitConverter
from EasyChemML.Environment import Environment

from EasyChemML.Encoder.DEPR_EvoFP import EvoFP
from EasyChemML.Encoder.impl_EvoFP.EVOFingerprint_Enum import FeatureTyp
from EasyChemML.JobSystem.JobFactory.JobFactory import Job_Factory
from EasyChemML.JobSystem.JobFactory.Module.Jobs import ModelTrainEvalJob
from EasyChemML.JobSystem.Runner.Module.LocalRunner import LocalRunner
from EasyChemML.JobSystem.Utilities.Config import Config
from EasyChemML.Metrik.MetricStack import MetricStack
from EasyChemML.Metrik.Module.AccuracyScore import AccuracyScore

from EasyChemML.Metrik.Module.F1_Score import F1_Score
from EasyChemML.Metrik.Module.MeanAbsoluteError import MeanAbsoluteError
from EasyChemML.Metrik.Module.R2_Score import R2_Score

from EasyChemML.Splitter.Module.RangeSplitter import RangeSplitter
from EasyChemML.Splitter.Splitcreator import Splitcreator
from EasyChemML.Utilities.Dataset import Dataset

from EasyChemML.Model.CatBoost_r import CatBoost_r
from EasyChemML.Model.CatBoost_c import CatBoost_c

fingerprints_files = glob.glob('./fingerprints/*.fp')
thread_count = 12

for fingerprint_file in fingerprints_files:
    head, tail = os.path.split(fingerprint_file)

    if 'match' in tail:
        feature_typ = FeatureTyp.match_feature
    elif 'count' in tail:
        feature_typ = FeatureTyp.count_feature
    else:
        raise Exception('not found')

    env = Environment(WORKING_path=tail + '_Output')

    load_dataset = {}
    load_dataset['lipo'] = XLSX('../_DATASETS/Lipophilicity.xlsx', sheet_name='Sheet1',
                                columns=['Smiles', 'Output'])
    load_dataset['tox'] = XLSX('../_DATASETS/Tox_Karmaus.xlsx', sheet_name='Random',
                               columns=['SMILES', 'LD50'])
    load_dataset['oe62'] = XLSX('../_DATASETS/OE62_Total_Energy.xlsx',
                                sheet_name='Sheet1', columns=['SMILES', 'PBE0_Energy'])
    load_dataset['hiv'] = XLSX('../_DATASETS/HIV_classify.xlsx', sheet_name='Random',
                               columns=['SMILES', 'Output'])

    di = DataImporter(env)
    dh = di.load_data_InNewBatchPartition(load_dataset, max_chunksize=100000)

    MolRdkitConverter().convert(datatable=dh['lipo'], columns=['Smiles'], n_jobs=thread_count)
    MolRdkitConverter().convert(datatable=dh['tox'], columns=['SMILES'], n_jobs=thread_count)
    MolRdkitConverter().convert(datatable=dh['oe62'], columns=['SMILES'], n_jobs=thread_count)
    MolRdkitConverter().convert(datatable=dh['hiv'], columns=['SMILES'], n_jobs=thread_count)

    print('--------------------------------- MOL CONVERTER FINISHED -----------------------------------------------')

    evo_fp_modul = EvoFP()
    smart_fp = evo_fp_modul.load_fingerprint(fingerprint_file)
    evo_fp_modul.convert(smart_fp=smart_fp, datatable=dh['lipo'], columns=['Smiles'], n_jobs=thread_count,
                         feature_typ=feature_typ)

    print('--------------------------------- LIP FP FINISHED -----------------------------------------------')

    evo_fp_modul.convert(smart_fp=smart_fp, datatable=dh['tox'], columns=['SMILES'], n_jobs=thread_count,
                         feature_typ=feature_typ)

    print('--------------------------------- TOX FP FINISHED -----------------------------------------------')

    evo_fp_modul.convert(smart_fp=smart_fp, datatable=dh['oe62'], columns=['SMILES'], n_jobs=thread_count,
                         feature_typ=feature_typ)

    print('--------------------------------- OE62 FP FINISHED -----------------------------------------------')

    evo_fp_modul.convert(smart_fp=smart_fp, datatable=dh['hiv'], columns=['SMILES'], n_jobs=thread_count,
                         feature_typ=feature_typ)

    print('--------------------------------- HIV FP FINISHED -----------------------------------------------')

    splitter_lipo = RangeSplitter(0, 840)
    splitter_tox = RangeSplitter(12305, 17580)
    splitter_oe62 = RangeSplitter(0, 12298)
    splitter_hiv = RangeSplitter(0, 8225)
    split_creator = Splitcreator()

    splitset_lipo = split_creator.generate_split(dh['lipo'], splitter_lipo)
    splitset_tox = split_creator.generate_split(dh['tox'], splitter_tox)
    splitset_oe62 = split_creator.generate_split(dh['oe62'], splitter_oe62)
    splitset_hiv = split_creator.generate_split(dh['hiv'], splitter_hiv)

    dataset_lip = Dataset(dh['lipo'], name='lipo', feature_col=['Smiles'], target_col=['Output'],
                          split=splitset_lipo, env=env)
    dataset_tox = Dataset(dh['tox'], name='tox', feature_col=['SMILES'], target_col=['LD50'], split=splitset_tox,
                          env=env)
    dataset_oe62 = Dataset(dh['oe62'], name='oe62', feature_col=['SMILES'], target_col=['PBE0_Energy'],
                           env=env)
    dataset_hiv = Dataset(dh['hiv'], name='hiv', feature_col=['SMILES'], target_col=['Output'],
                          split=splitset_hiv, env=env)

    job_factory = Job_Factory(env)
    job_runner = LocalRunner(env)

    r2score = R2_Score()
    mae = MeanAbsoluteError()
    metricStack_r = MetricStack({'r2': r2score, 'mae': mae})

    f1_Score = F1_Score()
    accuracy_score = AccuracyScore({'normalize': True})
    metricStack_c = MetricStack({'f1_score': f1_Score, 'accuracy_score': accuracy_score})

    catboost_r_model_BIG = Config(CatBoost_r, {'verbose': 0, 'thread_count': thread_count, 'depth': 12, 'iterations': 500,
                                         'allow_writing_files': False})
    catboost_r_model_LIPO = Config(CatBoost_r, {'verbose': 0, 'thread_count': thread_count, 'depth': 10, 'iterations': 100,
                                          'allow_writing_files': False})
    catboost_c_model = Config(CatBoost_c, {'verbose': 0, 'thread_count': thread_count, 'allow_writing_files': False})
    job: ModelTrainEvalJob = job_factory.create_ModelTrainEvalJob('lipoJOB', dataset_lip, catboost_r_model_LIPO,
                                                                  metricStack_r,
                                                                  dataset_lip.get_Splitset().
                                                                  get_outer_split(0))
    job_runner.run_Job(job)

    print(f'Test_lipo: {job.result_metric_TEST}')
    print(f'Train_lipo: {job.result_metric_TRAIN}')

    job: ModelTrainEvalJob = job_factory.create_ModelTrainEvalJob('toxJOB', dataset_tox, catboost_r_model_BIG,
                                                                  metricStack_r,
                                                                  dataset_tox.get_Splitset().
                                                                  get_outer_split(0))
    job_runner.run_Job(job)

    print(f'Test_tox: {job.result_metric_TEST}')
    print(f'Train_tox: {job.result_metric_TRAIN}')

    job: ModelTrainEvalJob = job_factory.create_ModelTrainEvalJob('oe62JOB', dataset_oe62, catboost_r_model_BIG,
                                                                  metricStack_r,
                                                                  dataset_oe62.get_Splitset().
                                                                  get_outer_split(0))
    job_runner.run_Job(job)

    print(f'Test_oe62: {job.result_metric_TEST}')
    print(f'Train_oe62: {job.result_metric_TRAIN}')

    job: ModelTrainEvalJob = job_factory.create_ModelTrainEvalJob('hivJOB', dataset_hiv, catboost_c_model,
                                                                  metricStack_c,
                                                                  dataset_hiv.get_Splitset().
                                                                  get_outer_split(0))
    job_runner.run_Job(job)

    print(f'Test_hiv: {job.result_metric_TEST}')
    print(f'Train_hiv: {job.result_metric_TRAIN}')
