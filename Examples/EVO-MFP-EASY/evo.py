from EasyChemML.DataImport.DataImporter import DataImporter
from EasyChemML.DataImport.Module.XLSX import XLSX
from EasyChemML.Encoder.impl_RdkitConverter.MolRdkitConverter import MolRdkitConverter
from EasyChemML.Encoder.impl_EvoFP.Fitnessfunction.ModelWrapper_Fitness import ModelWrapper_Fitness
from EasyChemML.Environment import Environment

from EasyChemML.Encoder.DEPR_EvoFP import EvoFP
from EasyChemML.Encoder.impl_EvoFP.EVOFingerprint_Enum import FeatureTyp
from EasyChemML.JobSystem.JobFactory.JobFactory import Job_Factory
from EasyChemML.JobSystem.JobFactory.Module.Jobs import ModelTrainEvalJob
from EasyChemML.JobSystem.Runner.Module.LocalRunner import LocalRunner
from EasyChemML.JobSystem.Utilities.Config import Config
from EasyChemML.Metrik.MetricStack import MetricStack
from EasyChemML.Metrik.Module.ExplainedVarianceScore import ExplainedVarianceScore

from EasyChemML.Metrik.Module.MaxError import MaxError
from EasyChemML.Metrik.Module.MeanAbsoluteError import MeanAbsoluteError
from EasyChemML.Metrik.Module.MeanSquaredError import MeanSquaredError
from EasyChemML.Metrik.Module.R2_Score import R2_Score

from EasyChemML.Splitter.Module.RangeSplitter import RangeSplitter
from EasyChemML.Splitter.Splitcreator import Splitcreator
from EasyChemML.Utilities.Dataset import Dataset

from EasyChemML.Model.CatBoost_r import CatBoost_r

feature_typ = FeatureTyp.match_feature
thread_count = 12

env = Environment()

load_dataset = {}
load_dataset['lipo'] = XLSX('../_DATASETS/Lipophilicity.xlsx', sheet_name='Sheet1',
                                columns=['Smiles', 'Output'])

di = DataImporter(env)
dh = di.load_data_InNewBatchPartition(load_dataset)
job_factory = Job_Factory(env)
job_runner = LocalRunner(env)
split_creator = Splitcreator()

MolRdkitConverter().convert(datatable=dh['lipo'], columns=['Smiles'], n_jobs=thread_count)

print('--------------------------------- MOL CONVERTER FINISHED -----------------------------------------------')

splitter_lipo = RangeSplitter(1, 2200)
splitset_lipo = split_creator.generate_split(dh['lipo'], splitter_lipo)
dataset_lipo = Dataset(dh['lipo'], name='lipo', feature_col=['Smiles'], target_col=['Output'],
                          split=splitset_lipo, env=env)


catboost_r_lipo = Config(CatBoost_r, {'verbose': 0, 'thread_count': thread_count, 'allow_writing_files': False})
modelwrapper:ModelWrapper_Fitness = ModelWrapper_Fitness(train_dataset=dataset_lipo.to_SharedDataset(), feature_typ=FeatureTyp.count_feature, metric=R2_Score(), model=catboost_r_lipo)

evo_fp_modul = EvoFP()
fingerprint = evo_fp_modul.train(n_jobs=thread_count, populationSize=10, newgen_rate=0.1, gen_recombinationrate=0.15,
                                 newpop_perStep=0.2, usebestforkids=0.4, keepbest_perStep=0.2, fp_size=16, aging_rate=0.0,
                                 evo_steps=1,
                                 fitfunc=modelwrapper, feature_typ=FeatureTyp.count_feature, fitfunc_worker=1,
                                 environment=env)

fingerprint.save(fingerprint, 'saved.fp')

evo_fp_modul.convert(smart_fp=fingerprint, datatable=dh['lipo'], columns=['Smiles'], n_jobs=thread_count,
                     feature_typ=feature_typ)

r2score = R2_Score()
mean_squared_error = MeanSquaredError()
mean_absolute_error = MeanAbsoluteError()
explained_variance_score = ExplainedVarianceScore()
max_error = MaxError()


metricStack_r = MetricStack({'r2': r2score, 'mean_squared_error': mean_squared_error, 'mean_absolute_error':mean_absolute_error, 'explained_variance_score':explained_variance_score, 'max_error':max_error})

catboost_r_lipo = Config(CatBoost_r,
                        {'verbose': False, 'thread_count': thread_count, 'allow_writing_files': False})

job: ModelTrainEvalJob = job_factory.create_ModelTrainEvalJob(dataset_lipo.name, dataset_lipo, catboost_r_lipo,
                                                              metricStack_r,
                                                              dataset_lipo.get_Splitset().get_outer_split(0))
job_runner.run_Job(job)

print(job.result_metric_TEST['r2'])
print(job.result_metric_TRAIN['r2'])

