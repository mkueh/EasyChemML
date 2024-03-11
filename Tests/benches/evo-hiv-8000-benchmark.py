from EasyChemML.DataImport.DataImporter import DataImporter
from EasyChemML.DataImport.Module.XLSX import XLSX
from EasyChemML.Encoder.impl_RdkitConverter.MolRdkitConverter import MolRdkitConverter
from EasyChemML.Encoder.impl_EvoFP.Fitnessfunction.ModelWrapper_Fitness import ModelWrapper_Fitness
from EasyChemML.Environment import EasyProjectEnvironment

from EasyChemML.Encoder.EvoFP import EvoFP
from EasyChemML.Encoder.impl_EvoFP.EVOFingerprint_Enum import FeatureTyp
from EasyChemML.JobSystem.JobFactory.JobFactory import Job_Factory
from EasyChemML.JobSystem.Utilities.Config import Config
from EasyChemML.Metrik.Module.F1_Score import F1_Score

from EasyChemML.JobSystem.Runner.Module.LocalRunner import LocalRunner

from EasyChemML.Splitter.Module.RangeSplitter import RangeSplitter
from EasyChemML.Splitter.Splitcreator import Splitcreator
from EasyChemML.Utilities.Dataset import Dataset
from time import perf_counter

from EasyChemML.Model.CatBoost_c import CatBoost_c


# import cProfile


def start_timer(text, seperator='!--!'):
    print(seperator + ' ' + text + ' ' + seperator)
    return perf_counter()


def stop_timer(started_timer):
    stopping_timer = perf_counter()
    print('Evo fingerprint evolution takes ', stopping_timer - started_timer, ' time in seconds')


feature_type = FeatureTyp.count_feature
thread_count = 1

load_dataset = {'hiv': XLSX('../../Examples/_DATASETS/HIV_classify_8000.xlsx', sheet_name='Random',
                            columns=['SMILES', 'Output'])}

for i in range(10):
    print("****************************************************************************************************")
    print(f"Round {i}: Preprocessing started")
    env = EasyProjectEnvironment(f'TestFolder_{i}')
    di = DataImporter(env)
    dh = di.load_data_InNewBatchPartition(load_dataset)
    job_factory = Job_Factory(env)
    job_runner = LocalRunner(env)
    split_creator = Splitcreator()

    MolRdkitConverter().convert(datatable=dh['hiv'], columns=['SMILES'], n_jobs=12)

    print('--------------------------------- MOL CONVERTER FINISHED -----------------------------------------------')

    print("****************************************************************************************************")
    start = start_timer("Evo-FP generation")

    # splitter_hiv = RangeSplitter(605351, 632226)
    splitter_hiv = RangeSplitter(0, 850)

    splitset_hiv = split_creator.generate_split(dh['hiv'], splitter_hiv)
    dataset_hiv = Dataset(dh['hiv'], name='hiv', feature_col=['SMILES'], target_col=['Output'],
                          split=splitset_hiv, env=env)

    catboost_c_hiv = Config(CatBoost_c, {'verbose': False, 'thread_count': thread_count})
    modelwrapper: ModelWrapper_Fitness = ModelWrapper_Fitness(train_dataset=dataset_hiv.to_SharedDataset(),
                                                              feature_typ=feature_type, metric=F1_Score(),
                                                              model=catboost_c_hiv)

    evo_fp_modul = EvoFP()
    # profiler = cProfile.Profile()
    # profiler.enable()
    fingerprint = evo_fp_modul.train(n_jobs=thread_count,
                                     evo_steps=10,
                                     populationSize=10,
                                     fp_size=64,
                                     feature_typ=feature_type,
                                     gen_recombinationrate=0.06,
                                     newgen_rate=0.04,
                                     newpop_perStep=0.2,
                                     usebestforkids=0.4,
                                     keepbest_perStep=0.2,
                                     aging_rate=0.0,
                                     fitfunc=modelwrapper, fitfunc_worker=1,
                                     environment=env)
    # profiler.disable()
    # profiler.dump_stats('evo_fp_stats.prof')
    stop_timer(start)
    print("****************************************************************************************************")
    env.clean()
    print(f"Round {i}: Environment cleaned")
