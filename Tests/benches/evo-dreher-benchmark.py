from EasyChemML.DataImport.DataImporter import DataImporter
from EasyChemML.DataImport.Module.XLSX import XLSX
from EasyChemML.Encoder.impl_RdkitConverter.MolRdkitConverter import MolRdkitConverter
from EasyChemML.Encoder.impl_EvoFP.Fitnessfunction.ModelWrapper_Fitness import ModelWrapper_Fitness
from EasyChemML.Environment import EasyProjectEnvironment

from EasyChemML.Encoder.EvoFP import EvoFP
from EasyChemML.Encoder.impl_EvoFP.EVOFingerprint_Enum import FeatureTyp
from EasyChemML.JobSystem.JobFactory.JobFactory import Job_Factory
from EasyChemML.JobSystem.Utilities.Config import Config

from EasyChemML.Metrik.Module.R2_Score import R2_Score
from EasyChemML.JobSystem.Runner.Module.LocalRunner import LocalRunner

from EasyChemML.Splitter.Module.RangeSplitter import RangeSplitter
from EasyChemML.Splitter.Splitcreator import Splitcreator
from EasyChemML.Utilities.Dataset import Dataset
from time import perf_counter

from EasyChemML.Model.CatBoost_r import CatBoost_r
# import cProfile


def start_timer(text, seperator='!--!'):
    print(seperator + ' ' + text + ' ' + seperator)
    return perf_counter()


def stop_timer(started_timer):
    stopping_timer = perf_counter()
    print('Evo fingerprint evolution takes ', stopping_timer - started_timer, ' time in seconds')


feature_type = FeatureTyp.count_feature
thread_count = 1

load_dataset = {
    'doyle_Test1': XLSX('../../Examples/_DATASETS/Dreher_and_Doyle_input_data.xlsx',
                        sheet_name='Test1',
                        columns=['Ligand', 'Additive', 'Base', 'Aryl halide', 'Output'])}

for i in range(10):
    print("****************************************************************************************************")
    print(f"Round {i}: Preprocessing started")
    env = EasyProjectEnvironment(f'TestFolder_{i}')
    di = DataImporter(env)
    dh = di.load_data_InNewBatchPartition(load_dataset)
    job_factory = Job_Factory(env)
    job_runner = LocalRunner(env)
    split_creator = Splitcreator()

    MolRdkitConverter().convert(datatable=dh['doyle_Test1'], columns=['Ligand', 'Additive', 'Base', 'Aryl halide'],
                                n_jobs=12)

    print('--------------------------------- MOL CONVERTER FINISHED -----------------------------------------------')

    print("****************************************************************************************************")
    start = start_timer("Evo-FP generation")

    # splitter_doyle_Test1 = RangeSplitter(605351, 632226)
    splitter_doyle_Test1 = RangeSplitter(3057, 3955)

    splitset_doyle_Test1 = split_creator.generate_split(dh['doyle_Test1'], splitter_doyle_Test1)
    dataset_doyle_Test1 = Dataset(dh['doyle_Test1'], name='doyle_Test1',
                                  feature_col=['Ligand', 'Additive', 'Base', 'Aryl halide'], target_col=['Output'],
                                  split=splitset_doyle_Test1, env=env)

    catboost_r_doyle_Test1 = Config(CatBoost_r,
                                    {'verbose': 0, 'thread_count': thread_count, 'allow_writing_files': False})
    modelwrapper: ModelWrapper_Fitness = ModelWrapper_Fitness(train_dataset=dataset_doyle_Test1.to_SharedDataset(),
                                                              feature_typ=feature_type, metric=R2_Score(),
                                                              model=catboost_r_doyle_Test1)

    evo_fp_modul = EvoFP()
    # profiler = cProfile.Profile()
    # profiler.enable()
    fingerprint = evo_fp_modul.train(n_jobs=thread_count,
                                     evo_steps=10,
                                     populationSize=10,
                                     fp_size=64,
                                     feature_typ=feature_type,
                                     gen_recombinationrate=0.15,
                                     newgen_rate=0.1,
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
