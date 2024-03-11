import glob, os, pandas

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

from EasyChemML.Metrik.Module.F1_Score import F1_Score
from EasyChemML.Metrik.Module.R2_Score import R2_Score
from EasyChemML.JobSystem.Runner.Module.LocalRunner import LocalRunner

from EasyChemML.Splitter.Module.RangeSplitter import RangeSplitter
from EasyChemML.Splitter.Splitcreator import Splitcreator
from EasyChemML.Utilities.Dataset import Dataset

from EasyChemML.Model.CatBoost_r import CatBoost_r
from EasyChemML.Model.CatBoost_c import CatBoost_c

def __lagestArray(ars: []):
    max_size = -1

    for arr in ars:
        try:
            tmp_arr = list(arr)
        except:
            continue

        if isinstance(tmp_arr, list) and max_size < len(tmp_arr):
            max_size = len(tmp_arr)
    return max_size

def __exportToCSV(arrays: [], columns_a: [str], path, CSV_filename):
    DataContainer = {}
    maxsize = __lagestArray(ars=arrays)

    full_name = CSV_filename + '.csv'
    i = 0
    for c in columns_a:
        # TODO SO UGLY
        val = arrays[i][0]
        if isinstance(val, dict):
            arrays[i]: dict
            cols = list(val.keys())
            col_arrays = []
            for col in cols:
                if isinstance(val[col], list):
                    col_arrays.append(val[col])
                else:
                    col_arrays.append([val[col]])
            __exportToCSV(col_arrays, cols, path, CSV_filename + f'_metric_{c}')
            i = i + 1
        else:
            arrays[i] = list(arrays[i])
            if len(arrays[i]) < maxsize:
                arrays[i].extend([''] * (maxsize - len(arrays[i])))
            DataContainer[c] = arrays[i]
            i = i + 1

    dataFrame_results = pandas.DataFrame(DataContainer)
    dataFrame_results.to_csv(os.path.join(path, full_name), header=True, index=False, sep=',')


feature_typ = FeatureTyp.match_feature
thread_count = 50

pop_files = []
lipo_arr_test = []
lipo_arr_train = []
tox_arr_test = []
tox_arr_train = []
oe62_arr_test = []
oe62_arr_train = []
hiv_arr_test = []
hiv_arr_train = []

for pop_file in glob.glob('./*.pop'):
    head, tail = os.path.split(pop_file)
    pop_files.append(tail)

    env = Environment()

    load_dataset = {}
    load_dataset['lipo'] = XLSX('../DATASETS/Lipophilicity.xlsx', sheet_name='Sheet1',
                                columns=['Smiles', 'Output'])
    load_dataset['tox'] = XLSX('../DATASETS/Tox_Karmaus.xlsx', sheet_name='Random',
                               columns=['SMILES', 'LD50'])
    load_dataset['oe62'] = XLSX('../DATASETS/OE62_Total_Energy.xlsx',
                                sheet_name='Sheet1', columns=['SMILES', 'PBE0_Energy'])
    load_dataset['hiv'] = XLSX('../DATASETS/HIV_classify.xlsx', sheet_name='Random',
                               columns=['SMILES', 'Output'])

    di = DataImporter(env)
    dh = di.load_data_InNewBatchPartition(load_dataset)
    job_factory = Job_Factory(env)
    job_runner = LocalRunner(env)

    MolRdkitConverter().convert(datatable=dh['lipo'], columns=['Smiles'], n_jobs=thread_count)
    MolRdkitConverter().convert(datatable=dh['tox'], columns=['SMILES'], n_jobs=thread_count)
    MolRdkitConverter().convert(datatable=dh['oe62'], columns=['SMILES'], n_jobs=thread_count)
    MolRdkitConverter().convert(datatable=dh['hiv'], columns=['SMILES'], n_jobs=thread_count)

    print('--------------------------------- MOL CONVERTER FINISHED -----------------------------------------------')

    evo_fp_modul = EvoFP()
    smart_fp = evo_fp_modul.load_fingerprint_popFile(pop_file, 0)
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
                           split=splitset_oe62, env=env)
    dataset_hiv = Dataset(dh['hiv'], name='hiv', feature_col=['SMILES'], target_col=['Output'],
                          split=splitset_hiv, env=env)

    r2score = R2_Score()
    metricStack_r = MetricStack({'r2': r2score})

    f1_Score = F1_Score()
    metricStack_c = MetricStack({'f1_score': f1_Score})

    catboostR_config = Config(CatBoost_r, {'verbose': False, 'thread_count': thread_count})
    catboostC_config = Config(CatBoost_c, {'verbose': False, 'thread_count': thread_count})
    job: ModelTrainEvalJob = job_factory.create_ModelTrainEvalJob('lipoJOB', dataset_lip, catboostR_config, metricStack_r,
                                                                  dataset_lip.get_Splitset().
                                                                  get_outer_split(0))
    job_runner.run_Job(job)

    lipo_arr_test.append(job.result_metric_TEST['r2'])
    lipo_arr_train.append(job.result_metric_TRAIN['r2'])

    job: ModelTrainEvalJob = job_factory.create_ModelTrainEvalJob('toxJOB', dataset_tox, catboostR_config, metricStack_r,
                                                                  dataset_tox.get_Splitset().
                                                                  get_outer_split(0))
    job_runner.run_Job(job)

    tox_arr_test.append(job.result_metric_TEST['r2'])
    tox_arr_train.append(job.result_metric_TRAIN['r2'])

    job: ModelTrainEvalJob = job_factory.create_ModelTrainEvalJob('oe62JOB', dataset_oe62, catboostR_config, metricStack_r,
                                                                  dataset_oe62.get_Splitset().
                                                                  get_outer_split(0))
    job_runner.run_Job(job)

    oe62_arr_test.append(job.result_metric_TEST['r2'])
    oe62_arr_train.append(job.result_metric_TRAIN['r2'])

    job: ModelTrainEvalJob = job_factory.create_ModelTrainEvalJob('hivJOB', dataset_hiv, catboostC_config, metricStack_c,
                                                                  dataset_hiv.get_Splitset().
                                                                  get_outer_split(0))
    job_runner.run_Job(job)

    hiv_arr_test.append(job.result_metric_TEST['f1_score'])
    hiv_arr_train.append(job.result_metric_TRAIN['f1_score'])

columns_trainP = ['POP-File', 'Lipo-Train', 'TOX-Train', 'OE62-Train', 'HIV-Train', 'Lipo-Test', 'TOX-Test', 'OE62-Test', 'HIV-Test']
arrays = [pop_files, lipo_arr_train, tox_arr_train, oe62_arr_train, hiv_arr_train, lipo_arr_test, tox_arr_test, oe62_arr_test, hiv_arr_test]

__exportToCSV(arrays=arrays,
              columns_a=columns_trainP,
              path="/",
              CSV_filename=('PerfomanceEVO'))

