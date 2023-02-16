from EasyChemML.DataImport.DataImporter import DataImporter
from EasyChemML.DataImport.Module.XLSX import XLSX
from EasyChemML.Encoder.MFF import MFF
from EasyChemML.Encoder.impl_RdkitConverter.MolRdkitConverter import MolRdkitConverter
from EasyChemML.Environment import Environment
from EasyChemML.HyperparameterSearch.Algorithm.Hyperopt import Hyperopt
from EasyChemML.HyperparameterSearch.Utilities.ConfigStack import ConfigStack

from EasyChemML.JobSystem.Utilities.Config import Config
from EasyChemML.JobSystem.JobFactory.JobFactory import Job_Factory
from EasyChemML.JobSystem.JobFactory.Module.Jobs import ModelTrainEvalJob
from EasyChemML.Metrik.MetricStack import MetricStack

from EasyChemML.Metrik.Module.R2_Score import R2_Score
from EasyChemML.Model.scikit_RandomForestRegressor import scikit_RandomForestRegressor
from EasyChemML.JobSystem.Runner.Module.LocalRunner import LocalRunner

from EasyChemML.Splitter.Module.RangeSplitter import RangeSplitter
from EasyChemML.Splitter.Module.ShuffleSplitter import ShuffleSplitter
from EasyChemML.Splitter.Splitcreator import Splitcreator
from EasyChemML.Utilities.Dataset import Dataset

from EasyChemML.HyperparameterSearch.Utilities.HyperParamterTyps import IntRange

from EasyChemML.Model.CatBoost_r import CatBoost_r

thread_count = 12

env = Environment()


def main():
    load_dataset = {}
    load_dataset['doyle_Test1'] = XLSX('Tests/_TestDataset/DreherDoyle.xlsx',
                                       sheet_name='Test1',
                                       columns=['Ligand', 'Additive', 'Base', 'Aryl halide', 'Output'], range=(0, 100))

    di = DataImporter(env)
    dh = di.load_data_InNewBatchPartition(load_dataset)
    job_factory = Job_Factory(env)
    job_runner = LocalRunner(env)

    MolRdkitConverter().convert(datatable=dh['doyle_Test1'], columns=['Ligand', 'Additive', 'Base', 'Aryl halide'],
                                n_jobs=thread_count)

    print('--------------------------------- MOL CONVERTER FINISHED -----------------------------------------------')

    mff_encoder = MFF()
    mff_encoder.convert(datatable=dh['doyle_Test1'], columns=['Ligand', 'Additive', 'Base', 'Aryl halide'],
                        n_jobs=thread_count, fp_length=128)

    splitter_doyle_Test1_outer = RangeSplitter(50, 75)
    splitter_doyle_Test1_inner = ShuffleSplitter(3, test_size=0.3)
    split_creator = Splitcreator()
    splitset_doyle_Test1 = split_creator.generate_split(dh['doyle_Test1'], splitter_doyle_Test1_outer,
                                                        splitter_doyle_Test1_inner)

    dataset_doyle_Test1 = Dataset(dh['doyle_Test1'], name='doyle_Test1',
                                  feature_col=['Ligand', 'Additive', 'Base', 'Aryl halide'], target_col=['Output'],
                                  split=splitset_doyle_Test1, env=env)

    r2score = R2_Score()
    metricStack_r = MetricStack({'r2': r2score})

    catboost_r_doyle = Config(CatBoost_r, {'verbose': False, 'thread_count': thread_count, 'allow_writing_files': False,
                                           'iterations': IntRange(1, 50, 1)})
    randomForest_r_doyle = Config(scikit_RandomForestRegressor, {'n_estimators': IntRange(1, 50, 1)})
    configStack = ConfigStack([catboost_r_doyle, randomForest_r_doyle])

    hyperParameterSearch = Hyperopt(env)
    best_config = hyperParameterSearch.performe_HyperparamterSearch(configStack,
                                                                    job_runner, dataset_doyle_Test1, r2score, 20)

    job: ModelTrainEvalJob = job_factory.create_ModelTrainEvalJob(dataset_doyle_Test1.name, dataset_doyle_Test1,
                                                                  best_config,
                                                                  metricStack_r,
                                                                  dataset_doyle_Test1.get_Splitset().get_outer_split(0))
    job_runner.run_Job(job)

    print(job.result_metric_TEST['r2'])
    print(job.result_metric_TRAIN['r2'])


if __name__ == '__main__':
    main()
