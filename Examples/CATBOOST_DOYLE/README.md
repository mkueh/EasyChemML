```python
#Imports

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

# ----------------------------------- Data Part ----------------------------------

# Set Environments
env = EasyProjectEnvironment('TestFolder')
di = DataImporter(env)
job_factory = Job_Factory(env)
job_runner = LocalRunner(env)

# Define which data should be loaded
zinc20_hdfLoader = {'dreher_dataset': XLSX('Examples/_DATASETS/Dreher_and_Doyle_input_data.xlsx', sheet_name='FullCV_01')}

# Load the data inside EasyChemML
bp = di.load_data_InNewBatchPartition(zinc20_hdfLoader, max_chunksize=100000)

# Convert to MolRDKit objects
molRdkit_converter = MolRdkitConverter()
molRdkit_converter.convert(bp['dreher_dataset'], columns=['Ligand', 'Additive', 'Base', 'Aryl halide'], n_jobs=10)

# Convert MolRDKit objects to MFF
mff_encoder = MFF()
mff_encoder.convert(datatable=bp['dreher_dataset'], columns=['Ligand', 'Additive', 'Base', 'Aryl halide'], fp_length=16,
                    n_jobs=64)

# Define Datasplitting
split_creator = Splitcreator()
splitter_boilingpoint = ShuffleSplitter(1, 42, test_size=0.1)
splitset_boilingpoint = split_creator.generate_split(bp['dreher_dataset'], splitter_boilingpoint)

# Define a Dataset that holds the data (feature, targets) and the splitting
dataset_boilingpoint = Dataset(bp['dreher_dataset'],
                               name='dreher_dataset',
                               feature_col=['Ligand', 'Additive', 'Base', 'Aryl halide'],
                               target_col=['Output'],
                               split=splitset_boilingpoint, env=env)

# ----------------------------------- Training Part ----------------------------------

# define metrics
r2score = R2_Score()
mae = MeanAbsoluteError()

# combine all metrics to be calculated
metricStack_r = MetricStack({'r2': r2score, 'mae': mae})

# define Model-Configuration
catboost_r = Config(
    CatBoost_r,
    {'verbose': 50,
     'thread_count': 64,
     'allow_writing_files': False,
     'iterations': 100,
     'depth': 4}
)

# define the job to be performed
job: ModelTrainEvalJob = job_factory.create_ModelTrainEvalJob(
    'dataset_boilingpoint_0',
    dataset_boilingpoint,
    catboost_r,
    metricStack_r,
    dataset_boilingpoint.get_Splitset().
    get_outer_split(0)
)

# run the job 
job_runner.run_Job(job)

# ----------------------------------- Result Part ----------------------------------

print(f'Test_lipo: {job.result_metric_TEST}')
print(f'Train_lipo: {job.result_metric_TRAIN}')

job.trained_Model.save_model('model_reaxys.catb')

# ----------------------------------- Reload trained Model ----------------------------------

# define model class
loaded_model = CatBoost_r()

# load saved model
loaded_model.load_model(path='model_reaxys.catb')

X_indices = list(range(len(bp['dreher_dataset'])))
job_predict = ModelPredictJob(job_name='Predict', trained_Model=loaded_model, X=bp['dreher_dataset'],
                              X_cols=['Ligand', 'Additive', 'Base', 'Aryl halide'])

# run only the prediction
job_runner.run_Job(job_predict)

# print results
for val in job_predict.predicted_vals:
    print(str(val))

# clean tmp files
env.clean()
```