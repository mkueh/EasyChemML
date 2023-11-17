# EasyChem-ML

EasyChem-ML is a modular structure-based machine learning tool for property, reactivity and structure prediction in (organic) chemistry, which should be easily adoptable to different problem sets.

# Current Status

EasyChem-ML is developed and maintained as a PhD project by Marius Kühnemund. The whole project is neither finished nor complete. Frequent modifications can not be excluded ;) ... that's how it is in PhD.



**In addition, not all modules are public available, as these were/are used for projects that have not been published yet**

\
\
***The framework has already been used for different research projects:*** 

* *Frederik Sandfort, Felix Strieth-Kalthoff, Marius Kühnemund, Christian Beecks, Frank Glorius, A Structure-Based Platform for Predicting Chemical Reactivity,
Chem, Volume 6, Issue 6, 2020, Pages 1379-1390, ISSN 2451-9294,* https://doi.org/10.1016/j.chempr.2020.02.017.

* *F. Strieth-Kalthoff, F. Sandfort, M. Kühnemund, F. R. Schäfer, H. Kuchen, F.Glorius, Angew. Chem. Int. Ed. 2022, 61, e202204647; Angew. Chem. 2022, 134, e202204647.* https://onlinelibrary.wiley.com/doi/full/10.1002/anie.202204647


# Install

For the installation, you need a working C++(For Cython) and a Rust compiler. 

## Linux

    # Install Rust-Compiler
    curl https://sh.rustup.rs -sSf | bash -s -- -y

    # Install Dependencies
    pip install setuptools-rust

    # Install EasyChemML
    git clone https://github.com/mkueh/EasyChemML.git
    cd EasyChemML
    pip install ./

## Windows

    # install Rust-Compiler
    # https://www.rust-lang.org/learn/get-started

    # Install Dependencies
    pip install setuptools-rust

    # Install EasyChemML
    git clone https://github.com/mkueh/EasyChemML.git
    cd EasyChemML
    pip install ./

# Short Example

```python
env = EasyProjectEnvironment('TestFolder')
job_factory = Job_Factory(env)
job_runner = LocalRunner(env)

# ----------------------------------- Dataloading --------------------------------------

dataLoader = {'dreher_dataset': XLSX('Examples/DATASETS/Dreher_and_Doyle_input_data.xlsx', sheet_name='FullCV_01')}
di = DataImporter(env)
bp = di.load_data_InNewBatchPartition(dataLoader, max_chunksize=100000)

# ----------------------------------- Preprocessing --------------------------------------

molRdkit_converter = MolRdkitConverter()
molRdkit_converter.convert(bp['dreher_dataset'], columns=['Ligand', 'Additive', 'Base', 'Aryl halide'], n_jobs=10)

mff_encoder = MFF()
mff_encoder.convert(datatable=bp['dreher_dataset'], columns=['Ligand', 'Additive', 'Base', 'Aryl halide'], fp_length=16, n_jobs=64)

# ----------------------------------- Splitting --------------------------------------

split_creator = Splitcreator()
splitter_boilingpoint = ShuffleSplitter(1, 42, test_size=0.1)
splitset_boilingpoint = split_creator.generate_split(bp['dreher_dataset'], splitter_boilingpoint)

dataset_boilingpoint = Dataset(bp['dreher_dataset'],
                               name='dreher_dataset',
                               feature_col=['Ligand', 'Additive', 'Base', 'Aryl halide'],
                               target_col=['Output'],
                               split=splitset_boilingpoint, env=env)

# ----------------------------------- Metric definition-----------------------------------

r2score = R2_Score()
mae = MeanAbsoluteError()
metricStack_r = MetricStack({'r2': r2score, 'mae': mae})

catboost_r = Config(
    CatBoost_r,
    {'verbose': 50,
     'thread_count': 64,
     'allow_writing_files': False,
     'iterations': 100,
     'depth': 4}
)

# ----------------------------------- Job definition-----------------------------------

job: ModelTrainEvalJob = job_factory.create_ModelTrainEvalJob(
    'dataset_boilingpoint_0',
    dataset_boilingpoint,
    catboost_r,
    metricStack_r,
    dataset_boilingpoint.get_Splitset().
    get_outer_split(0)
)


# ----------------------------------- Training ----------------------------------------
job_runner.run_Job(job)

print(f'Test_lipo: {job.result_metric_TEST}')
print(f'Train_lipo: {job.result_metric_TRAIN}')

job.trained_Model.save_model('model_reaxys.catb')

loaded_model = CatBoost_r()
loaded_model.load_model(path='model_reaxys.catb')

X_indices = list(range(len(bp['dreher_dataset'])))
job_predict = ModelPredictJob(job_name='Predict', trained_Model=loaded_model, X=bp['dreher_dataset'],
                              X_cols=['Ligand', 'Additive', 'Base', 'Aryl halide'])
job_runner.run_Job(job_predict)

for val in job_predict.predicted_vals:
    print(str(val))


# ----------------------------------- Cleanup ----------------------------------------

env.clean()
```


# Examples

Look inside the Example folder

# Current implemented Modules

## Dataloader
* CSV
* HDF5
* XLSX

## Splitter
* AllTestSplitter (all Data in the Testset)
* AllTrainSplitter (all Data in the Trainset)
* RangeSplitter (define a Test/Train range)
* ShuffleSplitter (Splitts Random)

## Encoder
* BertTokenizer
* Fingerprints (all Rdkit)
* MFF
* OnehotEncoder
* EasyDescriptor (support only Molmass yet)


## Model
* Catboost (Classification and Regression)
* Pytorch Models
  - Bert-Transformer
* RandomForest (Classification and Regression)

## Jobrunner
* Localrunner (Runs the Jobs on the local machine)
