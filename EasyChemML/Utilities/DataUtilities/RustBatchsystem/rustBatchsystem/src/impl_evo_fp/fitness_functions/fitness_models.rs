use crate::impl_evo_fp::fitness_functions::single_dataset_fitness::FitnessFunctionConfig;
use crate::impl_evo_fp::get_data::{
    generate_classification_feature_data, generate_regression_feature_data,
};
use crate::impl_evo_fp::population::member::Member;
use linfa::Dataset;
use ndarray::{Array, Array1, ArrayBase, Ix1, Ix2, OwnedRepr};
use numpy::PyArray;
use pyo3::prelude::PyModule;
use pyo3::types::{IntoPyDict, PyString};
use pyo3::{FromPyObject, Python};
use rdkit::ROMol;

#[derive(Clone, Debug)]
pub enum Model {
    CatBoostRegressor,
    CatBoostClassifier,
}

impl Model {
    pub fn get_model(&self) -> String {
        match self {
            Model::CatBoostRegressor => "CatBoostRegressor".to_string(),
            Model::CatBoostClassifier => "CatBoostClassifier".to_string(),
        }
    }
}

pub fn generate_classification_predictions(
    population_member: &Member,
    features: &Vec<ROMol>,
    targets: Array1<usize>,
    bool_matching: bool,
    fitness_function_config: &FitnessFunctionConfig,
) -> (
    Vec<Dataset<usize, usize, Ix1>>,
    Vec<(Array1<usize>, Option<Array1<f32>>)>,
) {
    let mut rng = rand::thread_rng();

    let mut test_folds = Vec::new();
    let mut cross_validation_predictions = Vec::new();

    let feature_data = generate_classification_feature_data(
        &population_member.fingerprint,
        features,
        bool_matching,
    );

    let dataset = Dataset::new(feature_data, targets);
    // Manual cross validation
    for _ in 0..fitness_function_config.k_folds {
        let (train_fold, test_fold) = dataset
            .shuffle(&mut rng)
            .split_with_ratio(fitness_function_config.split_ratio);
        test_folds.push(test_fold.clone());
        cross_validation_predictions.push(calculate_predictions(
            fitness_function_config.clone(),
            &train_fold,
            test_fold.records(),
        ));
    }
    (test_folds, cross_validation_predictions)
}

pub fn generate_regression_predictions(
    population_member: &Member,
    features: &Vec<ROMol>,
    targets: Array1<f64>,
    bool_matching: bool,
    fitness_function_config: &FitnessFunctionConfig,
) -> (
    Vec<Dataset<f64, f64, Ix1>>,
    Vec<(Array1<f64>, Option<Array1<f32>>)>,
) {
    let mut rng = rand::thread_rng();

    let mut test_folds = Vec::new();
    let mut cross_validation_predictions = Vec::new();

    let feature_data =
        generate_regression_feature_data(&population_member.fingerprint, features, bool_matching);

    let dataset = Dataset::new(feature_data, targets);
    // Manual cross validation
    for _ in 0..fitness_function_config.k_folds {
        let (train_fold, test_fold) = dataset
            .shuffle(&mut rng)
            .split_with_ratio(fitness_function_config.split_ratio);
        test_folds.push(test_fold.clone());
        cross_validation_predictions.push(calculate_predictions(
            fitness_function_config.clone(),
            &train_fold,
            test_fold.records(),
        ));
    }
    (test_folds, cross_validation_predictions)
}

// Usage just for single output!
fn calculate_predictions<T: numpy::Element + for<'a> FromPyObject<'a>>(
    fitness_function_config: FitnessFunctionConfig,
    train_data: &Dataset<T, T, Ix1>,
    test_records: &ArrayBase<OwnedRepr<T>, Ix2>,
) -> (Array1<T>, Option<Array1<f32>>) {
    Python::with_gil(|py| -> (Array1<T>, Option<Array1<f32>>) {
        let catboost = PyModule::import(py, "catboost").expect("Could not import catboost");
        let regressor = catboost
            .getattr(PyString::new(
                py,
                &fitness_function_config.model.get_model(),
            ))
            .expect("Could not get regressor");
        let model = regressor
            .call(
                (),
                Some(
                    fitness_function_config
                        .model_params
                        .clone()
                        .into_py_dict(py),
                ),
            )
            .expect("Could not initiate model");

        model
            .call_method1(
                "fit",
                (
                    PyArray::from_array(py, &train_data.records),
                    PyArray::from_array(py, &train_data.targets),
                ),
            )
            .expect("Could not fit model");

        let py_predictions = model
            .call_method1("predict", (PyArray::from_array(py, &test_records),))
            .expect("Could not predict");

        let probabilities: Option<Array1<f32>> =
            if matches!(fitness_function_config.model, Model::CatBoostClassifier) {
                let probabilities: Vec<f32> = model
                    .call_method1("predict_proba", (PyArray::from_array(py, &test_records),))
                    .expect("Could not predict probabilities")
                    .extract()
                    .unwrap();
                Some(Array::from(probabilities))
            } else {
                None
            };
        let predictions_vec: Vec<T> = py_predictions.extract().unwrap();
        let predictions = Array::from(predictions_vec);
        (predictions, probabilities)
    })
}
