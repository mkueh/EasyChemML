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

#[allow(clippy::type_complexity)]
pub fn generate_classification_predictions(
    population_member: &Member,
    features: &[Vec<ROMol>],
    targets: Array1<usize>,
    bool_matching: bool,
    fitness_function_config: &FitnessFunctionConfig,
) -> (
    Vec<Dataset<usize, usize, Ix1>>,
    Vec<(Array1<usize>, Option<Array1<Vec<f32>>>)>,
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
        cross_validation_predictions.push(calculate_predictions(
            &fitness_function_config,
            &train_fold,
            &test_fold.records(),
        ));
        test_folds.push(test_fold);
    }
    (test_folds, cross_validation_predictions)
}

#[allow(clippy::type_complexity)]

pub fn generate_regression_predictions(
    population_member: &Member,
    features: &[Vec<ROMol>],
    targets: Array1<f64>,
    bool_matching: bool,
    fitness_function_config: &FitnessFunctionConfig,
) -> (
    Vec<Dataset<f64, f64, Ix1>>,
    Vec<(Array1<f64>, Option<Array1<Vec<f32>>>)>,
) {
    let mut rng = rand::thread_rng();

    let mut test_folds = Vec::new();
    let mut cross_validation_predictions = Vec::new();

    // Dimension of the feature data: (length of data, fp_size)
    let feature_data =
        generate_regression_feature_data(&population_member.fingerprint, features, bool_matching);

    let dataset = Dataset::new(feature_data, targets);

    // Manual cross validation
    for _ in 0..fitness_function_config.k_folds {
        let (train_fold, test_fold) = dataset
            .shuffle(&mut rng)
            .split_with_ratio(fitness_function_config.split_ratio);
        cross_validation_predictions.push(calculate_predictions(
            &fitness_function_config,
            &train_fold,
            &test_fold.records(),
        ));
        test_folds.push(test_fold);
    }
    (test_folds, cross_validation_predictions)
}

// Usage just for single output/target!
fn calculate_predictions<T: numpy::Element + for<'a> FromPyObject<'a>>(
    fitness_function_config: &FitnessFunctionConfig,
    train_data: &Dataset<T, T, Ix1>,
    test_records: &ArrayBase<OwnedRepr<T>, Ix2>,
) -> (Array1<T>, Option<Array1<Vec<f32>>>) {
    Python::with_gil(|py| -> (Array1<T>, Option<Array1<Vec<f32>>>) {
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

        let py_test_records = PyArray::from_array(py, test_records);
        let py_predictions = model
            .call_method1("predict", (py_test_records,))
            .expect("Could not predict");

        let probabilities: Option<Array1<Vec<f32>>> =
            if matches!(fitness_function_config.model, Model::CatBoostClassifier) {
                let probabilities: Vec<Vec<f32>> = model
                    .call_method1("predict_proba", (py_test_records,))
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
#[cfg(test)]
mod tests {
    use super::*;
    use crate::impl_evo_fp::fitness_functions::single_dataset_fitness::FitnessFunctionConfig;
    use ndarray::{arr1, arr2};
    use pyo3::IntoPy;
    use std::collections::HashMap;

    use crate::impl_evo_fp::fitness_functions::fitness_metrics::Metric::RegressionMetric;
    use crate::impl_evo_fp::fitness_functions::fitness_metrics::RegressionMetric::R2ScoreSingleTarget;
    use crate::impl_evo_fp::fitness_functions::fitness_models::Model::CatBoostRegressor;
    use crate::impl_evo_fp::get_data::convert_data_to_mol;
    use crate::impl_evo_fp::population::Population;

    #[test]
    fn generate_regression_predictions_returns_correct_values_for_valid_input() {
        let features = convert_data_to_mol(&arr2(&[
            ["Cn1c(CN2CCN(c3ccc(Cl)cc3)CC2)nc2ccccc21".to_string()],
            ["COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1".to_string()],
            ["O=C(NC1Cc2ccccc2N(C[C@@H](O)CO)C1=O)c1cc2cc(Cl)sc2[nH]1".to_string()],
            ["Cc1cccc(C[C@H](NC(=O)c2cc(C(C)(C)C)nn2C)C(=O)NCC#N)c1".to_string()],
            ["OC1(C#Cc2ccc(-c3ccccc3)cc2)CN2CCC1CC2".to_string()],
            ["COc1cc(OC)c(S(=O)(=O)NCc2ccccc2N2CCCCC2)cc1NC(=O)CCC(=O)O".to_string()],
            ["Oc1cc(OC)c(S(=O)(=O)NCc2ccccc2N2CCCCC2)cc1NC(=O)CCC(=O)O".to_string()],
            ["CNc1cccc(CCOc2ccc(C[C@H](NC(=O)c3c(Cl)cccc3Cl)C(=O)O)cc2C)n1".to_string()],
            ["COc1ccc(-c2coc3cc(OC)cc(OC)c3c2=O)cc1".to_string()],
            ["Oc1ncnc2scc(-c3ccsc3)c12".to_string()],
            ["CS(=O)(=O)c1ccc(Oc2ccc(C#C[C@]3(O)CN4CCC3CC4)cc2)cc1".to_string()],
            ["Cc1cc(Nc2nc(N[C@@H](C)c3ccc(F)cn3)c(C#N)nc2C)n[nH]1".to_string()],
            ["O=C1CCCCCN1".to_string()],
            ["CCCSc1ncccc1C(=O)N1CCCC1c1ccncc1".to_string()],
            ["Cc1ccc(S(=O)(=O)Nc2c(C(=O)NC3CCCCC3C)cnn2-c2ccccc2)cc1".to_string()],
            ["Nc1ccc(-c2nc3ccc(O)cc3s2)cc1".to_string()],
            ["COc1ccc(N2CCN(C(=O)[C@@H]3CCCC[C@H]3C(=O)NCC#N)CC2)cc1".to_string()],
            ["CCC(COC(=O)c1cc(OC)c(OC)c(OC)c1)(c1ccccc1)N(C)C".to_string()],
            ["COc1cc(-n2nnc3cc(-c4ccc(Cl)cc4)sc3c2=O)ccc1N1CC[C@@H](O)C1".to_string()],
            ["CO[C@H]1CN(CCn2c(=O)ccc3ccc(C#N)cc32)CC[C@H]1NCc1ccc2c(n1)NC(=O)CO2".to_string()],
            ["CC(C)(CCCCCOCCc1ccccc1)NCCc1ccc(O)c2nc(O)sc12".to_string()],
            ["O=C(Nc1nnc(C(=O)Nc2ccc(N3CCOCC3)cc2)o1)c1ccc(Cl)cc1".to_string()],
        ]));
        let population = Population::generate_new_population(1, 5, 5, 5, false, &features);
        let member = &population.members[0];
        let regression_targets = Some(arr1(&[
            3.54, -1.18, 3.69, 3.37, 3.1, 3.14, -0.72, 0.34, 3.05, 2.25, 1.51, 2.61, -0.08, 1.95,
            1.34, 3.2, 1.6, 3.77, 3.15, 0.32, 2.92, 1.92,
        ]));
        Python::with_gil(|py| {
            let fitness_function_config = FitnessFunctionConfig {
                model: CatBoostRegressor,
                fitness_metric: RegressionMetric(R2ScoreSingleTarget),
                model_params: HashMap::from([
                    ("verbose".to_string(), 0.into_py(py)),
                    ("allow_writing_files".to_string(), false.into_py(py)),
                ]),
                k_folds: 5,
                split_ratio: 0.8,
            };

            let bool_matching = true;
            let result = generate_regression_predictions(
                member,
                &features,
                regression_targets.unwrap(),
                bool_matching,
                &fitness_function_config,
            );

            assert!(result.0.len() > 0);
        });
    }
}
