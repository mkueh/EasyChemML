use crate::impl_evo_fp::fitness_functions::fitness_metrics::{
    calculate_classification_scores, calculate_regression_scores, Metric, MetricDirection,
};
use crate::impl_evo_fp::fitness_functions::fitness_models::{
    generate_classification_predictions, generate_regression_predictions, Model,
};
use crate::impl_evo_fp::population::member::Member;
use ndarray::{Array, Ix1};
use pyo3::prelude::*;
use rdkit::ROMol;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct FitnessFunctionConfig {
    pub model: Model,
    pub fitness_metric: Metric,
    pub model_params: HashMap<String, PyObject>,
    pub k_folds: usize,
    pub split_ratio: f32,
}

pub fn calculate_fingerprint_fitness(
    population_member: &Member,
    features: &[Vec<ROMol>],
    regression_targets: &Option<Array<f64, Ix1>>,
    classification_targets: &Option<Array<usize, Ix1>>,
    bool_matching: bool,
    fitness_function_config: &FitnessFunctionConfig,
) -> Option<f32> {
    let metric_scores = if classification_targets.is_some() && regression_targets.is_some() {
        panic!("Regression and classification targets are both set!")
    } else if classification_targets.is_some() {
        let (test_folds, cross_validation_predictions) = generate_classification_predictions(
            population_member,
            features,
            classification_targets.clone().unwrap(),
            bool_matching,
            fitness_function_config,
        );

        calculate_classification_scores(
            &fitness_function_config.fitness_metric,
            &cross_validation_predictions,
            &test_folds,
        )
    } else {
        let (test_folds, cross_validation_predictions) = generate_regression_predictions(
            population_member,
            features,
            regression_targets.clone().unwrap(),
            bool_matching,
            fitness_function_config,
        );
        calculate_regression_scores(
            &fitness_function_config.fitness_metric,
            &cross_validation_predictions,
            &test_folds,
        )
    };
    let fingerprint_fitness = metric_scores.iter().sum::<f32>() / metric_scores.len() as f32;
    if fitness_function_config.fitness_metric.direction() == MetricDirection::Minimize {
        // Subtract for sorting when finding the best metric
        Some(0.0 - fingerprint_fitness)
    } else {
        Some(fingerprint_fitness)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::impl_evo_fp::fitness_functions::fitness_metrics::RegressionMetric::R2ScoreSingleTarget;
    use crate::impl_evo_fp::get_data::convert_data_to_mol;
    use crate::impl_evo_fp::population::Population;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_calculate_fingerprint_fitness_with_regressor() {
        let data = convert_data_to_mol(&arr2(&[
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
        let population = Population::generate_new_population(10, 5, 5, 5, false, &data);
        let regression_targets = arr1(&[
            3.54, -1.18, 3.69, 3.37, 3.1, 3.14, -0.72, 0.34, 3.05, 2.25, 1.51, 2.61, -0.08, 1.95,
            1.34, 3.2, 1.6, 3.77, 3.15, 0.32, 2.92, 1.92,
        ]);
        let bool_matching = true;
        Python::with_gil(|py| {
            let fitness_function_config = FitnessFunctionConfig {
                model: Model::CatBoostRegressor,
                fitness_metric: Metric::RegressionMetric(R2ScoreSingleTarget),
                model_params: HashMap::from([
                    ("verbose".to_string(), 0.into_py(py)),
                    ("allow_writing_files".to_string(), false.into_py(py)),
                ]),
                k_folds: 5,
                split_ratio: 0.8,
            };
            let metrics = calculate_fingerprint_fitness(
                &population.members[0],
                &data,
                &Some(regression_targets),
                &None,
                bool_matching,
                &fitness_function_config,
            );
            println!("predictions: {:?}", metrics)
        });
    }
}
