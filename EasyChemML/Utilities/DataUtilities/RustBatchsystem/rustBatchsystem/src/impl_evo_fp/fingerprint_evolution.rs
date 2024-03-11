mod fingerprint_inheritance;

use crate::impl_evo_fp::fitness_functions::single_dataset_fitness::FitnessFunctionConfig;
use crate::impl_evo_fp::population::member::Member;
use crate::impl_evo_fp::population::Population;
use ndarray::{Array, Ix1};
use num::ToPrimitive;
use rdkit::ROMol;

#[derive(Debug, PartialEq, thiserror::Error)]
pub enum FingerprintEvolutionError {
    #[error("could not convert smiles to romol (nullptr)")]
    UnknownConversionError,
    #[error("could not mutate fingerprint")]
    MutationError,
}

#[derive(Debug, Clone)]
pub struct EvolutionConfig {
    pub evolution_steps: u8,
    pub population_size: usize,
    pub fp_size: usize,
    pub max_primitive_count: u8,
    pub max_bound_count: u8,
    pub bool_matching: bool,
    // crossover or mutation?
    pub gene_recombination_rate: f32,
    pub new_gene_rate: f32,
    // how much should be newly generated
    pub proportion_new_population: f32,
    // How many should be used for mutation (generation of children)?
    pub proportion_best_parents: f32,
    pub proportion_best_fingerprints: f32,
    pub work_folder: String,
    pub gene_recombination_attempts: i8,
}

#[inline]
pub fn create_the_one_fingerprint(
    evolution_config: &EvolutionConfig,
    fitness_function_config: &FitnessFunctionConfig,
    mol_data: Vec<Vec<ROMol>>,
    regression_targets: &Option<Array<f64, Ix1>>,
    classification_targets: &Option<Array<usize, Ix1>>,
) -> Result<Member, std::io::Error> {
    println!("One fingerprint to rule them all!!");
    let (existing_population, mut found_steps) =
        Population::highest_saved_population(&evolution_config.work_folder)?;
    let mut population = if let Some(exist_population) = existing_population {
        exist_population
    } else {
        println!("Generating first population");
        Population::generate_new_population(
            evolution_config.population_size,
            evolution_config.fp_size,
            evolution_config.max_primitive_count,
            evolution_config.max_bound_count,
            evolution_config.bool_matching,
            &mol_data,
        )
    };
    if found_steps as u8 >= evolution_config.evolution_steps {
        panic!("Found higher population than indicated evolution steps")
    }
    if found_steps == 0 {
        println!("Generating first population metrics");
        population.calculate_population_metrics(
            &mol_data,
            regression_targets.clone(),
            classification_targets.clone(),
            evolution_config.bool_matching,
            fitness_function_config.clone(),
        );
        population.save_population(&evolution_config.work_folder, 0)?;
        found_steps += 1;
    }

    let mut evolved_population: Population = Population::empty();
    for i in found_steps..=evolution_config.evolution_steps as i8 {
        println!("Evolution step {}", i);
        evolved_population = evolution_step(&evolution_config, &mol_data, &mut population);
        evolved_population.calculate_population_metrics(
            &mol_data,
            regression_targets.clone(),
            classification_targets.clone(),
            evolution_config.bool_matching,
            fitness_function_config.clone(),
        );
        evolved_population
            .members
            .sort_unstable_by(|member_a, member_b| {
                member_b
                    .metric
                    .unwrap()
                    .partial_cmp(&member_a.metric.unwrap())
                    .unwrap()
            });
        evolved_population.save_population(&evolution_config.work_folder, i)?;
    }
    println!("Finished evolution");
    println!("The one fingerprint: {} ", evolved_population.members[0]);
    Ok(evolved_population.members[0].clone())
}

fn evolution_step(
    evolution_config: &EvolutionConfig,
    feature_data: &[Vec<ROMol>],
    start_population: &mut Population,
) -> Population {
    let number_best =
        evolution_config.proportion_best_fingerprints * evolution_config.population_size as f32;
    let number_best = number_best.to_usize().unwrap();
    let number_new_fps =
        evolution_config.proportion_new_population * evolution_config.population_size as f32;
    let number_new_fps = number_new_fps.to_usize().unwrap();
    let number_kids = evolution_config.population_size - number_best - number_new_fps;
    let number_parents =
        evolution_config.proportion_best_parents * evolution_config.population_size as f32;
    let number_parents = number_parents.to_usize().unwrap();

    let mut new_population = Population::generate_new_population(
        number_new_fps,
        evolution_config.fp_size,
        evolution_config.max_primitive_count,
        evolution_config.max_bound_count,
        evolution_config.bool_matching,
        feature_data,
    );
    let mut best_members = start_population.n_best_members(number_best);

    let parents = start_population.n_best_members(number_parents);

    let mut kids = fingerprint_inheritance::evolve_population(
        evolution_config,
        number_kids,
        &parents,
        feature_data,
    );
    new_population.members.append(&mut kids);
    new_population.members.append(&mut best_members);
    new_population
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::impl_evo_fp::fitness_functions::fitness_metrics::ClassificationMetric::{
        F1Score, Precision,
    };
    use crate::impl_evo_fp::fitness_functions::fitness_metrics::Metric::{
        ClassificationMetric, RegressionMetric,
    };
    use crate::impl_evo_fp::fitness_functions::fitness_metrics::RegressionMetric::R2ScoreSingleTarget;
    use crate::impl_evo_fp::fitness_functions::fitness_models::Model::{
        CatBoostClassifier, CatBoostRegressor,
    };
    use crate::impl_evo_fp::get_data::convert_data_to_mol;
    use ndarray::{arr1, arr2, Array1, Array2};
    use pyo3::{IntoPy, Python};
    use std::collections::HashMap;
    use std::fs::File;
    use std::time::Instant;
    use tempfile::tempdir;

    #[test]
    fn test_create_the_one_fingerprint() -> std::io::Result<()> {
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
        let temp_dir = tempdir().unwrap();
        let path = temp_dir.path().to_str().unwrap();
        // Create instances of the required types
        let evolution_config = EvolutionConfig {
            evolution_steps: 3,
            population_size: 10,
            fp_size: 3,
            max_primitive_count: 5,
            max_bound_count: 4,
            bool_matching: true,
            gene_recombination_rate: 0.6,
            new_gene_rate: 0.6,
            proportion_new_population: 0.3,
            proportion_best_parents: 0.2,
            proportion_best_fingerprints: 0.5,
            work_folder: path.to_string(),
            gene_recombination_attempts: 5,
        };
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

            let regression_targets = Some(arr1(&[
                3.54, -1.18, 3.69, 3.37, 3.1, 3.14, -0.72, 0.34, 3.05, 2.25, 1.51, 2.61, -0.08,
                1.95, 1.34, 3.2, 1.6, 3.77, 3.15, 0.32, 2.92, 1.92,
            ]));

            // Call the function
            let result = create_the_one_fingerprint(
                &evolution_config,
                &fitness_function_config,
                features,
                &regression_targets,
                &None,
            );
            assert!(result.is_ok());
            // Check that the function returned the expected result
            match result {
                Ok(_) => println!("YAAAAS!!"),
                Err(err) => panic!("The function returned an error: {}", err),
            }

            // Delete the temporary directory
            temp_dir.close().unwrap();
        });
        Ok(())
    }

    #[test]
    fn test_create_the_one_fingerprint_classification() -> std::io::Result<()> {
        let features = convert_data_to_mol(&arr2(&[
            ["Cc1[nH]c(=O)[nH]c1C(=O)c1ccc(-n2ccnc2)cc1".to_string()],
            ["C(=Nc1cnc2ccccc2c1)c1ccc2ccccc2n1".to_string()],
            ["O=C1c2ccc3cc(CCl)cc(O)c3c2C(=O)c2cccc(O)c21".to_string()],
            ["CCCCCCCCCCCCCCCCCCSCC(COP(=O)(O)OP(=O)(O)OCC1OC(n2ccc(=O)[nH]c2=O)CC1N=[N+]=[N-])OC(=O)CCCCCCCCCCCCCCC.[NaH]".to_string()],
            ["CCOC(=O)c1c(NS(=O)(=O)c2ccc(C)cc2)sc2c1CCCC2".to_string()],
            ["COC12CC(=O)C(=C(O)C=Cc3ccccc3)C(=O)C1(O)Cc1ccccc1O2".to_string()],
            ["O=C1CSC(=Nc2ccccc2)N1c1ncccn1".to_string()],
            ["C=C(CCCC(C)=CC=CC1(C)CCCc2ccoc21)CC(=O)O".to_string()],
            ["CC1=[O+][V-]2(=O)([O+]=C(C)C1)[O+]=C(C)CC(C)=[O+]2".to_string()],
            ["O=c1c(=Cc2ccco2)sc2nc3ccccc3n12".to_string()],
            ["CC1=C(c2ccccc2)CON(c2ccccc2)C1".to_string()],
            ["N#Cc1c(N)oc(-c2ccccc2)c1-c1ccccc1".to_string()],
            ["C[n+]1ccn2c3nc4ccccc4nc3c([O-])c(C#N)c21".to_string()],
            ["CC(C)OP(=O)(OC(C)C)C(=NNc1ccc([N+](=O)[O-])cc1)NN".to_string()],
            ["CCN(CC)c1ccc2c(c1)Oc1ccc(N(C(C)=O)c3ccccc3)cc1C21OC(=O)c2ccccc21".to_string()],
            ["NC1=c2nc(-c3ccccc3)c(-c3ccccc3)nc2=NS(=O)(O)=N1".to_string()],
            ["O=C1OC(=C(Br)c2ccc([N+](=O)[O-])cc2)c2ccccc21".to_string()],
            ["COc1ccc(NC(=O)C(=O)C2CNC(=O)NC2=O)c(OC)c1".to_string()],
            ["Cc1cn(C2CC(F)C(CO)O2)c(=O)[nH]c1=O".to_string()],
            ["CCOC(=O)c1sc2c(c1O)c(=O)n(-c1ccccc1)c(=S)n2-c1ccccc1".to_string()],
            ["COc1cc(-c2cccs2)sc1-c1cccs1".to_string()],
            ["N=C(N)NS(=O)(=O)c1ccc(N=c2c3ccccc3n(Cc3ccccc3)c3ccccc23)cc1".to_string()],
            ["O=P(OC1=CCCCC1)(c1ccccc1)c1ccccc1".to_string()],
            ["Cc1cc(Cl)ccc1NC(=O)C1C(=O)C(=O)N(c2ccc(Cl)cc2C)C1=O".to_string()],
            ["Cc1cc(C)c(C(=O)C(=Cc2ccccc2)c2ccccc2)c(C)c1".to_string()],
            ["CCN(CC)C1Oc2cc3c(cc2C(c2cccc(OC)c2O)C1C)OCO3".to_string()],
            ["COC(=O)C1(Cc2ccc3c(c2)CCC3)Cc2ccc3c(c2C1O)CCC3".to_string()],
            ["N#Cc1c(Cl)c2ccccc2n2c1nc1ccccc12".to_string()],
            ["Cc1c2ccnc(C(N)=O)c2c(C)c2c1[nH]c1ccccc12".to_string()],
            ["O=C(O)c1cccc(OCCOc2cccc(C(=O)O)c2[N+](=O)[O-])c1[N+](=O)[O-]".to_string()],
            ["C=C1C2CCC(C)(C)C1CCC(=O)C2C(=O)OC".to_string()],
            ["C[Sn]1(C)OC(=O)CCN=Cc2c(ccc3ccccc23)O[Sn](C)(C)OC(=O)CCN=Cc2c(ccc3ccccc23)O1".to_string()],
            ["CC(=NNC(=O)c1ccccc1O)c1c(O)cc(C)oc1=O".to_string()],
        ]));
        let temp_dir = tempdir().unwrap();
        let path = temp_dir.path().to_str().unwrap();
        // Create instances of the required types
        let evolution_config = EvolutionConfig {
            evolution_steps: 3,
            population_size: 10,
            fp_size: 5,
            max_primitive_count: 4,
            max_bound_count: 4,
            bool_matching: false,
            gene_recombination_rate: 0.06,
            new_gene_rate: 0.04,
            proportion_new_population: 0.2,
            proportion_best_parents: 0.4,
            proportion_best_fingerprints: 0.2,
            work_folder: path.to_string(),
            gene_recombination_attempts: 10,
        };
        Python::with_gil(|py| {
            let fitness_function_config = FitnessFunctionConfig {
                model: CatBoostClassifier,
                fitness_metric: ClassificationMetric(F1Score),
                model_params: HashMap::from([
                    ("verbose".to_string(), 0.into_py(py)),
                    ("allow_writing_files".to_string(), false.into_py(py)),
                ]),
                k_folds: 5,
                split_ratio: 0.8,
            };

            let classification_targets = Some(arr1(&[
                0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0,
                0, 0, 1, 0, 0,
            ]));

            // Call the function
            let result = create_the_one_fingerprint(
                &evolution_config,
                &fitness_function_config,
                features,
                &None,
                &classification_targets,
            );
            assert!(result.is_ok());
            // Check that the function returned the expected result
            match result {
                Ok(_) => println!("YAAAAS!!"),
                Err(err) => panic!("The function returned an error: {}", err),
            }

            // Delete the temporary directory
            temp_dir.close().unwrap();
        });
        Ok(())
    }

    #[test]
    fn test_create_the_one_fingerprint_multi_column() -> std::io::Result<()> {
        let features = convert_data_to_mol(&arr2(&[
            [
                "Cn1c(CN2CCN(c3ccc(Cl)cc3)CC2)nc2ccccc21".to_string(),
                "COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1".to_string(),
            ],
            [
                "O=C(NC1Cc2ccccc2N(C[C@@H](O)CO)C1=O)c1cc2cc(Cl)sc2[nH]1".to_string(),
                "Cc1cccc(C[C@H](NC(=O)c2cc(C(C)(C)C)nn2C)C(=O)NCC#N)c1".to_string(),
            ],
            [
                "OC1(C#Cc2ccc(-c3ccccc3)cc2)CN2CCC1CC2".to_string(),
                "COc1cc(OC)c(S(=O)(=O)NCc2ccccc2N2CCCCC2)cc1NC(=O)CCC(=O)O".to_string(),
            ],
            [
                "Oc1cc(OC)c(S(=O)(=O)NCc2ccccc2N2CCCCC2)cc1NC(=O)CCC(=O)O".to_string(),
                "CNc1cccc(CCOc2ccc(C[C@H](NC(=O)c3c(Cl)cccc3Cl)C(=O)O)cc2C)n1".to_string(),
            ],
            [
                "COc1ccc(-c2coc3cc(OC)cc(OC)c3c2=O)cc1".to_string(),
                "Oc1ncnc2scc(-c3ccsc3)c12".to_string(),
            ],
            [
                "CS(=O)(=O)c1ccc(Oc2ccc(C#C[C@]3(O)CN4CCC3CC4)cc2)cc1".to_string(),
                "Cc1cc(Nc2nc(N[C@@H](C)c3ccc(F)cn3)c(C#N)nc2C)n[nH]1".to_string(),
            ],
            [
                "O=C1CCCCCN1".to_string(),
                "CCCSc1ncccc1C(=O)N1CCCC1c1ccncc1".to_string(),
            ],
            [
                "Cc1ccc(S(=O)(=O)Nc2c(C(=O)NC3CCCCC3C)cnn2-c2ccccc2)cc1".to_string(),
                "Nc1ccc(-c2nc3ccc(O)cc3s2)cc1".to_string(),
            ],
            [
                "COc1ccc(N2CCN(C(=O)[C@@H]3CCCC[C@H]3C(=O)NCC#N)CC2)cc1".to_string(),
                "CCC(COC(=O)c1cc(OC)c(OC)c(OC)c1)(c1ccccc1)N(C)C".to_string(),
            ],
            [
                "COc1cc(-n2nnc3cc(-c4ccc(Cl)cc4)sc3c2=O)ccc1N1CC[C@@H](O)C1".to_string(),
                "CO[C@H]1CN(CCn2c(=O)ccc3ccc(C#N)cc32)CC[C@H]1NCc1ccc2c(n1)NC(=O)CO2".to_string(),
            ],
            [
                "CC(C)(CCCCCOCCc1ccccc1)NCCc1ccc(O)c2nc(O)sc12".to_string(),
                "O=C(Nc1nnc(C(=O)Nc2ccc(N3CCOCC3)cc2)o1)c1ccc(Cl)cc1".to_string(),
            ],
        ]));
        let temp_dir = tempdir().unwrap();
        let path = temp_dir.path().to_str().unwrap();
        // Create instances of the required types
        let evolution_config = EvolutionConfig {
            evolution_steps: 3,
            population_size: 10,
            fp_size: 3,
            max_primitive_count: 5,
            max_bound_count: 4,
            bool_matching: true,
            gene_recombination_rate: 0.6,
            new_gene_rate: 0.6,
            proportion_new_population: 0.3,
            proportion_best_parents: 0.2,
            proportion_best_fingerprints: 0.5,
            work_folder: path.to_string(),
            gene_recombination_attempts: 5,
        };
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

            let regression_targets = Some(arr1(&[
                3.54, -1.18, 3.69, 3.37, 3.1, 3.14, -0.72, 0.34, 3.05, 2.25, 1.51,
            ]));

            // Call the function
            let result = create_the_one_fingerprint(
                &evolution_config,
                &fitness_function_config,
                features,
                &regression_targets,
                &None,
            );
            assert!(result.is_ok());
            // Check that the function returned the expected result
            match result {
                Ok(_) => println!("YAAAAS!!"),
                Err(err) => panic!("The function returned an error: {}", err),
            }

            // Delete the temporary directory
            temp_dir.close().unwrap();
        });
        Ok(())
    }

    #[test]
    fn test_big_create_the_one_fingerprint() -> std::io::Result<()> {
        let file_path = "/Users/alexandraebberg/Development/Github/rusty_evo_fp/src/impl_evo_fp/Lipophilicity.csv";
        let file = File::open(file_path).unwrap();
        let mut rdr = csv::ReaderBuilder::new().delimiter(b';').from_reader(&file);

        let target_data = rdr
            .records()
            .map(|r| {
                r.unwrap()[2]
                    .trim()
                    .replace(',', ".")
                    .parse::<f64>()
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let file_two = File::open(file_path).unwrap();
        let mut rdrtwo = csv::ReaderBuilder::new()
            .delimiter(b';')
            .from_reader(file_two);
        let data = rdrtwo
            .records()
            .map(|r| r.unwrap()[1].to_string())
            .collect::<Vec<_>>();
        let array_data = Array2::from_shape_vec((data.len(), 1), data).unwrap();
        let array_target_data = Array1::from_shape_vec(target_data.len(), target_data).unwrap();
        println!("Len targets: {:?}", array_target_data.len());

        let mol_data = convert_data_to_mol(&array_data);
        let temp_dir = tempdir().unwrap();
        let path = temp_dir.path().to_str().unwrap();
        // Create instances of the required types
        let evolution_config = EvolutionConfig {
            evolution_steps: 3,
            population_size: 10,
            fp_size: 5,
            max_primitive_count: 5,
            max_bound_count: 4,
            bool_matching: false,
            gene_recombination_rate: 0.15,
            new_gene_rate: 0.1,
            proportion_new_population: 0.2,
            proportion_best_parents: 0.4,
            proportion_best_fingerprints: 0.2,
            work_folder: path.to_string(),
            gene_recombination_attempts: 5,
        };
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
            let start = Instant::now();

            let result = create_the_one_fingerprint(
                &evolution_config,
                &fitness_function_config,
                mol_data,
                &Some(array_target_data),
                &None,
            );
            let duration = start.elapsed();
            println!("Time elapsed in expensive_function() is: {:?}", duration);

            assert!(result.is_ok());
            match result {
                Ok(_) => println!("YAAAAS!!"),
                Err(err) => panic!("The function returned an error: {}", err),
            }

            // Delete the temporary directory
            temp_dir.close().unwrap();
        });
        Ok(())
    }
}
