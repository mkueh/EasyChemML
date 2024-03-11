use crate::impl_evo_fp::fitness_functions::single_dataset_fitness::{
    calculate_fingerprint_fitness, FitnessFunctionConfig,
};
use crate::impl_evo_fp::population::member::Member;
use crate::impl_evo_fp::smarts_fingerprint::SmartsFingerprint;
use ndarray::{Array, Ix1};
use rdkit::ROMol;
use serde::{Deserialize, Serialize};
use std::fs;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;
use walkdir::WalkDir;

pub mod member;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Population {
    pub members: Vec<Member>,
    pub path: Option<String>,
}

impl Population {
    pub fn empty() -> Population {
        Population {
            members: Vec::new(),
            path: None,
        }
    }
    pub fn generate_new_population(
        population_size: usize,
        fp_size: usize,
        max_primitive_count: u8,
        max_bound_count: u8,
        bool_matching: bool,
        data: &[Vec<ROMol>],
    ) -> Population {
        let fingerprints = SmartsFingerprint::generate_smarts_fingerprints(
            population_size,
            fp_size,
            max_primitive_count,
            max_bound_count,
            bool_matching,
            data,
        );
        let mut members: Vec<Member> = Vec::new();
        for fp in fingerprints {
            members.push(Member::new(fp));
        }
        Population {
            members,
            path: None,
        }
    }
    pub fn calculate_population_metrics(
        &mut self,
        features: &[Vec<ROMol>],
        regression_targets: Option<Array<f64, Ix1>>,
        classification_targets: Option<Array<usize, Ix1>>,
        bool_matching: bool,
        fitness_function_config: FitnessFunctionConfig,
    ) {
        self.members.iter_mut().for_each(|member| {
            if member.metric.is_none() {
                member.metric = calculate_fingerprint_fitness(
                    member,
                    features,
                    &regression_targets,
                    &classification_targets,
                    bool_matching,
                    &fitness_function_config,
                );
            }
        });
    }

    pub fn n_best_members(&mut self, n: usize) -> Vec<Member> {
        if n > self.members.len() {
            panic!("Cannot get more best members than there are members in the population");
        }
        // Sort in descending order
        self.members.sort_unstable_by(|member_a, member_b| {
            member_b
                .metric
                .unwrap()
                .partial_cmp(&member_a.metric.unwrap())
                .unwrap()
        });
        let result_vec = &self.members[0..n].to_vec();
        result_vec.to_owned()
    }

    pub fn save_population(&mut self, path: &str, step: i8) -> Result<(), std::io::Error> {
        let file_path = &format!("{}/evolution_step_{}.bin", path, step);
        if Path::new(file_path).exists() {
            fs::remove_file(file_path)?;
        }
        let mut writer = BufWriter::new(File::create(file_path)?);
        let serializing_result = bincode::serialize_into(&mut writer, &self);
        if serializing_result.is_err() {
            panic!("Could not serialize population")
        }
        writer.flush()?;
        self.path = Some(file_path.to_string());
        drop(writer);
        Ok(())
    }

    pub fn saved_population(path: &str, found_max_step: i8) -> Result<Population, std::io::Error> {
        let file_path = &format!("{}/evolution_step_{}.bin", path, found_max_step);
        let reader = BufReader::new(File::open(file_path).unwrap());
        let deserialized_population = bincode::deserialize_from(reader).unwrap();
        Ok(deserialized_population)
    }

    pub fn highest_saved_population(
        path: &str,
    ) -> Result<(Option<Population>, i8), std::io::Error> {
        let mut found_max_step: u32 = 0;
        const RADIX: u32 = 10;

        for entry in WalkDir::new(path).into_iter().filter_map(|e| e.ok()) {
            println!("{:?}", entry.path());
            if entry.file_type().is_file()
                && entry.path().extension().is_some()
                && entry.path().extension().unwrap() == "bin"
                && entry.path().to_str().unwrap().contains("evolution_step_")
            {
                let step = entry
                    .path()
                    .file_stem()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .chars()
                    .last()
                    .unwrap()
                    .to_digit(RADIX)
                    .unwrap();
                if step > found_max_step {
                    found_max_step = step;
                }
            }
        }
        if found_max_step == 0 {
            Ok((None, 0))
        } else {
            let deserialized_population = Population::saved_population(path, found_max_step as i8)?;
            println!("{:?}", deserialized_population);
            Ok((Some(deserialized_population), found_max_step as i8))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::impl_evo_fp::fitness_functions::fitness_metrics::Metric::RegressionMetric;
    use crate::impl_evo_fp::fitness_functions::fitness_metrics::RegressionMetric::R2ScoreSingleTarget;
    use crate::impl_evo_fp::fitness_functions::fitness_models::Model::CatBoostRegressor;
    use crate::impl_evo_fp::get_data::convert_data_to_mol;
    use ndarray::{arr1, arr2};
    use pyo3::{IntoPy, Python};
    use std::collections::HashMap;
    use tempfile::tempdir;

    #[test]
    fn test_calculate_population_metrics() {
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
        let mut population = Population::generate_new_population(10, 5, 5, 5, false, &features);
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

            for member in &population.members {
                assert!(
                    member.metric.is_none(),
                    "Metric should not be calculated for each member beforehand"
                );
            }

            population.calculate_population_metrics(
                &features,
                regression_targets,
                None,
                bool_matching,
                fitness_function_config,
            );

            for member in &population.members {
                assert!(
                    member.metric.is_some(),
                    "Metric should be calculated for each member"
                );
            }
        });
    }

    #[test]
    fn test_save_population() {
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
        let mut population = Population::generate_new_population(10, 5, 5, 5, false, &features);
        let temp_dir = tempdir().unwrap();
        let path = temp_dir.path().to_str().unwrap();
        let step = 3;
        let saving_result = &mut population.save_population(path, step);
        let file_path = &format!("{}/evolution_step_{}.bin", path, step);
        let file = BufReader::new(File::open(file_path).unwrap());
        let deserialized_population: Population = bincode::deserialize_from(file).unwrap();
        assert_eq!(
            population.members.len(),
            deserialized_population.members.len()
        );
        assert_eq!(
            population.members[0].fingerprint.patterns,
            deserialized_population.members[0].fingerprint.patterns
        );
        assert_eq!(saving_result.is_ok(), true);
        temp_dir.close().unwrap();
    }

    #[test]
    fn test_get_saved_population() {
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
        let mut population = Population::generate_new_population(10, 5, 5, 5, false, &features);
        let temp_dir = tempdir().unwrap();
        let path = temp_dir.path().to_str().unwrap();
        let step = 3;
        let _ = &mut population.save_population(path, step);
        let deserialization_result = Population::saved_population(path, step);
        match deserialization_result {
            Ok(deserialized_population) => {
                // Add assertions to check the deserialized_population
                // For example, if the Population struct has a field named size, you can check that
                assert_eq!(
                    deserialized_population.members.len(),
                    population.members.len(),
                    "The function should return the same population that was saved"
                );
            }
            Err(err) => panic!("The function returned an error: {}", err),
        }
    }
    #[test]
    fn test_get_highest_saved_population() {
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
        let mut population = Population::generate_new_population(10, 5, 5, 5, false, &features);
        let temp_dir = tempdir().unwrap();
        let path = temp_dir.path().to_str().unwrap();
        let step = 3;
        let _ = &mut population.save_population(path, step);

        for i in 0..3 {
            let file_path = format!("{}/evolution_step_{}.bin", path, i);
            let mut file = File::create(&file_path).expect("Unable to create file");
            file.write_all(&[i]).expect("Unable to write data");
        }

        let deserialization_result = Population::highest_saved_population(path);
        match deserialization_result {
            Ok((Some(deserialized_population), _)) => {
                assert_eq!(
                    deserialized_population.members.len(),
                    population.members.len(),
                    "The function should return the population with the highest step"
                );
            }
            Ok((None, _)) => {
                panic!("The function should return Some(Population), but it returned None")
            }
            Err(err) => panic!("The function returned an error: {}", err),
        }

        // Delete the temporary directory
        temp_dir.close().unwrap();
    }
}
