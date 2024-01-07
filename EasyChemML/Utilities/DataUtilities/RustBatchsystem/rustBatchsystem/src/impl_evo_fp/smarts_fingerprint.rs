use crate::impl_evo_fp::smarts_fingerprint::smarts_pattern::SMARTSPattern;
use rand::prelude::SliceRandom;
use rand::thread_rng;
use rdkit::ROMol;
use serde::{Deserialize, Serialize};

pub mod smarts;
pub mod smarts_pattern;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct SmartsFingerprint {
    pub patterns: Vec<SMARTSPattern>,
}

impl SmartsFingerprint {
    pub fn new(smarts_patterns: Vec<SMARTSPattern>) -> SmartsFingerprint {
        SmartsFingerprint {
            patterns: smarts_patterns,
        }
    }
    pub fn get_number_matches(&self, bool_matching: bool, mol: &ROMol) -> Vec<usize> {
        if bool_matching {
            let mut matches = Vec::new();
            for pattern in &self.patterns {
                matches
                    .push(smarts::get_matching_bool(&pattern.to_ro_mol().unwrap(), &mol) as usize)
            }
            matches
        } else {
            let mut matches = Vec::new();
            for pattern in &self.patterns {
                matches.push(smarts::get_matching_count(
                    &pattern.to_ro_mol().unwrap(),
                    &mol,
                ))
            }
            matches
        }
    }

    pub fn generate_smarts_fingerprints(
        population_size: u8,
        fp_size: u8,
        max_primitive_count: u8,
        max_bound_count: u8,
        bool_matching: bool,
        data: &Vec<ROMol>,
    ) -> Vec<SmartsFingerprint> {
        let total_smarts_pattern_number = population_size * fp_size;
        let mut patterns_counter = 0;
        let mut smarts_patterns: Vec<SMARTSPattern> = Vec::new();

        while patterns_counter < total_smarts_pattern_number {
            smarts_patterns.push(SMARTSPattern::generate_smarts_pattern(
                max_primitive_count,
                max_bound_count,
                bool_matching,
                data,
            ));
            patterns_counter += 1;
        }
        smarts_patterns.shuffle(&mut thread_rng());

        let patterns_split: Vec<&[SMARTSPattern]> =
            smarts_patterns.chunks(population_size as usize).collect();
        let fingerprints: Vec<SmartsFingerprint> = patterns_split
            .iter()
            .map(|chunk| SmartsFingerprint::new(chunk.to_vec()))
            .collect();
        fingerprints
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::impl_evo_fp::get_data::convert_data_to_mol;
    use ndarray::arr2;

    #[test]
    fn test_generate_smarts_fingerprints() {
        let population_size = 5;
        let fp_size = 5;
        let max_primitive_count = 5;
        let max_bound_count = 5;
        let bool_matching = true;
        let data = convert_data_to_mol(&arr2(&[
            ["Cn1c(CN2CCN(c3ccc(Cl)cc3)CC2)nc2ccccc21".to_string()],
            ["COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1".to_string()],
            ["O=C(NC1Cc2ccccc2N(C[C@@H](O)CO)C1=O)c1cc2cc(Cl)sc2[nH]1".to_string()],
            ["Cc1cccc(C[C@H](NC(=O)c2cc(C(C)(C)C)nn2C)C(=O)NCC#N)c1".to_string()],
            ["OC1(C#Cc2ccc(-c3ccccc3)cc2)CN2CCC1CC2".to_string()],
            ["COc1cc(OC)c(S(=O)(=O)NCc2ccccc2N2CCCCC2)cc1NC(=O)CCC(=O)O".to_string()],
        ]));
        let fingerprints = SmartsFingerprint::generate_smarts_fingerprints(
            population_size,
            fp_size,
            max_primitive_count,
            max_bound_count,
            bool_matching,
            &data,
        );
        assert_eq!(fingerprints.len(), population_size as usize);
    }
}
