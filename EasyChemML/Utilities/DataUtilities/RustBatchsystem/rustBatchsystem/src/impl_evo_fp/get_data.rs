use crate::impl_evo_fp::population::generate_new_population;
use crate::impl_evo_fp::smarts_fingerprint::smarts_pattern::SMARTSPattern;
use crate::impl_evo_fp::smarts_fingerprint::SmartsFingerprint;
use itertools::enumerate;
use ndarray::{arr2, Array2};
use rdkit::ROMol;

pub fn convert_data_to_mol(data: &Array2<String>) -> Vec<ROMol> {
    let mol_data: Vec<ROMol> = data
        .iter()
        .map(|row| {
            ROMol::from_smiles(row).expect("Cannot transform smile to mol in convert_data_to_mol")
        })
        .collect();
    mol_data
}

pub fn generate_regression_feature_data(
    fingerprint: &SmartsFingerprint,
    data: &Vec<ROMol>,
    bool_matching: bool,
) -> Array2<f64> {
    let mut feature_array: Array2<f64> = Array2::zeros((data.len(), fingerprint.patterns.len()));

    for (row_index, entry) in enumerate(data) {
        let matches = fingerprint.get_number_matches(bool_matching, entry);
        for (col_index, value) in matches.iter().enumerate() {
            feature_array[[row_index, col_index]] = *value as f64;
        }
    }
    feature_array
}

pub fn generate_classification_feature_data(
    fingerprint: &SmartsFingerprint,
    data: &Vec<ROMol>,
    bool_matching: bool,
) -> Array2<usize> {
    let mut feature_array: Array2<usize> = Array2::zeros((data.len(), fingerprint.patterns.len()));

    for (row_index, entry) in enumerate(data) {
        let matches = fingerprint.get_number_matches(bool_matching, entry);
        for (col_index, value) in matches.iter().enumerate() {
            feature_array[[row_index, col_index]] = *value;
        }
    }
    feature_array
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_regression_feature_data() {
        let data = convert_data_to_mol(&arr2(&[
            ["Cn1c(CN2CCN(c3ccc(Cl)cc3)CC2)nc2ccccc21".to_string()],
            ["COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1".to_string()],
            ["O=C(NC1Cc2ccccc2N(C[C@@H](O)CO)C1=O)c1cc2cc(Cl)sc2[nH]1".to_string()],
            ["Cc1cccc(C[C@H](NC(=O)c2cc(C(C)(C)C)nn2C)C(=O)NCC#N)c1".to_string()],
            ["OC1(C#Cc2ccc(-c3ccccc3)cc2)CN2CCC1CC2".to_string()],
            ["COc1cc(OC)c(S(=O)(=O)NCc2ccccc2N2CCCCC2)cc1NC(=O)CCC(=O)O".to_string()],
        ]));
        let population = generate_new_population(3, 5, 5, 5, true, &data);
        let fp = population.members[0].fingerprint.clone();
        let bool_matching = false;
        let feature_data = generate_regression_feature_data(&fp, &data, bool_matching);
        assert_eq!(feature_data.shape(), &[6, 3]);
    }
}
