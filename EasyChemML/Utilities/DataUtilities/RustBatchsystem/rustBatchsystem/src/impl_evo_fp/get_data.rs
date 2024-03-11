use crate::impl_evo_fp::smarts_fingerprint::SmartsFingerprint;
use itertools::enumerate;
use ndarray::{Array, Array2, Axis};
use rdkit::ROMol;

pub fn convert_data_to_mol(data: &Array2<String>) -> Vec<Vec<ROMol>> {
    let mol_vec: Vec<Vec<ROMol>> = data
        .axis_iter(Axis(0))
        .map(|row| {
            row.iter()
                .map(|smiles| {
                    ROMol::from_smiles(smiles)
                        .expect("Cannot transform smiles to mol in convert_data_to_mol")
                })
                .collect()
        })
        .collect();
    mol_vec
}

pub fn bench_convert_data_to_mol(data: &Vec<Vec<String>>) -> Vec<Vec<ROMol>> {
    let mol_vec: Vec<Vec<ROMol>> = data
        .iter()
        .map(|row| {
            row.iter()
                .map(|smiles| {
                    ROMol::from_smiles(smiles)
                        .expect("Cannot transform smiles to mol in convert_data_to_mol")
                })
                .collect()
        })
        .collect();
    mol_vec
}

pub fn generate_regression_feature_data(
    fingerprint: &SmartsFingerprint,
    data: &[Vec<ROMol>],
    bool_matching: bool,
) -> Array2<f64> {
    let feature_array_length = fingerprint.patterns.len() * data[0].len();

    let mut feature_array: Array2<f64> = Array::zeros((data.len(), feature_array_length));

    for (row_index, row) in enumerate(data) {
        for (col_index, entry) in enumerate(row) {
            let matches = fingerprint.number_substructure_matches(bool_matching, entry);
            // match value ist eine Zahl, 0 oder 1
            for (pattern_index, match_value) in enumerate(matches) {
                feature_array[[
                    row_index,
                    col_index * fingerprint.patterns.len() + pattern_index,
                ]] = match_value as f64;
            }
        }
    }
    feature_array
}

pub fn generate_classification_feature_data(
    fingerprint: &SmartsFingerprint,
    data: &[Vec<ROMol>],
    bool_matching: bool,
) -> Array2<usize> {
    let feature_array_length = fingerprint.patterns.len() * data[0].len();
    let mut feature_array: Array2<usize> = Array::zeros((0, feature_array_length));

    for row in data.iter() {
        let mut row_array = Array::zeros(0);
        for entry in row.iter() {
            // dimension of matches: (1 x fingerprint.patterns.len())
            let matches =
                Array::from_vec(fingerprint.number_substructure_matches(bool_matching, entry));
            row_array.append(Axis(0), matches.view()).unwrap();
        }
        feature_array
            .append(
                Axis(0),
                row_array
                    .view()
                    .into_shape((1, feature_array_length))
                    .unwrap(),
            )
            .unwrap();
    }
    feature_array
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::impl_evo_fp::population::Population;
    use ndarray::arr2;

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
        let population = Population::generate_new_population(3, 5, 5, 5, true, &data);
        let fp = population.members[0].fingerprint.clone();
        let bool_matching = false;
        let feature_data = generate_regression_feature_data(&fp, &data, bool_matching);
        assert_eq!(feature_data.shape(), &[6, 5]);
    }
    #[test]
    fn test_generate_regression_mutli_column_feature_data() {
        let data = convert_data_to_mol(&arr2(&[
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
        ]));
        let population = Population::generate_new_population(3, 5, 5, 5, true, &data);
        let fp = population.members[0].fingerprint.clone();
        let bool_matching = false;
        let feature_data = generate_regression_feature_data(&fp, &data, bool_matching);
        assert_eq!(feature_data.shape(), &[3, 10]);
    }

    #[test]
    fn test_generate_classification_feature_data() {
        let data = convert_data_to_mol(&arr2(&[
            ["Cn1c(CN2CCN(c3ccc(Cl)cc3)CC2)nc2ccccc21".to_string()],
            ["COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1".to_string()],
            ["O=C(NC1Cc2ccccc2N(C[C@@H](O)CO)C1=O)c1cc2cc(Cl)sc2[nH]1".to_string()],
            ["Cc1cccc(C[C@H](NC(=O)c2cc(C(C)(C)C)nn2C)C(=O)NCC#N)c1".to_string()],
            ["OC1(C#Cc2ccc(-c3ccccc3)cc2)CN2CCC1CC2".to_string()],
            ["COc1cc(OC)c(S(=O)(=O)NCc2ccccc2N2CCCCC2)cc1NC(=O)CCC(=O)O".to_string()],
        ]));
        let population = Population::generate_new_population(3, 5, 5, 5, true, &data);
        let fp = population.members[0].fingerprint.clone();
        let bool_matching = false;
        let feature_data = generate_classification_feature_data(&fp, &data, bool_matching);
        assert_eq!(feature_data.shape(), &[6, 5]);
    }

    #[test]
    fn test_generate_classification_multi_column_feature_data() {
        let data = convert_data_to_mol(&arr2(&[
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
        ]));
        let population = Population::generate_new_population(3, 5, 5, 5, true, &data);
        let fp = population.members[0].fingerprint.clone();
        let bool_matching = false;
        let feature_data = generate_classification_feature_data(&fp, &data, bool_matching);
        assert_eq!(feature_data.shape(), &[3, 10]);
    }

    #[test]
    fn convert_data_to_mol_handles_valid_input() {
        let data = arr2(&[
            ["Cn1c(CN2CCN(c3ccc(Cl)cc3)CC2)nc2ccccc21".to_string()],
            ["COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1".to_string()],
            ["O=C(NC1Cc2ccccc2N(C[C@@H](O)CO)C1=O)c1cc2cc(Cl)sc2[nH]1".to_string()],
            ["Cc1cccc(C[C@H](NC(=O)c2cc(C(C)(C)C)nn2C)C(=O)NCC#N)c1".to_string()],
            ["OC1(C#Cc2ccc(-c3ccccc3)cc2)CN2CCC1CC2".to_string()],
            ["COc1cc(OC)c(S(=O)(=O)NCc2ccccc2N2CCCCC2)cc1NC(=O)CCC(=O)O".to_string()],
        ]);
        let result = convert_data_to_mol(&data);
        assert_eq!(result.len(), 6);
        assert_eq!(result[0].len(), 2);
    }

    #[test]
    fn convert_data_to_mol_handles_multi_column_input() {
        let data = arr2(&[
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
        ]);
        let result = convert_data_to_mol(&data);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].len(), 2);
    }
}
