use rdkit::{substruct_match, ROMol, RWMol, SubstructMatchParameters};

///
///
///
/// # Arguments
/// SMART is relevant if there is a variance between lines
///
/// * `smart`: query molecule
/// * `mol`: molecule for query
/// * `bool_match`: is boolean matching activated? else: count matches
///
/// returns: bool
pub fn is_smarts_relevant(smarts: ROMol, mol_data: &[Vec<ROMol>], bool_matching: bool) -> bool {
    let first_row_matches: Vec<usize> = mol_data
        .first()
        .unwrap()
        .iter()
        .map(|mol| substruct_match(mol, &smarts, &SubstructMatchParameters::default()).len())
        .collect();

    for (col_index, first_match) in first_row_matches.iter().enumerate() {
        if bool_matching {
            let first_match = *first_match > 0usize;
            if mol_data.iter().any(|row| {
                let match_bool = matching_bool(&smarts, &row[col_index]);
                first_match != match_bool
            }) {
                return true;
            }
        } else if mol_data.iter().any(|row| {
            let match_num = matching_count(&smarts, &row[col_index]);
            *first_match != match_num
        }) {
            return true;
        }
    }
    false
}

pub fn is_smarts_valid(pattern: &str) -> bool {
    let mol_result = RWMol::from_smarts(pattern);
    match mol_result {
        Ok(rw_mol) => rw_mol.to_ro_mol().num_atoms(true) > 0,
        Err(_) => false,
    }
}

pub fn matching_count(smarts: &ROMol, mol: &ROMol) -> usize {
    substruct_match(mol, smarts, &SubstructMatchParameters::default()).len()
}

pub fn matching_bool(smarts: &ROMol, mol: &ROMol) -> bool {
    let match_num = substruct_match(mol, smarts, &SubstructMatchParameters::default()).len();
    match_num > 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_smarts_relevant_returns_true_when_boolean_matching_and_variance_exists() {
        let smarts = ROMol::from_smiles("C").unwrap();
        let mol1 = ROMol::from_smiles("S(=O)").unwrap();
        let mol2 = ROMol::from_smiles("CCC").unwrap();
        let mol3 = ROMol::from_smiles("o1").unwrap();
        let mol4 = ROMol::from_smiles("O").unwrap();
        let mol_data = vec![
            vec![mol1.clone(), mol3.clone()],
            vec![mol2.clone(), mol4.clone()],
        ];
        assert_eq!(is_smarts_relevant(smarts, &mol_data, true), true);
    }

    #[test]
    fn is_smarts_relevant_returns_false_when_boolean_matching_and_no_variance_exists() {
        let smarts = ROMol::from_smiles("C").unwrap();
        let mol1 = ROMol::from_smiles("CC").unwrap();
        let mol_data = vec![
            vec![mol1.clone(), mol1.clone()],
            vec![mol1.clone(), mol1.clone()],
        ];
        assert_eq!(is_smarts_relevant(smarts, &mol_data, true), false);
    }

    #[test]
    fn is_smarts_relevant_returns_true_when_count_matching_and_variance_exists() {
        let smarts = ROMol::from_smiles("C").unwrap();
        let mol1 = ROMol::from_smiles("S(=O)").unwrap();
        let mol2 = ROMol::from_smiles("CCC").unwrap();
        let mol_data = vec![vec![mol1.clone()], vec![mol2.clone()]];
        assert_eq!(is_smarts_relevant(smarts, &mol_data, false), true);
    }

    #[test]
    fn is_smarts_relevant_returns_false_when_count_matching_and_no_variance_exists() {
        let smarts = ROMol::from_smiles("C").unwrap();
        let mol1 = ROMol::from_smiles("CC").unwrap();
        let mol_data = vec![vec![mol1.clone()], vec![mol1.clone()]];
        assert_eq!(is_smarts_relevant(smarts, &mol_data, false), false);
    }
}
