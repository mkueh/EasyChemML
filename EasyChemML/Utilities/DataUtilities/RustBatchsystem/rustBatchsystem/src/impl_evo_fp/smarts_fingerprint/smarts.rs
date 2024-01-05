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
pub fn is_smarts_relevant(smarts: ROMol, mol_data: &Vec<ROMol>, bool_matching: bool) -> bool {
    let first_match = substruct_match(
        mol_data.first().unwrap(),
        &smarts,
        &SubstructMatchParameters::default(),
    )
    .len();
    for mol in mol_data.iter() {
        if bool_matching {
            let match_bool = get_matching_bool(&smarts, &mol);
            if first_match != (match_bool as usize) {
                return true;
            }
        } else {
            let match_num = get_matching_count(&smarts, &mol);
            if first_match != match_num {
                return true;
            }
        }
    }
    false
}
pub fn is_smarts_valid(pattern: &str) -> bool {
    let mol_result = RWMol::from_smarts(pattern);
    return match mol_result {
        Ok(_) => true,
        Err(_) => false,
    };
}

pub fn get_matching_count(smarts: &ROMol, mol: &ROMol) -> usize {
    let match_num = substruct_match(mol, smarts, &SubstructMatchParameters::default()).len();
    match_num
}

pub fn get_matching_bool(smarts: &ROMol, mol: &ROMol) -> bool {
    let match_num = substruct_match(mol, smarts, &SubstructMatchParameters::default()).len();
    if match_num > 0 {
        true
    } else {
        false
    }
}
