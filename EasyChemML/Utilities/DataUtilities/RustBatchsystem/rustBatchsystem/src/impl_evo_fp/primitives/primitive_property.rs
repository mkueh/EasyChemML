use crate::impl_evo_fp::primitives::*;
use phf::phf_map;
use rdkit::ROMol;

static PROPERTY_LIST: phf::Map<&'static str, [i8; 2]> = phf_map! {
    "D"=> [0, 6],
    "h"=> [1, 4],
    "R"=> [1, 5],
    "r"=> [3, 21],
    "v"=> [1, 5],
    "X"=> [1, 7],
    "x"=> [0, 5],
    "-"=> [0, 3],
    "+"=> [0, 3],
    "z"=> [0, 5],
    "Z"=> [0, 5]
};

pub fn is_primitive_useful(
    primitive_str: &str,
    mol_data: &[Vec<ROMol>],
    bool_matching: bool,
) -> bool {
    let smarts_str = format!("[{}]", primitive_str);
    let mol_result = RWMol::from_smarts(&smarts_str);
    let mol_smarts = match mol_result {
        Ok(mol) => mol,
        Err(_e) => return false,
    };
    is_smarts_relevant(mol_smarts.to_ro_mol(), mol_data, bool_matching)
}
//noinspection ALL
pub fn create_relev_property_list(mol_data: &[Vec<ROMol>], bool_matching: bool) -> Vec<String> {
    let mut all_properties: Vec<String> = vec![];
    for (prop, bounds) in PROPERTY_LIST.into_iter() {
        let mut complete_property: Vec<String> = Vec::new();
        let boundaries =
            calc_relevant_boundaries(prop, bounds[0], bounds[1], mol_data, bool_matching);
        let intervals = create_boundary_intervals(boundaries);

        for interval in intervals {
            let property_int_str = format!("{}{{{}-{}}}", prop, interval.0, interval.1); //e. g. "D{1-6}", "D{2-3}"
            if is_primitive_useful(prop, mol_data, bool_matching) {
                complete_property.push(property_int_str);
            }
        }
        // println!(
        //     "Found {:?} property patterns: {:?}",
        //     prop, &complete_property
        // );
        all_properties.append(&mut complete_property);
    }
    all_properties
}

//noinspection ALL
fn calc_relevant_boundaries(
    prop_str: &str,
    start: i8,
    end: i8,
    mol_data: &[Vec<ROMol>],
    bool_matching: bool,
) -> [i8; 2] {
    let mut relev_boundaries: Vec<i8> = Vec::new();
    for num in start..=end {
        let boundary_str = format!("{}{{{}-{}}}", prop_str, num, num);
        if is_primitive_useful(&boundary_str, mol_data, bool_matching) {
            relev_boundaries.push(num);
        }
    }
    if relev_boundaries.is_empty() {
        [0, 0]
    } else {
        // return the relevant boundaries for the property (prop_str), e. g. D{1-(4} , [1, 4])
        [relev_boundaries[0], *relev_boundaries.last().unwrap()]
    }
}

fn create_boundary_intervals(boundaries: [i8; 2]) -> Vec<(i8, i8)> {
    let mut intervals = Vec::new();
    let start = boundaries[0];
    let end = boundaries[1];
    for i in start..=end {
        // add 1 to be inclusive
        for j in i..=end {
            intervals.push((i, j));
        }
    }
    intervals
}
