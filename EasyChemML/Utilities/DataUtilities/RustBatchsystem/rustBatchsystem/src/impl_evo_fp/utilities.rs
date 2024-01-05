use itertools::Itertools;
use rdkit::ROMol;
pub fn get_elements_of_mol(mut mol: ROMol) -> Vec<i32> {
    let num_atoms = mol.num_atoms(true);
    let mut elements: Vec<i32> = Vec::new();
    for index in 0..num_atoms {
        let atom = &mut mol.atom_with_idx(index);
        elements.push(atom.get_atomic_num());
    }
    elements.sort();
    elements.dedup();
    elements
}

pub fn atoms_in_dataset(data: &Vec<ROMol>) -> Vec<i32> {
    let element_vec: Vec<i32> = data
        .iter()
        .map(|mol| get_elements_of_mol(mol.clone()))
        .flatten()
        .unique()
        .collect();
    println!("Found atoms: {:?}", &element_vec);
    element_vec
}
