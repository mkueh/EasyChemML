use itertools::Itertools;
use rdkit::ROMol;
pub fn elements_of_mol(mut mol: ROMol) -> Vec<i32> {
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

pub fn atoms_in_dataset(data: &[Vec<ROMol>]) -> Vec<i32> {
    let element_vec: Vec<i32> = data
        .iter()
        .flatten()
        .flat_map(|mol| elements_of_mol(mol.clone()))
        .unique()
        .collect::<Vec<i32>>();
    // println!("Found atoms: {:?}", &element_vec);
    element_vec
}

#[cfg(test)]
mod tests {
    use super::*;
    use rdkit::ROMol;

    #[test]
    fn elements_of_single_molecule() {
        let mol = ROMol::from_smiles("CCO").unwrap();
        let result = elements_of_mol(mol);
        assert_eq!(result, vec![6, 8]);
    }

    #[test]
    fn elements_of_empty_molecule() {
        let mol = ROMol::from_smiles("").unwrap();
        let result = elements_of_mol(mol);
        assert_eq!(result, vec![]);
    }

    #[test]
    fn atoms_in_single_dataset() {
        let data = vec![vec![
            ROMol::from_smiles("CCO").unwrap(),
            ROMol::from_smiles("CCN").unwrap(),
        ]];
        let result = atoms_in_dataset(&data);
        assert_eq!(result, vec![6, 7, 8]);
    }

    #[test]
    fn atoms_in_dataset_with_empty_molecules() {
        let data = vec![vec![
            ROMol::from_smiles("").unwrap(),
            ROMol::from_smiles("").unwrap(),
        ]];
        let result = atoms_in_dataset(&data);
        assert_eq!(result, vec![]);
    }
}
