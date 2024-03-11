use crate::impl_evo_fp::primitives::Primitives;
use crate::impl_evo_fp::smarts_fingerprint::smarts::*;

use rand::Rng;
use rdkit::{ROMol, RWMol};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::iter::zip;
use uuid::Uuid;

pub static BONDS: [char; 6] = [
    // bonds ->  atoms + bonds + atoms
    '-', // single bond (aliphatic)
    //'/',           //directional bond "up"
    //'\ '.strip(),  //directional bond "down"
    //'/?',          //directional bond "up or unspecified"
    //'\?',          //directional bond "down or unspecified"
    '=', //double bond
    '#', //triple bond
    ':', //aromatic bond
    '~', //any bond (wildcard)
    '@',
]; //any ring bond

pub static LOGIC_BI_OPERATIONS: [char; 1] = [
    //'&', //e1&e2 -> e1 and e2 (high precedence) (NOT SUPPORTED BY RDKIT)
    //',', //e1,e2 -> e1 or e2
    ';', //e1;e2 -> e1 and e2 (low precedence)
];

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct SMARTSPattern {
    pub(crate) atomics: Vec<String>,
    pub(crate) bonds: Vec<char>,
    create_info: HashMap<String, String>,
    id: Uuid,
}
#[derive(Debug, PartialEq, thiserror::Error)]
pub enum SMARTSPatternError {
    #[error("Could not convert RWMol to ROMol")]
    UnknownConversionError,

    #[error("Could not convert RWMol to ROMol (empty ROMol)")]
    EmptyMolError,
}

impl SMARTSPattern {
    pub fn new(
        atomics: Vec<String>,
        bonds: Vec<char>,
        create_info: HashMap<String, String>,
    ) -> SMARTSPattern {
        SMARTSPattern {
            atomics,
            bonds,
            create_info,
            id: Uuid::new_v4(),
        }
    }
    pub fn generate_smarts_pattern(
        max_primitive_count: u8,
        max_bound_count: u8,
        bool_matching: bool,
        data: &[Vec<ROMol>],
    ) -> SMARTSPattern {
        let mut rng = rand::thread_rng();
        // let mut step_counter: usize = 0;
        let primitives = Primitives::new(data, bool_matching);

        loop {
            let bound_count = if max_bound_count == 1 {
                1
            } else {
                rng.gen_range(1..=max_bound_count)
            };
            let mut bonds = Vec::new();
            let mut atomics = Vec::new();
            for i in 0..=bound_count {
                let primitive_count = if max_primitive_count == 1 && bound_count == 0 {
                    1
                } else {
                    rng.gen_range(bound_count..=max_primitive_count)
                };
                // returns (atomics, step_counter)
                let pattern_tuple =
                    Primitives::generate_primitive_pattern(primitive_count, &primitives);
                atomics.push(pattern_tuple.0);
                // step_counter += pattern_tuple.1;

                // bonds are between primitives so one iteration less
                if i == bound_count {
                    break;
                }
                let bond = BONDS[rng.gen_range(0..BONDS.len())];
                bonds.push(bond);
            }

            let pattern = SMARTSPattern {
                bonds,
                atomics,
                create_info: HashMap::from([("createTyp".to_string(), "NewGenerator".to_string())]),
                id: Uuid::new_v4(),
            };
            // check if pattern is relevant for the data
            // each pattern in atomics is a valid smarts (tested in generate primitive pattern)
            if is_smarts_relevant(pattern.to_ro_mol().unwrap(), data, bool_matching) {
                // println!("Finished SMARTS generation in {} steps", step_counter);
                return pattern;
            }
        }
    }
    pub fn to_rw_mol(&self) -> Result<RWMol, Box<dyn Error>> {
        let smarts = self.to_string();
        match RWMol::from_smarts(&smarts) {
            Err(e) => Err(e),
            Ok(mol) => Ok(mol),
        }
    }
    pub fn to_ro_mol(&self) -> Result<ROMol, SMARTSPatternError> {
        let rw_mol = self.to_rw_mol().expect("Could not convert to RWMol");
        let ro_mol = rw_mol.to_ro_mol();
        if ro_mol.num_atoms(true) == 0 {
            return Err(SMARTSPatternError::EmptyMolError);
        }
        Ok(ro_mol)
    }

    pub fn genetic_slice(
        &self,
        start: usize,
        end: usize,
        include_end_bond: bool,
    ) -> (Vec<String>, Vec<char>) {
        if include_end_bond {
            (
                self.atomics[start..=end].to_vec(),
                self.bonds[start..=end].to_vec(),
            )
        } else {
            (
                self.atomics[start..=end].to_vec(),
                self.bonds[start..end].to_vec(),
            )
        }
    }

    pub fn random_gene_intervall(&self) -> (usize, usize) {
        let mut rng = rand::thread_rng();
        let start = rng.gen_range(0..self.atomics.len());
        let end = rng.gen_range(start..self.atomics.len());
        if start == end && end != self.atomics.len() - 1 {
            return (start, end + 1);
        } else if start == end && end == self.atomics.len() - 1 {
            return (start - 1, end);
        }
        (start, end)
    }
}
impl Display for SMARTSPattern {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut result_string = String::new();
        let iter = zip(&self.atomics, &self.bonds);
        for (atomic, bond) in iter {
            result_string.push_str(atomic);
            result_string.push(*bond);
        }
        result_string.push_str(self.atomics.last().unwrap_or(&"".to_string()));

        write!(f, "{}", result_string)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::impl_evo_fp::get_data::convert_data_to_mol;
    use ndarray::arr2;

    #[test]
    fn test_generate_smarts_pattern() {
        let data = convert_data_to_mol(&arr2(&[
            ["Cn1c(CN2CCN(c3ccc(Cl)cc3)CC2)nc2ccccc21".to_string()],
            ["COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1".to_string()],
            ["O=C(NC1Cc2ccccc2N(C[C@@H](O)CO)C1=O)c1cc2cc(Cl)sc2[nH]1".to_string()],
            ["Cc1cccc(C[C@H](NC(=O)c2cc(C(C)(C)C)nn2C)C(=O)NCC#N)c1".to_string()],
            ["OC1(C#Cc2ccc(-c3ccccc3)cc2)CN2CCC1CC2".to_string()],
            ["COc1cc(OC)c(S(=O)(=O)NCc2ccccc2N2CCCCC2)cc1NC(=O)CCC(=O)O".to_string()],
        ]));
        let max_primitive_count = 3;
        let max_bound_count = 2;
        let bool_matching = true;

        let pattern = SMARTSPattern::generate_smarts_pattern(
            max_primitive_count,
            max_bound_count,
            bool_matching,
            &data,
        );

        assert_eq!(pattern.atomics.len(), pattern.bonds.len() + 1);
        assert!(pattern.to_rw_mol().is_ok());
    }

    #[test]
    fn test_to_rw_mol() {
        let pattern = SMARTSPattern {
            atomics: vec!["C".to_string(), "O".to_string()],
            bonds: vec!['-'],
            create_info: HashMap::new(),
            id: Uuid::new_v4(),
        };

        let rw_mol = pattern.to_rw_mol();

        assert!(rw_mol.is_ok());
    }

    #[test]
    fn test_to_ro_mol() {
        let pattern = SMARTSPattern {
            atomics: vec!["C".to_string(), "O".to_string()],
            bonds: vec!['-'],
            create_info: HashMap::new(),
            id: Uuid::new_v4(),
        };

        let ro_mol = pattern.to_ro_mol();

        assert!(ro_mol.is_ok());
    }

    #[test]
    fn test_display() {
        let pattern = SMARTSPattern {
            atomics: vec!["C".to_string(), "O".to_string()],
            bonds: vec!['-'],
            create_info: HashMap::new(),
            id: Uuid::new_v4(),
        };

        let display_string = pattern.to_string();

        assert_eq!(display_string, "C-O");
    }
}
