use crate::impl_evo_fp::primitives::primitive_property::*;
use crate::impl_evo_fp::smarts_fingerprint::smarts::*;
use crate::impl_evo_fp::smarts_fingerprint::smarts_pattern;
use crate::impl_evo_fp::utilities::*;

use rand::prelude::IteratorRandom;
use rand::Rng;
use rdkit::{ROMol, RWMol};

mod primitive_property;

#[derive(Clone, Debug)]
pub struct Primitives {
    pub primitive_list: Vec<String>,
}

impl Primitives {
    pub fn new(mol_data: &[Vec<ROMol>], bool_matching: bool) -> Primitives {
        // println!("Staring with primitives");
        let atoms = atoms_in_dataset(mol_data);
        let mut primitives: Vec<String> = atoms
            .iter()
            .map(|atom| ["#", &atom.to_string()].concat())
            .collect();
        primitives.append(&mut create_relev_property_list(mol_data, bool_matching));
        // @ - chirality anticlockwise, @@ - chirality clockwise
        for chirality in ["@", "@@"] {
            if is_primitive_useful(chirality, mol_data, bool_matching) {
                primitives.push(chirality.to_string());
            }
        }
        // println!("primitive count: {:?}", &primitives.len());
        // println!("{:?}", &primitives);
        Primitives {
            primitive_list: primitives,
        }
    }

    pub fn generate_primitive_pattern(
        primitive_count: u8,
        primitives: &Primitives,
    ) -> (String, usize) {
        let mut rng = rand::thread_rng();
        let mut step_counter = 1;
        let mut random_percentage: f32 = rng.gen();
        if random_percentage < 0.01 {
            return ("*".to_string(), step_counter);
        }
        loop {
            let mut pattern_string = String::from("[");
            for i in 0..primitive_count {
                let primitive_iter = primitives.primitive_list.iter();
                let mut primitive = primitive_iter.clone().choose(&mut rng).unwrap();
                while pattern_string.contains(primitive) {
                    primitive = primitive_iter.clone().choose(&mut rng).unwrap();
                }

                random_percentage = rng.gen::<f32>();
                if random_percentage > 0.5 {
                    pattern_string.push('!');
                }
                pattern_string.push_str(primitive);

                let logic_bi_operator = smarts_pattern::LOGIC_BI_OPERATIONS
                    .iter()
                    .choose(&mut rng)
                    .unwrap();
                if i != primitive_count - 1 {
                    pattern_string.push(*logic_bi_operator)
                }
            }
            pattern_string.push(']');

            if is_smarts_valid(&pattern_string) {
                return (pattern_string, step_counter);
            }
            step_counter += 1;
        }
    }
}
