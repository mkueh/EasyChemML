use crate::impl_evo_fp::fingerprint_evolution::{EvolutionConfig, FingerprintEvolutionError};
use crate::impl_evo_fp::population::member::Member;
use crate::impl_evo_fp::smarts_fingerprint::smarts::is_smarts_relevant;
use crate::impl_evo_fp::smarts_fingerprint::smarts_pattern::SMARTSPattern;
use crate::impl_evo_fp::smarts_fingerprint::{smarts_pattern, SmartsFingerprint};
use itertools::Itertools;
use rand::prelude::SliceRandom;
use rand::Rng;
use rdkit::ROMol;
use std::cmp::max;
use std::collections::HashMap;

pub fn evolve_population(
    evolution_config: &EvolutionConfig,
    number_kids: usize,
    parents: &[Member],
    feature_data: &Vec<ROMol>,
) -> Vec<Member> {
    let mut rng = rand::thread_rng();
    let mut kids = Vec::new();

    for _ in 0..number_kids {
        let eltern = &parents
            .choose_multiple(&mut rng, 2)
            .cloned()
            .collect_tuple()
            .unwrap();
        kids.push(mutate(
            eltern,
            &evolution_config,
            feature_data,
        ))
    }
    println!("Mutation complete!");
    kids
}

fn mutate(
    parents: &(Member, Member),
    evolution_config: &EvolutionConfig,
    feature_data: &Vec<ROMol>,
) -> Member {
    loop {
        let mut rng = rand::thread_rng();
        let mut new_genes = Vec::new();
        let mut mother = parents.0.fingerprint.clone();
        mother.patterns.shuffle(&mut rng);
        let mut father = parents.1.fingerprint.clone();
        father.patterns.shuffle(&mut rng);

        for i in 0..evolution_config.fp_size {
            let mutation_probability = rng.gen::<f32>();
            if mutation_probability <= evolution_config.gene_recombination_rate {
                let mut mutter = mother.patterns[i].clone();
                let mut vater = father.patterns[i].clone();
                // Father and mother genes are mutated
                let sliced_pattern = slice_mutation(
                    evolution_config,
                    (&mut vater, &mut mutter),
                    feature_data,
                );
                if sliced_pattern.is_ok() {
                    new_genes.push(sliced_pattern.unwrap());
                } else {
                    continue;
                }
            } else if mutation_probability
                >= (evolution_config.gene_recombination_rate + evolution_config.new_gene_rate)
            {
                // Father or mother genes are chosen
                new_genes.push(choice_mutation((&father.patterns[i], &mother.patterns[i])));
            } else {
                new_genes.push(SMARTSPattern::generate_smarts_pattern(
                    evolution_config.max_primitive_count,
                    evolution_config.max_bound_count,
                    evolution_config.bool_matching,
                    feature_data,
                ));
            };
        }
        return Member::new(SmartsFingerprint::new(new_genes));
    }
}

fn choice_mutation(parents: (&SMARTSPattern, &SMARTSPattern)) -> SMARTSPattern {
    let mut rng = rand::thread_rng();
    let gender_probability = rng.gen::<f32>();
    // Either father or mother is chosen
    if gender_probability >= 0.5 {
        parents.0.clone()
    } else {
        parents.1.clone()
    }
}

fn slice_mutation(
    evolution_config: &EvolutionConfig,
    parents: (&mut SMARTSPattern, &mut SMARTSPattern),
    feature_data: &Vec<ROMol>,
) -> Result<SMARTSPattern, FingerprintEvolutionError> {
    for _ in 0..evolution_config.gene_recombination_attempts {
        let mut new_atomics = Vec::new();
        let mut new_bonds = Vec::new();
        let mut rng = rand::thread_rng();
        let end_bond_probability = rng.gen::<f32>();
        let (mother_start, mother_end) = &parents.0.random_gene_intervall();
        let (father_start, father_end) = &parents.1.random_gene_intervall();

        // Order of appending matters for the genes!
        if end_bond_probability >= 0.5 && father_end < &(&parents.1.atomics.len() - 1) {
            let (mother_slice_atomics, mother_slice_bonds) =
                &parents
                    .0
                    .genetic_slice(mother_start.clone(), mother_end.clone(), false);
            let (father_slice_atomics, father_slice_bonds) =
                &parents
                    .1
                    .genetic_slice(father_start.clone(), father_end.clone(), true);
            new_atomics.extend(father_slice_atomics.clone());
            new_atomics.extend(mother_slice_atomics.clone());

            new_bonds.extend(father_slice_bonds);
            new_bonds.extend(mother_slice_bonds);
        } else if mother_end < &(&parents.0.atomics.len() - 1) {
            let (mother_slice_atomics, mother_slice_bonds) =
                &parents
                    .0
                    .genetic_slice(mother_start.clone(), mother_end.clone(), true);
            let (father_slice_atomics, father_slice_bonds) =
                &parents
                    .1
                    .genetic_slice(father_start.clone(), father_end.clone(), false);
            new_atomics.extend(mother_slice_atomics.clone());
            new_atomics.extend(father_slice_atomics.clone());

            new_bonds.extend(mother_slice_bonds.clone());
            new_bonds.extend(father_slice_bonds.clone());
        } else {
            let mut existing_bonds: Vec<char> = Vec::new();
            existing_bonds.extend(&parents.0.bonds);
            existing_bonds.extend(&parents.1.bonds);
            let connection_bond: char = if existing_bonds.len() <= 1
                || max(parents.0.atomics.len(), parents.1.atomics.len().clone())
                    == existing_bonds.len()
            {
                smarts_pattern::BONDS.choose(&mut rng).unwrap().clone()
            } else {
                existing_bonds.choose(&mut rng).unwrap().clone()
            };
            let connection_bond_probability = rng.gen::<f32>();
            if connection_bond_probability >= 0.5 {
                new_bonds.extend(&parents.1.bonds);
                new_bonds.push(connection_bond);
                new_bonds.extend(&parents.0.bonds);

                new_atomics.append(&mut parents.1.atomics.clone());
                new_atomics.append(&mut parents.0.atomics.clone());
            } else {
                new_bonds.extend(&parents.0.bonds);
                new_bonds.push(connection_bond);
                new_bonds.extend(&parents.1.bonds);

                new_atomics.append(&mut parents.0.atomics.clone());
                new_atomics.append(&mut parents.1.atomics.clone());
            }
        }
        let mutation_result = SMARTSPattern::new(
            new_atomics,
            new_bonds,
            HashMap::from([(
                "createTyp".to_string(),
                format!(
                    "Slice Mutation: mother: {}, father: {} ",
                    &parents.0, &parents.1
                ),
            )]),
        );
        let romol_mutation = &mutation_result.to_ro_mol().unwrap();
        if is_smarts_relevant(
            romol_mutation.to_owned(),
            feature_data,
            evolution_config.bool_matching,
        ) {
            return Ok(mutation_result);
        }
    }
    Err(FingerprintEvolutionError::MutationError)
}
