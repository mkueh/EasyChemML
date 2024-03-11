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
    feature_data: &[Vec<ROMol>],
) -> Vec<Member> {
    println!("Start mutation!");
    let mut rng = rand::thread_rng();
    let mut kids = Vec::new();

    for _ in 0..number_kids {
        let eltern = &parents
            .choose_multiple(&mut rng, 2)
            .cloned()
            .collect_tuple()
            .unwrap();
        kids.push(mutate(eltern, evolution_config, feature_data))
    }
    println!("Mutation complete!");
    kids
}

fn mutate(
    parents: &(Member, Member),
    evolution_config: &EvolutionConfig,
    feature_data: &[Vec<ROMol>],
) -> Member {
    let mut new_genes = Vec::new();
    let mut rng = rand::thread_rng();
    let mut mother = parents.0.fingerprint.clone();
    mother.patterns.shuffle(&mut rng);
    let mut father = parents.1.fingerprint.clone();
    father.patterns.shuffle(&mut rng);
    let mut gene_index = 0;

    loop {
        let mutation_probability = rng.gen::<f32>();
        if mutation_probability <= evolution_config.gene_recombination_rate {
            let mut mutter = mother.patterns[gene_index].clone();
            let mut vater = father.patterns[gene_index].clone();
            // Father and mother genes are mutated
            let sliced_pattern =
                slice_mutation(evolution_config, (&mut vater, &mut mutter), feature_data);
            if let Ok(pattern) = sliced_pattern {
                new_genes.push(pattern);
                gene_index += 1;
            } else {
                continue;
            }
        } else if mutation_probability
            >= (evolution_config.gene_recombination_rate + evolution_config.new_gene_rate)
        {
            // Father or mother genes are chosen
            new_genes.push(choice_mutation((
                &father.patterns[gene_index],
                &mother.patterns[gene_index],
            )));
            gene_index += 1;
        } else {
            new_genes.push(SMARTSPattern::generate_smarts_pattern(
                evolution_config.max_primitive_count,
                evolution_config.max_bound_count,
                evolution_config.bool_matching,
                feature_data,
            ));
            gene_index += 1;
        };
        if new_genes.len() == evolution_config.fp_size {
            return Member::new(SmartsFingerprint::new(new_genes));
        }
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
    feature_data: &[Vec<ROMol>],
) -> Result<SMARTSPattern, FingerprintEvolutionError> {
    for _ in 0..evolution_config.gene_recombination_attempts {
        let mut new_atomics = Vec::new();
        let mut new_bonds = Vec::new();
        let mut rng = rand::thread_rng();
        let end_bond_probability = rng.gen::<f32>();
        // gene indices
        let (mother_start, mother_end) = &parents.0.random_gene_intervall();
        let (father_start, father_end) = &parents.1.random_gene_intervall();

        // Order of appending matters for the genes!
        // if we don't include all the father's elements
        if end_bond_probability >= 0.5 && father_end < &(&parents.1.atomics.len() - 1) {
            let (mother_slice_atomics, mother_slice_bonds) =
                &parents.0.genetic_slice(*mother_start, *mother_end, false);
            let (father_slice_atomics, father_slice_bonds) =
                &parents.1.genetic_slice(*father_start, *father_end, true);
            new_atomics.extend(father_slice_atomics.clone());
            new_atomics.extend(mother_slice_atomics.clone());

            new_bonds.extend(father_slice_bonds);
            new_bonds.extend(mother_slice_bonds);
            // if we don't include all the mother's elements
        } else if mother_end < &(&parents.0.atomics.len() - 1) {
            let (mother_slice_atomics, mother_slice_bonds) =
                &parents.0.genetic_slice(*mother_start, *mother_end, true);
            let (father_slice_atomics, father_slice_bonds) =
                &parents.1.genetic_slice(*father_start, *father_end, false);
            new_atomics.extend(mother_slice_atomics.clone());
            new_atomics.extend(father_slice_atomics.clone());

            new_bonds.extend(mother_slice_bonds.clone());
            new_bonds.extend(father_slice_bonds.clone());
        } else {
            // if we include all the father's and all the mother's elements, we need an additional bond
            let mut existing_bonds: Vec<char> = Vec::new();
            existing_bonds.extend(&parents.0.bonds);
            existing_bonds.extend(&parents.1.bonds);
            let connection_bond: char = if existing_bonds.len() <= 1
                || max(parents.0.atomics.len(), parents.1.atomics.len()) == existing_bonds.len()
            {
                *smarts_pattern::BONDS.choose(&mut rng).unwrap()
            } else {
                *existing_bonds.choose(&mut rng).unwrap()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::impl_evo_fp::fingerprint_evolution::EvolutionConfig;
    use crate::impl_evo_fp::get_data::convert_data_to_mol;
    use crate::impl_evo_fp::population::Population;
    use ndarray::arr2;

    #[test]
    fn test_evolve_population() {
        let feature_data = convert_data_to_mol(&arr2(&[
            ["Cn1c(CN2CCN(c3ccc(Cl)cc3)CC2)nc2ccccc21".to_string()],
            ["COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1".to_string()],
            ["O=C(NC1Cc2ccccc2N(C[C@@H](O)CO)C1=O)c1cc2cc(Cl)sc2[nH]1".to_string()],
            ["Cc1cccc(C[C@H](NC(=O)c2cc(C(C)(C)C)nn2C)C(=O)NCC#N)c1".to_string()],
            ["OC1(C#Cc2ccc(-c3ccccc3)cc2)CN2CCC1CC2".to_string()],
            ["COc1cc(OC)c(S(=O)(=O)NCc2ccccc2N2CCCCC2)cc1NC(=O)CCC(=O)O".to_string()],
            ["Oc1cc(OC)c(S(=O)(=O)NCc2ccccc2N2CCCCC2)cc1NC(=O)CCC(=O)O".to_string()],
            ["CNc1cccc(CCOc2ccc(C[C@H](NC(=O)c3c(Cl)cccc3Cl)C(=O)O)cc2C)n1".to_string()],
            ["COc1ccc(-c2coc3cc(OC)cc(OC)c3c2=O)cc1".to_string()],
            ["Oc1ncnc2scc(-c3ccsc3)c12".to_string()],
            ["CS(=O)(=O)c1ccc(Oc2ccc(C#C[C@]3(O)CN4CCC3CC4)cc2)cc1".to_string()],
            ["Cc1cc(Nc2nc(N[C@@H](C)c3ccc(F)cn3)c(C#N)nc2C)n[nH]1".to_string()],
            ["O=C1CCCCCN1".to_string()],
            ["CCCSc1ncccc1C(=O)N1CCCC1c1ccncc1".to_string()],
            ["Cc1ccc(S(=O)(=O)Nc2c(C(=O)NC3CCCCC3C)cnn2-c2ccccc2)cc1".to_string()],
            ["Nc1ccc(-c2nc3ccc(O)cc3s2)cc1".to_string()],
            ["COc1ccc(N2CCN(C(=O)[C@@H]3CCCC[C@H]3C(=O)NCC#N)CC2)cc1".to_string()],
            ["CCC(COC(=O)c1cc(OC)c(OC)c(OC)c1)(c1ccccc1)N(C)C".to_string()],
            ["COc1cc(-n2nnc3cc(-c4ccc(Cl)cc4)sc3c2=O)ccc1N1CC[C@@H](O)C1".to_string()],
            ["CO[C@H]1CN(CCn2c(=O)ccc3ccc(C#N)cc32)CC[C@H]1NCc1ccc2c(n1)NC(=O)CO2".to_string()],
            ["CC(C)(CCCCCOCCc1ccccc1)NCCc1ccc(O)c2nc(O)sc12".to_string()],
            ["O=C(Nc1nnc(C(=O)Nc2ccc(N3CCOCC3)cc2)o1)c1ccc(Cl)cc1".to_string()],
        ]));

        // Create EvolutionConfig
        let evolution_config = EvolutionConfig {
            evolution_steps: 3,
            population_size: 10,
            fp_size: 10,
            max_primitive_count: 5,
            max_bound_count: 4,
            bool_matching: true,
            gene_recombination_rate: 0.15,
            new_gene_rate: 0.1,
            proportion_new_population: 0.3,
            proportion_best_parents: 0.2,
            proportion_best_fingerprints: 0.5,
            work_folder: "./".to_string(),
            gene_recombination_attempts: 10,
        };

        // Create parent Members
        let parents = Population::generate_new_population(
            evolution_config.population_size,
            evolution_config.fp_size,
            evolution_config.max_primitive_count,
            evolution_config.max_bound_count,
            evolution_config.bool_matching,
            &feature_data,
        );

        // Call evolve_population function
        let new_population =
            evolve_population(&evolution_config, 5, &parents.members, &feature_data);

        // Assert that the new population size is equal to number_kids
        assert_eq!(new_population.len(), 5);
        for member in new_population {
            // Assert that the new population's fingerprint size is equal to fp_size
            assert_eq!(member.fingerprint.patterns.len(), evolution_config.fp_size);
        }
    }

    #[test]
    fn test_slice_mutation() {
        let feature_data = convert_data_to_mol(&arr2(&[
            ["Cn1c(CN2CCN(c3ccc(Cl)cc3)CC2)nc2ccccc21".to_string()],
            ["COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1".to_string()],
            ["O=C(NC1Cc2ccccc2N(C[C@@H](O)CO)C1=O)c1cc2cc(Cl)sc2[nH]1".to_string()],
            ["Cc1cccc(C[C@H](NC(=O)c2cc(C(C)(C)C)nn2C)C(=O)NCC#N)c1".to_string()],
            ["OC1(C#Cc2ccc(-c3ccccc3)cc2)CN2CCC1CC2".to_string()],
            ["COc1cc(OC)c(S(=O)(=O)NCc2ccccc2N2CCCCC2)cc1NC(=O)CCC(=O)O".to_string()],
            ["Oc1cc(OC)c(S(=O)(=O)NCc2ccccc2N2CCCCC2)cc1NC(=O)CCC(=O)O".to_string()],
            ["CNc1cccc(CCOc2ccc(C[C@H](NC(=O)c3c(Cl)cccc3Cl)C(=O)O)cc2C)n1".to_string()],
            ["COc1ccc(-c2coc3cc(OC)cc(OC)c3c2=O)cc1".to_string()],
            ["Oc1ncnc2scc(-c3ccsc3)c12".to_string()],
            ["CS(=O)(=O)c1ccc(Oc2ccc(C#C[C@]3(O)CN4CCC3CC4)cc2)cc1".to_string()],
            ["Cc1cc(Nc2nc(N[C@@H](C)c3ccc(F)cn3)c(C#N)nc2C)n[nH]1".to_string()],
            ["O=C1CCCCCN1".to_string()],
            ["CCCSc1ncccc1C(=O)N1CCCC1c1ccncc1".to_string()],
            ["Cc1ccc(S(=O)(=O)Nc2c(C(=O)NC3CCCCC3C)cnn2-c2ccccc2)cc1".to_string()],
            ["Nc1ccc(-c2nc3ccc(O)cc3s2)cc1".to_string()],
            ["COc1ccc(N2CCN(C(=O)[C@@H]3CCCC[C@H]3C(=O)NCC#N)CC2)cc1".to_string()],
            ["CCC(COC(=O)c1cc(OC)c(OC)c(OC)c1)(c1ccccc1)N(C)C".to_string()],
            ["COc1cc(-n2nnc3cc(-c4ccc(Cl)cc4)sc3c2=O)ccc1N1CC[C@@H](O)C1".to_string()],
            ["CO[C@H]1CN(CCn2c(=O)ccc3ccc(C#N)cc32)CC[C@H]1NCc1ccc2c(n1)NC(=O)CO2".to_string()],
            ["CC(C)(CCCCCOCCc1ccccc1)NCCc1ccc(O)c2nc(O)sc12".to_string()],
            ["O=C(Nc1nnc(C(=O)Nc2ccc(N3CCOCC3)cc2)o1)c1ccc(Cl)cc1".to_string()],
        ]));

        // Create EvolutionConfig
        let evolution_config = EvolutionConfig {
            evolution_steps: 3,
            population_size: 10,
            fp_size: 10,
            max_primitive_count: 5,
            max_bound_count: 4,
            bool_matching: true,
            gene_recombination_rate: 0.3,
            new_gene_rate: 0.1,
            proportion_new_population: 0.3,
            proportion_best_parents: 0.2,
            proportion_best_fingerprints: 0.5,
            work_folder: "./".to_string(),
            gene_recombination_attempts: 10,
        };

        let mut rng = rand::thread_rng();
        // Create parent Members
        let parents = Population::generate_new_population(
            2,
            evolution_config.fp_size,
            evolution_config.max_primitive_count,
            evolution_config.max_bound_count,
            evolution_config.bool_matching,
            &feature_data,
        );
        let mut father = parents.members[0].fingerprint.clone();
        father.patterns.shuffle(&mut rng);
        let mut mother = parents.members[1].fingerprint.clone();
        mother.patterns.shuffle(&mut rng);
        // Call slice_mutation function
        let result = slice_mutation(
            &evolution_config,
            (&mut mother.patterns[0], &mut father.patterns[0]),
            &feature_data,
        );

        // Assert that the result is a SMARTSPattern
        assert!(matches!(result, Ok(_)));
    }
}
