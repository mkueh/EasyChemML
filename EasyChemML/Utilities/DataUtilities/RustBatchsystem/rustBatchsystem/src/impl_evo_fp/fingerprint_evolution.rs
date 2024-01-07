mod fingerprint_inheritance;

use crate::impl_evo_fp::fitness_functions::single_dataset_fitness::FitnessFunctionConfig;
use crate::impl_evo_fp::get_data::convert_data_to_mol;
use crate::impl_evo_fp::population::{generate_new_population, Population};
use ndarray::{Array, Array2, Ix1};
use rdkit::ROMol;

#[derive(Debug, PartialEq, thiserror::Error)]
pub enum FingerprintEvolutionError {
    #[error("could not convert smiles to romol (nullptr)")]
    UnknownConversionError,
}

pub struct EvolutionConfig {
    pub evolution_steps: u8,
    pub population_size: u8,
    pub fp_size: u8,
    pub max_primitive_count: u8,
    pub max_bound_count: u8,
    pub bool_matching: bool,
    // crossover or mutation?
    pub gene_recombination_rate: f32,
    pub new_gene_rate: f32,
    // how much should be newly generated
    pub proportion_new_population: f32,
    // How many should be used for mutation (generation of children)?
    pub proportion_best_parents: f32,
    pub proportion_best_fingerprints: f32,
    pub work_folder: String,
}

pub fn create_the_one_fingerprint(
    evolution_config: EvolutionConfig,
    fitness_function_config: FitnessFunctionConfig,
    feature_data: &Array2<String>,
    regression_targets: &Option<Array<f64, Ix1>>,
    classification_targets: &Option<Array<usize, Ix1>>,
) -> Result<(), std::io::Error> {
    // TODO: move upwards to only receive mol_data in this method
    let mol_data = convert_data_to_mol(feature_data);
    println!("One fingerprint to rule them all!!");
    let (existing_population, found_steps) =
        Population::get_highest_saved_population(&evolution_config.work_folder)?;
    let mut population = if existing_population.is_none() {
        generate_new_population(
            evolution_config.population_size,
            evolution_config.fp_size,
            evolution_config.max_primitive_count,
            evolution_config.max_bound_count,
            evolution_config.bool_matching,
            &mol_data,
        )
    } else {
        existing_population.unwrap()
    };
    if found_steps as u8 >= evolution_config.evolution_steps {
        panic!("Found higher population that indicated evolution steps")
    }
    if found_steps == 0 {
        population.calculate_population_metrics(
            &mol_data,
            regression_targets.clone(),
            classification_targets.clone(),
            evolution_config.bool_matching,
            fitness_function_config.clone(),
        );
        population.save_population(&evolution_config.work_folder, 0)?;
    }

    let evolved_population;
    for i in 0..evolution_config.evolution_steps {
        evolved_population = evolution_step(&evolution_config, &mol_data, &mut population);
    }
    Ok(())
}

fn evolution_step(evolution_config: &EvolutionConfig, feature_data: &Vec<ROMol>, start_population: &mut Population) {
    let number_best =
        evolution_config.proportion_best_fingerprints * evolution_config.population_size;
    let number_new_fps =
        evolution_config.proportion_new_population * evolution_config.population_size;
    let number_kids = evolution_config.population_size - number_best - number_new_fps;

    let new_population_members = generate_new_population(
        number_new_fps,
        evolution_config.fp_size,
        evolution_config.max_primitive_count,
        evolution_config.max_bound_count,
        evolution_config.bool_matching,
        feature_data,
    );
    let best_members = start_population.get_n_best_members(number_best);

    let kids =
}
