use crate::impl_evo_fp::fitness_functions::single_dataset_fitness::FitnessFunctionConfig;
use crate::impl_evo_fp::get_data::convert_data_to_mol;
use crate::impl_evo_fp::population::generate_new_population;
use ndarray::{Array, Array2, Ix1};

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
) {
    // TODO: move upwards to only receive mol_data in this method
    let mol_data = convert_data_to_mol(feature_data);
    println!("One fingerprint to rule them all!!");
    let mut first_population = generate_new_population(
        evolution_config.population_size,
        evolution_config.fp_size,
        evolution_config.max_primitive_count,
        evolution_config.max_bound_count,
        evolution_config.bool_matching,
        &mol_data,
    );
    first_population.calculate_population_metrics(
        &mol_data,
        regression_targets.clone(),
        classification_targets.clone(),
        evolution_config.bool_matching,
        fitness_function_config.clone(),
    );
    let best_members = first_population.get_n_best_members(
        (evolution_config.population_size as f32 * evolution_config.proportion_best_fingerprints)
            as usize,
    );
}
