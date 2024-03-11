use crate::impl_evo_fp::fingerprint_evolution::EvolutionConfig;
use crate::impl_evo_fp::fitness_functions::fitness_metrics::Metric;
use crate::impl_evo_fp::fitness_functions::fitness_models::Model;
use crate::impl_evo_fp::fitness_functions::single_dataset_fitness::FitnessFunctionConfig;
use pyo3::{pyclass, pymethods, PyObject};
use std::collections::HashMap;

#[pyclass]
pub struct PyEvolutionConfig {
    #[pyo3(get)]
    evolution_steps: u8,
    #[pyo3(get)]
    population_size: usize,
    #[pyo3(get)]
    fp_size: usize,
    #[pyo3(get)]
    max_primitive_count: u8,
    #[pyo3(get)]
    max_bound_count: u8,
    #[pyo3(get)]
    bool_matching: bool,
    #[pyo3(get)]
    gene_recombination_rate: f32,
    #[pyo3(get)]
    new_gene_rate: f32,
    #[pyo3(get)]
    proportion_new_population: f32,
    #[pyo3(get)]
    proportion_best_parents: f32,
    #[pyo3(get)]
    proportion_best_fingerprints: f32,
    #[pyo3(get)]
    work_folder: String,
    #[pyo3(get)]
    gene_recombination_attempts: i8,
}

#[pymethods]
impl PyEvolutionConfig {
    #[new]
    pub fn new(
        evolution_steps: u8,
        population_size: usize,
        fp_size: usize,
        max_primitive_count: u8,
        max_bound_count: u8,
        bool_matching: bool,
        gene_recombination_rate: f32,
        new_gene_rate: f32,
        proportion_new_population: f32,
        proportion_best_parents: f32,
        proportion_best_fingerprints: f32,
        work_folder: String,
        gene_recombination_attempts: i8,
    ) -> Self {
        PyEvolutionConfig {
            evolution_steps,
            population_size,
            fp_size,
            max_primitive_count,
            max_bound_count,
            bool_matching,
            gene_recombination_rate,
            new_gene_rate,
            proportion_new_population,
            proportion_best_parents,
            proportion_best_fingerprints,
            work_folder,
            gene_recombination_attempts,
        }
    }
}
impl PyEvolutionConfig {
    pub fn evolution_config(self) -> EvolutionConfig {
        EvolutionConfig {
            evolution_steps: self.evolution_steps,
            population_size: self.population_size,
            fp_size: self.fp_size,
            max_primitive_count: self.max_primitive_count,
            max_bound_count: self.max_bound_count,
            bool_matching: self.bool_matching,
            gene_recombination_rate: self.gene_recombination_rate,
            new_gene_rate: self.new_gene_rate,
            proportion_new_population: self.proportion_new_population,
            proportion_best_parents: self.proportion_best_parents,
            proportion_best_fingerprints: self.proportion_best_fingerprints,
            work_folder: self.work_folder,
            gene_recombination_attempts: self.gene_recombination_attempts,
        }
    }
}

#[pyclass]
pub struct PyFitnessFunctionConfig {
    #[pyo3(get)]
    model: String,
    #[pyo3(get)]
    fitness_metric: String,
    #[pyo3(get)]
    model_params: HashMap<String, PyObject>,
    #[pyo3(get)]
    k_folds: usize,
    #[pyo3(get)]
    split_ratio: f32,
}

#[pymethods]
impl crate::impl_evo_fp::python_interface::evo_fingerprint::PyFitnessFunctionConfig {
    #[new]
    pub fn new(
        model: String,
        fitness_metric: String,
        model_params: HashMap<String, PyObject>,
        k_folds: usize,
        split_ratio: f32,
    ) -> Self {
        crate::impl_evo_fp::python_interface::evo_fingerprint::PyFitnessFunctionConfig {
            model,
            fitness_metric,
            model_params,
            k_folds,
            split_ratio,
        }
    }
}

impl PyFitnessFunctionConfig {
    pub fn fitness_function_config(&self) -> FitnessFunctionConfig {
        FitnessFunctionConfig {
            model: Model::match_model(&self.model),
            fitness_metric: Metric::match_metric(&self.fitness_metric),
            model_params: self.model_params.clone(),
            k_folds: self.k_folds,
            split_ratio: self.split_ratio,
        }
    }
}
