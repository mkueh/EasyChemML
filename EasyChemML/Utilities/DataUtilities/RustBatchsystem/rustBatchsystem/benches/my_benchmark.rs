use criterion::{criterion_group, criterion_main, Criterion, SamplingMode};
use pyo3::prelude::*;
use rusty_evo_fp::impl_evo_fp::chem_datasets::csv_import;
use rusty_evo_fp::impl_evo_fp::fingerprint_evolution::{
    create_the_one_fingerprint, EvolutionConfig,
};
use rusty_evo_fp::impl_evo_fp::fitness_functions::fitness_metrics::ClassificationMetric::F1Score;
use rusty_evo_fp::impl_evo_fp::fitness_functions::fitness_metrics::Metric::ClassificationMetric;
use rusty_evo_fp::impl_evo_fp::fitness_functions::fitness_models::Model::CatBoostClassifier;
use rusty_evo_fp::impl_evo_fp::fitness_functions::single_dataset_fitness::FitnessFunctionConfig;
use std::collections::HashMap;
use std::time::Instant;
use tempfile::tempdir;

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("EVO-FP-Performance");
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);

    let (mol_data, target_data) = csv_import::hiv_8000();

    Python::with_gil(|py| {
        let fitness_function_config = FitnessFunctionConfig {
            model: CatBoostClassifier,
            fitness_metric: ClassificationMetric(F1Score),
            model_params: HashMap::from([
                ("verbose".to_string(), 0.into_py(py)),
                ("allow_writing_files".to_string(), false.into_py(py)),
            ]),
            k_folds: 5,
            split_ratio: 0.8,
        };
        let mut i = 0;

        group.bench_function("FP-EVO-Reg", |b| {
            b.iter(|| {
                println!("round: {}", i);
                i += 1;
                let temp_dir = tempdir().unwrap();
                let path = temp_dir.path().to_str().unwrap();
                // Create instances of the required types
                let evolution_config = EvolutionConfig {
                    evolution_steps: 10,
                    population_size: 10,
                    fp_size: 64,
                    max_primitive_count: 4,
                    max_bound_count: 4,
                    bool_matching: false,
                    gene_recombination_rate: 0.06,
                    new_gene_rate: 0.04,
                    proportion_new_population: 0.2,
                    proportion_best_parents: 0.4,
                    proportion_best_fingerprints: 0.2,
                    work_folder: path.to_string(),
                    gene_recombination_attempts: 10,
                };
                let cloned_mol = mol_data.clone();
                let cloned_target_data = target_data.clone();

                let start = Instant::now();

                let _result = create_the_one_fingerprint(
                    &evolution_config,
                    &fitness_function_config,
                    cloned_mol,
                    &None,
                    &Some(cloned_target_data),
                );
                let duration = start.elapsed();

                println!("______________________________________________________________");
                println!("Time for this round: {:?}", &duration);
                println!("______________________________________________________________");
            })
        })
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
