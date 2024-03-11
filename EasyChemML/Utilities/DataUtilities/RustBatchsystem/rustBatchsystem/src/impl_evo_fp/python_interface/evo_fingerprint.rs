use crate::impl_evo_fp::python_interface::configuration::PyEvolutionConfig;
use crate::impl_evo_fp::python_interface::configuration::PyFitnessFunctionConfig;
use crate::impl_evo_fp::python_interface::data_processing::{
    convert_smiles_batch_data, convert_target_batch_data,
};
use crate::impl_evo_fp::python_interface::smarts_fingerprint::{
    PySmartsFingerprint, PySmartsPattern,
};
use crate::impl_evo_fp::{fingerprint_evolution, get_data};
use crate::BatchSystem::BatchTablesImplementation::BatchTable::BatchTable;
use crate::BatchSystem::PythonInterfaces::py_rustBatchTable::{
    BatchTableF64Py, BatchTableStringPy,
};
use ndarray::arr1;
use pyo3::prelude::*;
use std::ops::Not;

/// This function is used to train and return an EVO-fingerprint.
/// The SMILES batch table is converted into molecular data.
/// Depending on the value of the boolean for regression, the targets are set for regression or classification.
/// `is_regression` is necessary for the internal processing of the function.
///
/// # Arguments
///
/// * `py_evolution_config` - A PyEvolutionConfig object that contains the evolution configuration.
/// * `py_fitness_function_config` - A PyFitnessFunctionConfig object that contains the fitness function configuration.
/// * `smiles_batch_table` - A reference to a BatchTableStringPy object that contains the SMILES batch data.
/// * `target_batch_table_f64` - A reference to a BatchTableF64Py object that contains the target batch data.
/// * `is_regression` - A boolean value that indicates whether it's a regression task.
///
/// # Returns
///
/// * `PyResult<PySmartsFingerprint>` - A Python object that represents the trained fingerprint.

#[pyfunction]
fn train_the_one_fingerprint(
    py_evolution_config: PyEvolutionConfig,
    py_fitness_function_config: PyFitnessFunctionConfig,
    smiles_batch_table: &BatchTableStringPy,
    target_batch_table_f64: &BatchTableF64Py,
    is_regression: bool,
) -> PyResult<PySmartsFingerprint> {
    let evolution_config = py_evolution_config.evolution_config();
    let fitness_function_config = py_fitness_function_config.fitness_function_config();

    let mol_array = convert_smiles_batch_data(smiles_batch_table);
    let target_chunks = convert_target_batch_data(target_batch_table_f64);

    let regression_targets = if is_regression {
        Some(target_chunks.clone())
    } else {
        None
    };
    let classification_targets = if is_regression.not() {
        Some(arr1(
            &target_chunks.into_iter().map(|x| x as usize).collect(),
        ))
    } else {
        None
    };

    let member = fingerprint_evolution::create_the_one_fingerprint(
        evolution_config,
        fitness_function_config,
        mol_array,
        &regression_targets,
        &classification_targets,
    )
    .unwrap();
    let mut fingerprint = member.fingerprint;
    fingerprint
        .save_fingerprint(&evolution_config.work_folder)
        .expect("Saving unsuccessful");
    let py_patterns: Vec<PySmartsPattern> = fingerprint
        .patterns
        .iter()
        .map(|x| PySmartsPattern::from_smarts_pattern(x.clone()))
        .collect();
    let py_fingerprint = PySmartsFingerprint::new(fingerprint.id.to_string(), py_patterns);
    Ok(py_fingerprint)
}

/// This function is used to convert the given SMILES batch table into molecular fingerprints.
///
/// # Arguments
///
/// * `smiles_batch_table` - A reference to a BatchTableStringPy object that contains the SMILES batch data.
/// * `py_fingerprint` - A PySmartsFingerprint object that represents the fingerprint.
/// * `bool_matching` - A boolean value that indicates whether boolean or numerical matching should be performed.
///
/// # Returns
///
/// * `PyResult<PyObject>` - A Python object that represents the generated feature data.
#[pyfunction]
fn convert(
    smiles_batch_table: &BatchTableStringPy,
    py_fingerprint: PySmartsFingerprint,
    bool_matching: bool,
) -> PyResult<PyObject> {
    let mol_data = convert_smiles_batch_data(smiles_batch_table);

    let fingerprint = py_fingerprint.into_smarts_fingerprint();

    let feature_data =
        get_data::generate_classification_feature_data(&fingerprint, &mol_data, bool_matching);
    feature_data.to_object(Python(Default::default()))
}

pub fn register_evo_submodule(py: Python) -> &PyModule {
    let m = PyModule::new(py, "evo_fingerprint").unwrap();
    m.add_function(wrap_pyfunction!(train_the_one_fingerprint, m)?)?;
    m.add_function(wrap_pyfunction!(convert, m)?)?;
    m.add_class::<PyEvolutionConfig>()?;
    m.add_class::<PyFitnessFunctionConfig>()?;
    m.add_class::<PySmartsFingerprint>()?;
    m.add_class::<PySmartsPattern>()?;
    m
}
