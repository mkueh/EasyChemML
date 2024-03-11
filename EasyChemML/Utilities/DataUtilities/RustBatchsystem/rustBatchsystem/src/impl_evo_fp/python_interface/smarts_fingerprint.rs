use crate::impl_evo_fp::smarts_fingerprint::smarts_pattern::SMARTSPattern;
use crate::impl_evo_fp::smarts_fingerprint::SmartsFingerprint;
use pyo3::{pyclass, pymethods};
use std::collections::HashMap;

#[pyclass]
pub struct PySmartsFingerprint {
    #[pyo3(get)]
    id: String,
    #[pyo3(get)]
    patterns: Vec<PySmartsPattern>,
    #[pyo3(get)]
    metrics: HashMap<String, f64>,
}

#[pymethods]
impl PySmartsFingerprint {
    #[new]
    pub fn new(id: String, patterns: Vec<PySmartsPattern>, metrics: HashMap<String, f64>) -> PySmartsFingerprint {
        PySmartsFingerprint { id, patterns, metrics }
    }
}

impl PySmartsFingerprint {
    pub fn into_smarts_fingerprint(self) -> SmartsFingerprint {
        let patterns = self
            .patterns
            .into_iter()
            .map(|x| SMARTSPattern::new(x.atomics, x.bonds, x.create_info))
            .collect();
        SmartsFingerprint::new(patterns)
    }
}

#[pyclass]
pub struct PySmartsPattern {
    #[pyo3(get)]
    atomics: Vec<String>,
    #[pyo3(get)]
    bonds: Vec<char>,
    #[pyo3(get)]
    create_info: HashMap<String, String>,
}

#[pymethods]
impl PySmartsPattern {
    #[new]
    pub fn new(
        atomics: Vec<String>,
        bonds: Vec<char>,
        create_info: HashMap<String, String>,
    ) -> PySmartsPattern {
        PySmartsPattern {
            atomics,
            bonds,
            create_info,
        }
    }
}
impl PySmartsPattern {
    pub fn from_smarts_pattern(pattern: SMARTSPattern) -> PySmartsPattern {
        PySmartsPattern {
            atomics: pattern.atomics,
            bonds: pattern.bonds,
            create_info: pattern.create_info,
        }
    }
}
