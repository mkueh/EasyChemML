use crate::BatchFunctions::BatchSort::sort;
use crate::BatchSystem::PythonInterfaces::py_rustBatchTable::{BatchTableF32Py, BatchTableF64Py, BatchTableI16Py, BatchTableI32Py, BatchTableI64Py, BatchTableI8Py};

use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::Python;


#[pyclass]
pub struct BatchSorter_Radix_py  {
    path: String
}

#[pymethods]
impl BatchSorter_Radix_py {

    #[new]
    pub fn new(path: String) -> BatchSorter_Radix_py {
        BatchSorter_Radix_py { path }
    }

    #[pyo3(text_signature = "($self, batchtable, tmp_path)")]
    pub fn sort_i64(&mut self, batchtable: &BatchTableI64Py){
        let mut original_bt = batchtable.batchtable.clone();
        sort::<i64>(original_bt, self.path.as_str())
    }

    #[pyo3(text_signature = "($self, batchtable, tmp_path)")]
    pub fn sort_i32(&mut self, batchtable: &BatchTableI32Py){
        let mut original_bt = batchtable.batchtable.clone();
        sort::<i32>(original_bt, self.path.as_str())
    }

    #[pyo3(text_signature = "($self, batchtable, tmp_path)")]
    pub fn sort_i16(&mut self, batchtable: &BatchTableI16Py){
        let mut original_bt = batchtable.batchtable.clone();
        sort::<i16>(original_bt, self.path.as_str())
    }

    #[pyo3(text_signature = "($self, batchtable, tmp_path)")]
    pub fn sort_i8(&mut self, batchtable: &BatchTableI8Py){
        let mut original_bt = batchtable.batchtable.clone();
        sort::<i8>(original_bt, self.path.as_str())
    }
}