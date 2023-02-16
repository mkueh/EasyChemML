use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::Python;

use crate::BatchFunctions::BatchListFunctions_duplicates;
use crate::BatchFunctions::BatchListFunctions_duplicates::duplicat_result;
use crate::BatchSystem::PythonInterfaces::py_rustBatchTable::{BatchTableI8Py, BatchTableI16Py, BatchTableI32Py, BatchTableI64Py};

#[pyclass]
pub struct BatchListFunctions_duplicates_py {}

#[pyclass]
pub struct Duplicat_result_i8 {
    #[pyo3(get)]
    pub counted_entries:usize,
    #[pyo3(get)]
    pub counted_duplicates:usize,
    #[pyo3(get)]
    pub entry_most_duplicates:usize,
    #[pyo3(get)]
    pub duplicates_dist: HashMap<usize, usize>,
    #[pyo3(get)]
    pub duplicates_by_last_col: HashMap<i8, usize>
}

#[pyclass]
pub struct Duplicat_result_i16 {
    #[pyo3(get)]
    pub counted_entries:usize,
    #[pyo3(get)]
    pub counted_duplicates:usize,
    #[pyo3(get)]
    pub entry_most_duplicates:usize,
    #[pyo3(get)]
    pub duplicates_dist: HashMap<usize, usize>,
    #[pyo3(get)]
    pub duplicates_by_last_col: HashMap<i16, usize>
}

#[pyclass]
pub struct Duplicat_result_i32 {
    #[pyo3(get)]
    pub counted_entries:usize,
    #[pyo3(get)]
    pub counted_duplicates:usize,
    #[pyo3(get)]
    pub entry_most_duplicates:usize,
    #[pyo3(get)]
    pub duplicates_dist: HashMap<usize, usize>,
    #[pyo3(get)]
    pub duplicates_by_last_col: HashMap<i32, usize>
}

#[pyclass]
pub struct Duplicat_result_i64 {
    #[pyo3(get)]
    pub counted_entries:usize,
    #[pyo3(get)]
    pub counted_duplicates:usize,
    #[pyo3(get)]
    pub entry_most_duplicates:usize,
    #[pyo3(get)]
    pub duplicates_dist: HashMap<usize, usize>,
    #[pyo3(get)]
    pub duplicates_by_last_col: HashMap<i64, usize>
}



#[pymethods]
impl BatchListFunctions_duplicates_py {

    #[new]
    pub fn new() -> BatchListFunctions_duplicates_py {
        BatchListFunctions_duplicates_py {  }
    }

    #[pyo3(text_signature = "($self, batchtable, calc_duplicates_by_last_col)")]
    pub fn count_duplicates_on_sorted_list_i8(&mut self, batchtable_i8: &BatchTableI8Py, calc_duplicates_by_last_col:bool) -> Duplicat_result_i8{
        let mut ground_batchtable = batchtable_i8.batchtable.clone();
        let duplicates_result: duplicat_result<i8> = BatchListFunctions_duplicates::count_duplicates_on_sorted_list(ground_batchtable, calc_duplicates_by_last_col);

        Duplicat_result_i8{
            counted_entries: duplicates_result.counted_entries,
            counted_duplicates: duplicates_result.counted_duplicates,
            entry_most_duplicates: duplicates_result.entry_most_duplicates,
            duplicates_dist: duplicates_result.duplicates_dist,
            duplicates_by_last_col: duplicates_result.duplicates_by_last_col
        }
    }

    #[pyo3(text_signature = "($self, batchtable, calc_duplicates_by_last_col)")]
    pub fn count_duplicates_on_sorted_list_i16(&mut self, batchtable_i16: &BatchTableI16Py, calc_duplicates_by_last_col:bool) -> Duplicat_result_i16{
        let mut ground_batchtable = batchtable_i16.batchtable.clone();
        let duplicates_result: duplicat_result<i16> = BatchListFunctions_duplicates::count_duplicates_on_sorted_list(ground_batchtable, calc_duplicates_by_last_col);

        Duplicat_result_i16{
            counted_entries: duplicates_result.counted_entries,
            counted_duplicates: duplicates_result.counted_duplicates,
            entry_most_duplicates: duplicates_result.entry_most_duplicates,
            duplicates_dist: duplicates_result.duplicates_dist,
            duplicates_by_last_col: duplicates_result.duplicates_by_last_col
        }
    }

    #[pyo3(text_signature = "($self, batchtable, calc_duplicates_by_last_col)")]
    pub fn count_duplicates_on_sorted_list_i32(&mut self, batchtable_i32: &BatchTableI32Py, calc_duplicates_by_last_col:bool) -> Duplicat_result_i32{
        let mut ground_batchtable = batchtable_i32.batchtable.clone();
        let duplicates_result: duplicat_result<i32> = BatchListFunctions_duplicates::count_duplicates_on_sorted_list(ground_batchtable, calc_duplicates_by_last_col);

        Duplicat_result_i32{
            counted_entries: duplicates_result.counted_entries,
            counted_duplicates: duplicates_result.counted_duplicates,
            entry_most_duplicates: duplicates_result.entry_most_duplicates,
            duplicates_dist: duplicates_result.duplicates_dist,
            duplicates_by_last_col: duplicates_result.duplicates_by_last_col
        }
    }

    #[pyo3(text_signature = "($self, batchtable, calc_duplicates_by_last_col)")]
    pub fn count_duplicates_on_sorted_list_i64(&mut self, batchtable_i64: &BatchTableI64Py, calc_duplicates_by_last_col:bool) -> Duplicat_result_i64{
        let mut ground_batchtable = batchtable_i64.batchtable.clone();
        let duplicates_result: duplicat_result<i64> = BatchListFunctions_duplicates::count_duplicates_on_sorted_list(ground_batchtable, calc_duplicates_by_last_col);

        Duplicat_result_i64{
            counted_entries: duplicates_result.counted_entries,
            counted_duplicates: duplicates_result.counted_duplicates,
            entry_most_duplicates: duplicates_result.entry_most_duplicates,
            duplicates_dist: duplicates_result.duplicates_dist,
            duplicates_by_last_col: duplicates_result.duplicates_by_last_col
        }
    }
}

