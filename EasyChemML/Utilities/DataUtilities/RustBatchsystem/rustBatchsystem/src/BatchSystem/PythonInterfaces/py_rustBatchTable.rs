use ndarray::{Array, Array2, Ix2};
use numpy::{PyArray, PyReadonlyArray2, ToPyArray};
use std::sync::{Arc, RwLock};

use pyo3::prelude::*;
use pyo3::Python;

use crate::BatchSystem::BatchTablesImplementation::BatchTable::{
    BatchTable, BatchTablesTypWrapper,
};
use crate::Utilities::array_helper;

#[pyclass()]
pub struct BatchTableI64Py {
    pub batchtable: Arc<RwLock<BatchTablesTypWrapper>>,

    #[pyo3(get, set)]
    pub shape: Vec<usize>,
}

#[pymethods]
impl BatchTableI64Py {
    // wir kriegen ein Chunk von Python und adden ihn in Rust
    fn add_chunk(&mut self, arr: PyReadonlyArray2<i64>) {
        let arr_view = arr.as_array();
        let mut batchtable = self.batchtable.write().unwrap();
        batchtable.add_chunk(&arr_view);
    }
    // wir kriegen ein Chunk von Python und overwriten ihn in Rust
    fn override_chunk(&mut self, arr: PyReadonlyArray2<i64>, index: usize) {
        let arr_view = arr.as_array();
        let mut batchtable = self.batchtable.write().unwrap();
        batchtable.override_chunk_with_array(&arr_view, index);
    }

    fn print_arr(&mut self) {
        let batchtable = self.batchtable.write().unwrap();
        for i in 0..batchtable.get_table_chunk_count() {
            let loaded_chunk: Array<i64, Ix2> = batchtable.load_chunk(i);
            array_helper::print_array_2D(&loaded_chunk);
        }
    }
    // von Rust zu Python
    fn load_chunk<'py>(&mut self, py: Python<'py>, index: usize) -> &'py PyArray<i64, Ix2> {
        let batchtable = self.batchtable.write().unwrap();
        let loaded_chunk: Array<i64, Ix2> = batchtable.load_chunk(index);
        let output_result = loaded_chunk.to_pyarray(py);
        return output_result;
    }
}

#[pyclass()]
pub struct BatchTableI32Py {
    pub batchtable: Arc<RwLock<BatchTablesTypWrapper>>,

    #[pyo3(get)]
    pub shape: Vec<usize>,
}

#[pymethods]
impl BatchTableI32Py {
    fn add_chunk(&mut self, arr: PyReadonlyArray2<i32>) {
        let arr_view = arr.as_array();
        let mut batchtable = self.batchtable.write().unwrap();
        batchtable.add_chunk(&arr_view);
    }

    fn override_chunk(&mut self, arr: PyReadonlyArray2<i32>, index: usize) {
        let arr_view = arr.as_array();
        let mut batchtable = self.batchtable.write().unwrap();
        batchtable.override_chunk_with_array(&arr_view, index);
    }

    fn print_arr(&mut self) {
        let batchtable = self.batchtable.write().unwrap();
        for i in 0..batchtable.get_table_chunk_count() {
            let loaded_chunk: Array<i32, Ix2> = batchtable.load_chunk(i);
            array_helper::print_array_2D(&loaded_chunk);
        }
    }

    fn load_chunk<'py>(&mut self, py: Python<'py>, index: usize) -> &'py PyArray<i32, Ix2> {
        let batchtable = self.batchtable.write().unwrap();
        let loaded_chunk: Array<i32, Ix2> = batchtable.load_chunk(index);
        let output_result = loaded_chunk.to_pyarray(py);
        return output_result;
    }
}

#[pyclass()]
pub struct BatchTableI16Py {
    pub batchtable: Arc<RwLock<BatchTablesTypWrapper>>,

    #[pyo3(get)]
    pub shape: Vec<usize>,
}

#[pymethods]
impl BatchTableI16Py {
    fn add_chunk(&mut self, arr: PyReadonlyArray2<i16>) {
        let arr_view = arr.as_array();
        let mut batchtable = self.batchtable.write().unwrap();
        batchtable.add_chunk(&arr_view);
    }

    fn override_chunk(&mut self, arr: PyReadonlyArray2<i16>, index: usize) {
        let arr_view = arr.as_array();
        let mut batchtable = self.batchtable.write().unwrap();
        batchtable.override_chunk_with_array(&arr_view, index);
    }

    fn print_arr(&mut self) {
        let batchtable = self.batchtable.write().unwrap();
        for i in 0..batchtable.get_table_chunk_count() {
            let loaded_chunk: Array<i16, Ix2> = batchtable.load_chunk(i);
            array_helper::print_array_2D(&loaded_chunk);
        }
    }

    fn load_chunk<'py>(&mut self, py: Python<'py>, index: usize) -> &'py PyArray<i16, Ix2> {
        let batchtable = self.batchtable.write().unwrap();
        let loaded_chunk: Array<i16, Ix2> = batchtable.load_chunk(index);
        let output_result = loaded_chunk.to_pyarray(py);
        return output_result;
    }
}

#[pyclass()]
pub struct BatchTableI8Py {
    pub batchtable: Arc<RwLock<BatchTablesTypWrapper>>,

    #[pyo3(get)]
    pub shape: Vec<usize>,
}

#[pymethods]
impl BatchTableI8Py {
    fn add_chunk(&mut self, arr: PyReadonlyArray2<i8>) {
        let arr_view = arr.as_array();
        let mut batchtable = self.batchtable.write().unwrap();
        batchtable.add_chunk(&arr_view);
    }

    fn override_chunk(&mut self, arr: PyReadonlyArray2<i8>, index: usize) {
        let arr_view = arr.as_array();
        let mut batchtable = self.batchtable.write().unwrap();
        batchtable.override_chunk_with_array(&arr_view, index);
    }

    fn print_arr(&mut self) {
        let batchtable = self.batchtable.write().unwrap();
        for i in 0..batchtable.get_table_chunk_count() {
            let loaded_chunk: Array<i8, Ix2> = batchtable.load_chunk(i);
            array_helper::print_array_2D(&loaded_chunk);
        }
    }

    fn load_chunk<'py>(&mut self, py: Python<'py>, index: usize) -> &'py PyArray<i8, Ix2> {
        let batchtable = self.batchtable.write().unwrap();
        let loaded_chunk: Array<i8, Ix2> = batchtable.load_chunk(index);
        let output_result = loaded_chunk.to_pyarray(py);
        return output_result;
    }
}

#[pyclass()]
pub struct BatchTableF32Py {
    pub batchtable: Arc<RwLock<BatchTablesTypWrapper>>,

    #[pyo3(get)]
    pub shape: Vec<usize>,
}

#[pymethods]
impl BatchTableF32Py {
    fn add_chunk(&mut self, arr: PyReadonlyArray2<f32>) {
        let arr_view = arr.as_array();
        let mut batchtable = self.batchtable.write().unwrap();
        batchtable.add_chunk(&arr_view);
    }

    fn override_chunk(&mut self, arr: PyReadonlyArray2<f32>, index: usize) {
        let arr_view = arr.as_array();
        let mut batchtable = self.batchtable.write().unwrap();
        batchtable.override_chunk_with_array(&arr_view, index);
    }

    fn print_arr(&mut self) {
        let batchtable = self.batchtable.write().unwrap();
        for i in 0..batchtable.get_table_chunk_count() {
            let loaded_chunk: Array<f32, Ix2> = batchtable.load_chunk(i);
            array_helper::print_array_2D(&loaded_chunk);
        }
    }

    fn load_chunk<'py>(&mut self, py: Python<'py>, index: usize) -> &'py PyArray<f32, Ix2> {
        let batchtable = self.batchtable.write().unwrap();
        let loaded_chunk: Array<f32, Ix2> = batchtable.load_chunk(index);
        let output_result = loaded_chunk.to_pyarray(py);
        return output_result;
    }
}

#[pyclass()]
pub struct BatchTableF64Py {
    pub batchtable: Arc<RwLock<BatchTablesTypWrapper>>,

    #[pyo3(get)]
    pub shape: Vec<usize>,
}

#[pymethods]
impl BatchTableF64Py {
    fn add_chunk(&mut self, arr: PyReadonlyArray2<f64>) {
        let arr_view = arr.as_array();
        let mut batchtable = self.batchtable.write().unwrap();
        batchtable.add_chunk(&arr_view);
    }

    fn override_chunk(&mut self, arr: PyReadonlyArray2<f64>, index: usize) {
        let arr_view = arr.as_array();
        let mut batchtable = self.batchtable.write().unwrap();
        batchtable.override_chunk_with_array(&arr_view, index);
    }

    pub fn print_arr(&mut self) {
        let batchtable = self.batchtable.write().unwrap();
        for i in 0..batchtable.get_table_chunk_count() {
            let loaded_chunk: Array<f64, Ix2> = batchtable.load_chunk(i);
            array_helper::print_array_2D(&loaded_chunk);
        }
    }

    fn load_chunk<'py>(&mut self, py: Python<'py>, index: usize) -> &'py PyArray<f64, Ix2> {
        let batchtable = self.batchtable.write().unwrap();
        let loaded_chunk: Array<f64, Ix2> = batchtable.load_chunk(index);
        let output_result = loaded_chunk.to_pyarray(py);
        return output_result;
    }
}

#[pyclass()]
pub struct BatchTableStringPy {
    pub batchtable: Arc<RwLock<BatchTablesTypWrapper>>,

    #[pyo3(get)]
    pub shape: Vec<usize>,
}

#[pymethods]
impl BatchTableStringPy {
    fn add_chunk_wrapper(&mut self, vec: Vec<String>) {
        let rust_arr = Array::from_shape_vec((vec.len(), 1), vec).unwrap();
        self.add_chunk(rust_arr)
    }

    fn override_chunk_wrapper(&mut self, vec: Vec<String>, index: usize) {
        let rust_arr = Array::from_shape_vec((vec.len(), 1), vec).unwrap();
        self.override_chunk(rust_arr, index);
    }

    pub fn print_arr(&mut self) {
        let batchtable = self.batchtable.write().unwrap();
        for i in 0..batchtable.get_table_chunk_count() {
            let loaded_chunk: Array<String, Ix2> = batchtable.load_chunk(i);
            array_helper::print_array_2D(&loaded_chunk);
        }
    }

    fn get_loaded_string_chunk(&mut self, index: usize) -> Vec<String> {
        let loaded_chunk = self.load_chunk(index);
        let pylist_chunk = loaded_chunk.into_raw_vec();
        return pylist_chunk;
    }
}

impl BatchTableStringPy {
    fn add_chunk(&mut self, arr: Array2<String>) {
        let mut batchtable = self.batchtable.write().unwrap();
        batchtable.add_chunk(&arr);
    }

    fn override_chunk(&mut self, arr: Array2<String>, index: usize) {
        let mut batchtable = self.batchtable.write().unwrap();
        batchtable.override_chunk_with_array(&arr, index);
    }

    pub fn load_chunk(&mut self, index: usize) -> Array<String, Ix2> {
        let batchtable = self.batchtable.write().unwrap();
        let loaded_chunk: Array<String, Ix2> = batchtable.load_chunk(index);

        return loaded_chunk;
    }
}
