use std::rc::Rc;
use std::env;
use ndarray::Ix2;
use numpy::{dtype, PyReadonlyArray1};


use pyo3::prelude::*;
use pyo3::types::PyList;
use crate::BatchFunctions::py_BatchListFunctions_duplicates::{Duplicat_result_i16, Duplicat_result_i32, Duplicat_result_i64, Duplicat_result_i8, BatchListFunctions_duplicates_py};

use crate::BatchSystem::BatchHolder::{BatchHolder, MemoryMode};
use crate::BatchSystem::BatchTablesImplementation::BatchTable::{BatchTable};
use crate::BatchFunctions::py_BatchSort::BatchSorter_Radix_py;
use crate::BatchSystem::PythonInterfaces::py_rustBatchTable::{BatchTableF32Py, BatchTableF64Py, BatchTableI16Py, BatchTableI32Py, BatchTableI64Py, BatchTableI8Py};
use crate::BatchSystem::PythonInterfaces::DType_Converter::{BatchDatatyp};


#[pyclass]
pub struct BatchHolder_py  {
    pub batchholder: BatchHolder
}

#[pymethods]
impl BatchHolder_py {

    #[new]
    pub fn new(path: String) -> BatchHolder_py {
        BatchHolder_py { batchholder: BatchHolder::new(&path)}
    }

    #[pyo3(text_signature = "($self, table_name, chunk_shape)")]
    pub fn create_new_table(&mut self, table_name: String, chunk_shape: Vec<usize>, memory_mode_string: String){
        if memory_mode_string == "InMemory"{
            self.batchholder.create_batchtable(table_name, chunk_shape, MemoryMode::InMemory);
        }else if memory_mode_string == "DirectIO" {
            self.batchholder.create_batchtable(table_name, chunk_shape, MemoryMode::DirectIo);
        }else {
            panic!("MemoryMode not found {}", memory_mode_string);
        }

    }

    #[pyo3(text_signature = "($self, table_name, chunk_shape)")]
    pub fn get_batchtable_i64(&mut self, table_name:String) -> PyResult<BatchTableI64Py>{
        let batchtable = self.batchholder.get_batchtable(table_name);
        let shape = batchtable.read().unwrap().get_shape();
        Ok(BatchTableI64Py{ batchtable, shape })
    }

    #[pyo3(text_signature = "($self, table_name, chunk_shape)")]
    pub fn get_batchtable_i32(&mut self, table_name:String) -> PyResult<BatchTableI32Py>{
        let batchtable = self.batchholder.get_batchtable(table_name);
        let shape = batchtable.read().unwrap().get_shape();
        Ok(BatchTableI32Py{ batchtable, shape })
    }

    #[pyo3(text_signature = "($self, table_name, chunk_shape)")]
    pub fn get_batchtable_i16(&mut self, table_name:String) -> PyResult<BatchTableI16Py>{
        let batchtable = self.batchholder.get_batchtable(table_name);
        let shape = batchtable.read().unwrap().get_shape();
        Ok(BatchTableI16Py{ batchtable, shape })
    }

    #[pyo3(text_signature = "($self, table_name, chunk_shape)")]
    pub fn get_batchtable_i8(&mut self, table_name:String) -> PyResult<BatchTableI8Py>{
        let batchtable = self.batchholder.get_batchtable(table_name);
        let shape = batchtable.read().unwrap().get_shape();
        Ok(BatchTableI8Py{ batchtable, shape })
    }

    #[pyo3(text_signature = "($self, table_name, chunk_shape)")]
    pub fn get_batchtable_f64(&mut self, table_name:String) -> PyResult<BatchTableF64Py>{
        let batchtable = self.batchholder.get_batchtable(table_name);
        let shape = batchtable.read().unwrap().get_shape();
        Ok(BatchTableF64Py{ batchtable, shape })
    }

    #[pyo3(text_signature = "($self, table_name, chunk_shape)")]
    pub fn get_batchtable_f32(&mut self, table_name:String) -> PyResult<BatchTableF32Py>{
        let batchtable = self.batchholder.get_batchtable(table_name);
        let shape = batchtable.read().unwrap().get_shape();
        Ok(BatchTableF32Py{ batchtable, shape })
    }

    #[pyo3(text_signature = "($self)")]
    pub fn clean(&mut self){
        self.batchholder.clean();
    }
}

#[pymodule]
fn pyRustBatchsystem(_py: Python, m: &PyModule) -> PyResult<()> {
    env::set_var("RUST_BACKTRACE", "1");
    m.add_class::<BatchHolder_py>()?;

    m.add_class::<BatchTableI64Py>()?;
    m.add_class::<BatchTableI32Py>()?;
    m.add_class::<BatchTableI16Py>()?;
    m.add_class::<BatchTableI8Py>()?;
    m.add_class::<BatchTableF64Py>()?;
    m.add_class::<BatchTableF32Py>()?;
    m.add_class::<BatchSorter_Radix_py>()?;

    m.add_class::<Duplicat_result_i8>()?;
    m.add_class::<Duplicat_result_i16>()?;
    m.add_class::<Duplicat_result_i32>()?;
    m.add_class::<Duplicat_result_i64>()?;
    m.add_class::<BatchListFunctions_duplicates_py>()?;

    Ok(())
}