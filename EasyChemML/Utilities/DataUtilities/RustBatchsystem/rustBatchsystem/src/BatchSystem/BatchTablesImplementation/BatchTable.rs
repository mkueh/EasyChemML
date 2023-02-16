use std::ops::DerefMut;
use std::path::PathBuf;
use ndarray::{Array, ArrayBase, ArrayView, Data, Dimension, RawData};
use ndarray_npy::{ReadableElement, WritableElement};
use serde::{Deserialize, Serialize};
use serde::de::DeserializeOwned;
use crate::BatchSystem::BatchHolder::MemoryMode;
use crate::BatchSystem::BatchTablesImplementation::BatchTable_FileSystem::BatchTable_FileSystem;
use crate::BatchSystem::BatchTablesImplementation::BatchTable_InMemory::BatchTable_InMemory;

pub trait BatchTable{

    fn add_chunk<T:Data, D: Dimension + Serialize>(&mut self, array: &ArrayBase<T,D>) where <T as RawData>::Elem: serde::ser::Serialize;

   fn load_chunk<T:DeserializeOwned, D: Dimension + DeserializeOwned>(&self, index: usize) -> Array<T,D>;

    fn get_size(&self) -> usize;

    fn get_shape(&self) -> Vec<usize>;

    fn get_table_chunk_count(&self) -> usize;

    fn delete(&mut self);

    fn get_table_name(&self) -> String;

    fn get_memory_mode(&self) -> MemoryMode;

    fn override_chunk_with_array<T:Data, D: Dimension + Serialize>(&mut self, array: &ArrayBase<T, D>, index: usize) where <T as RawData>::Elem: serde::ser::Serialize;

}

pub trait BatchTablePathBased{

    fn override_chunk_with_file(&mut self, file_path:&str, index:usize);

    fn get_chunk_path(&self, index:usize) -> PathBuf;

}

pub trait BatchTableMemoryBased{

    fn override_chunk_with_pointer(&mut self, vec:Vec<u8>, index:usize);

    fn get_raw_data_pointer(&self, index:usize) -> Vec<u8>;

}

#[derive(Clone)]
pub enum BatchTablesTypWrapper {
    InMemory(BatchTable_InMemory),
    DirectIo(BatchTable_FileSystem),
}

impl BatchTable for BatchTablesTypWrapper{

    fn add_chunk<T:Data, D: Dimension + Serialize>(&mut self, array: &ArrayBase<T,D>) where <T as RawData>::Elem: serde::ser::Serialize{
        match self{
            BatchTablesTypWrapper::InMemory(batchtable) => {batchtable.add_chunk(array)}
            BatchTablesTypWrapper::DirectIo(batchtable) => {batchtable.add_chunk(array)}
        }
    }

    fn load_chunk<T:DeserializeOwned, D: Dimension + DeserializeOwned>(&self, index: usize) -> Array<T,D>{
        match self{
            BatchTablesTypWrapper::InMemory(batchtable) => {batchtable.load_chunk(index)}
            BatchTablesTypWrapper::DirectIo(batchtable) => {batchtable.load_chunk(index)}
        }
    }

    fn get_size(&self) -> usize {
        match self{
            BatchTablesTypWrapper::InMemory(batchtable) => {batchtable.get_size()}
            BatchTablesTypWrapper::DirectIo(batchtable) => {batchtable.get_size()}
        }
    }

    fn get_shape(&self) -> Vec<usize> {
        match self{
            BatchTablesTypWrapper::InMemory(batchtable) => {batchtable.get_shape()}
            BatchTablesTypWrapper::DirectIo(batchtable) => {batchtable.get_shape()}
        }
    }

    fn get_table_chunk_count(&self) -> usize {
        match self{
            BatchTablesTypWrapper::InMemory(batchtable) => {batchtable.get_table_chunk_count()}
            BatchTablesTypWrapper::DirectIo(batchtable) => {batchtable.get_table_chunk_count()}
        }
    }

    fn delete(&mut self) {
        match self{
            BatchTablesTypWrapper::InMemory(batchtable) => {batchtable.delete()}
            BatchTablesTypWrapper::DirectIo(batchtable) => {batchtable.delete()}
        }
    }

    fn get_table_name(&self) -> String{
        match self{
            BatchTablesTypWrapper::InMemory(batchtable) => {batchtable.get_table_name()}
            BatchTablesTypWrapper::DirectIo(batchtable) => {batchtable.get_table_name()}
        }
    }

    fn get_memory_mode(&self) -> MemoryMode {
        match self{
            BatchTablesTypWrapper::InMemory(batchtable) => {batchtable.get_memory_mode()}
            BatchTablesTypWrapper::DirectIo(batchtable) => {batchtable.get_memory_mode()}
        }
    }

    fn override_chunk_with_array<T:Data, D: Dimension + Serialize>(&mut self, array: &ArrayBase<T, D>, index: usize) where <T as RawData>::Elem: serde::ser::Serialize{
        match self{
            BatchTablesTypWrapper::InMemory(batchtable) => {batchtable.override_chunk_with_array(array, index)}
            BatchTablesTypWrapper::DirectIo(batchtable) => {batchtable.override_chunk_with_array(array, index)}
        }
    }
}

impl BatchTablesTypWrapper{

    pub fn is_batch_table_pathbased(&self) -> bool{
        match self {
            BatchTablesTypWrapper::InMemory(_) => { false }
            BatchTablesTypWrapper::DirectIo(_) => { true }
        }
    }

    pub fn is_batch_table_memorybased(&self) -> bool{
        match self {
            BatchTablesTypWrapper::InMemory(_) => { true }
            BatchTablesTypWrapper::DirectIo(_) => { false }
        }
    }

    pub fn to_batch_table_file_system(&mut self) -> Option<&mut BatchTable_FileSystem>{
        match self {
            BatchTablesTypWrapper::InMemory(_) => { None }
            BatchTablesTypWrapper::DirectIo(batchtable) => { Some(batchtable) }
        }
    }

    pub fn to_batch_table_in_memory(&mut self) -> Option<&mut BatchTable_InMemory>{match self {
            BatchTablesTypWrapper::InMemory(batchtable) => { Some(batchtable) }
            BatchTablesTypWrapper::DirectIo(_) => { None }
        }}
    }


