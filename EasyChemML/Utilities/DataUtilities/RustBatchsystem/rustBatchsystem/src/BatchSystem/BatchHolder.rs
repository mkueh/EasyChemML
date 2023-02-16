use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::fs;
use std::rc::Rc;
use std::sync::{Arc, RwLock};
use hex;

use digest::{Digest};
use ndarray::{Data, Dimension, IxDyn};
use ndarray_npy::{ReadableElement, WritableElement};
use sha2::{Sha256};

use crate::BatchSystem::BatchTablesImplementation::BatchTable::{BatchTable, BatchTablesTypWrapper};
use crate::BatchSystem::BatchTablesImplementation::BatchTable_FileSystem::BatchTable_FileSystem;
use crate::BatchSystem::BatchTablesImplementation::BatchTable_InMemory::BatchTable_InMemory;

#[derive(PartialEq, Clone)]
pub enum MemoryMode{
    InMemory,
    DirectIo
}

pub struct BatchHolder{
    pub stored_batchtables: HashMap<String, Arc<RwLock<BatchTablesTypWrapper>>>,
    pub root_folder:  PathBuf,
}

impl BatchHolder{

    pub fn new(path_str: &str) -> Self {
        let stored_batch_tables: HashMap<String, Arc<RwLock<BatchTablesTypWrapper>>> = HashMap::new();
        let path_obj = Path::new(path_str);

        if !path_obj.is_dir() && path_obj.exists() {
            let message = format!("The given Path is a File {} ... failed by removing this file", path_str);
            fs::remove_file(path_obj).expect(&*message);
        }else if path_obj.is_dir() {
            let message = format!("The given Path is not empty {} ... failed by removing dir", path_str);
            fs::remove_dir_all(path_obj).expect(&*message);
        }

        let message = format!("Try to create given Path {} ... but its failed", path_str);
        fs::create_dir_all(Path::new(path_str)).expect(&*message);

        BatchHolder { stored_batchtables: stored_batch_tables, root_folder: Path::new(path_str).to_path_buf()}
    }

    pub fn get_batchtable(&mut self, table_name:String) -> Arc<RwLock<BatchTablesTypWrapper>> {
        let mut load_batchtable = self.stored_batchtables.get_mut(&table_name).unwrap();
        return load_batchtable.clone();
    }

    pub fn get_mut_batchtable(&mut self, table_name:String) -> Arc<RwLock<BatchTablesTypWrapper>> {
        let mut load_batchtable = self.stored_batchtables.get(&table_name).unwrap();
        return load_batchtable.clone();
    }

    pub fn create_batchtable(&mut self, table_name:String, chunk_shape:Vec<usize>, memory_mode: MemoryMode){
        let h = Self::create_hash(table_name.as_str(), Sha256::default());
        let h_str =  hex::encode(h);

        let joined_path = self.root_folder.join(h_str);
        let new_dir_path_as_str = joined_path.into_os_string().into_string().unwrap();

        let path = Path::new(new_dir_path_as_str.as_str());
        if path.exists() && path.is_dir(){
            let message = format!("The given Path is not empty {} ... failed by removing dir", new_dir_path_as_str.clone());
            fs::remove_dir_all(Path::new(path.to_str().unwrap())).expect(&*message);
        }else if path.exists(){
            let message = format!("The given Path is a File {} ... failed by removing file", new_dir_path_as_str.clone());
            fs::remove_file(Path::new(path.to_str().unwrap())).expect(&*message);
        }

        let message = format!("Try to create given Path {} ... but its failed", new_dir_path_as_str.clone());
        fs::create_dir(Path::new(path.to_str().unwrap())).expect(&*message);

        if memory_mode == MemoryMode::InMemory {
            let batchtable = BatchTable_InMemory{table_folder : new_dir_path_as_str, table_name:table_name.clone(), table_chunk_count:0, chunk_shape, memory_mode, data_holder: vec![], first_chunk_size: 0 };
            self.stored_batchtables.insert(table_name, Arc::new(RwLock::new(BatchTablesTypWrapper::InMemory(batchtable))));
        } else if memory_mode == MemoryMode::DirectIo {
            let batchtable = BatchTable_FileSystem{table_folder : new_dir_path_as_str, table_name:table_name.clone(), table_chunk_count:0, chunk_shape, memory_mode, first_chunk_size: 0 };
            self.stored_batchtables.insert(table_name, Arc::new(RwLock::new(BatchTablesTypWrapper::DirectIo(batchtable))));
        };

    }

    pub fn remove_batchtable(&mut self, table_name:String){
        let remove_table = self.stored_batchtables.get_mut(table_name.as_str()).unwrap();
        remove_table.write().unwrap().delete();

        self.stored_batchtables.remove(table_name.as_str());
    }

    pub fn clean(&mut self){
        let mut hashmap = &mut self.stored_batchtables;
        for (key, value) in hashmap.iter_mut() {
            value.write().unwrap().delete()
        }

        fs::remove_dir_all(self.root_folder.to_str().unwrap()).unwrap_or(());

        self.stored_batchtables.clear();
    }

    fn create_hash<H: Digest>(msg: &str, mut hasher: H) -> Vec<u8> {
        hasher.update(msg);
        hasher.finalize().as_slice().to_vec()
    }

}