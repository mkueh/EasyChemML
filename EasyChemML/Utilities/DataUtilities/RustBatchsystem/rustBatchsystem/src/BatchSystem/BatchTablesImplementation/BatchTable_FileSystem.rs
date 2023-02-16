use std::fmt::format;
use std::fs::File;
use std::fs;
use std::ops::Add;
use std::path::{Path, PathBuf};
use ndarray::{Array, Array2, ArrayBase, Data, Dimension, Ix2, IxDyn, RawData};
use ndarray_npy::{NpzReader, NpzWriter, ReadableElement, WritableElement};
use std::io::{Cursor, Seek, SeekFrom};
use lz4::{Decoder, EncoderBuilder};
use std::io::{self, Result};
use serde::{Deserialize, Serialize};
use serde::de::DeserializeOwned;
use crate::BatchSystem::BatchHolder::MemoryMode;
use crate::BatchSystem::BatchTablesImplementation::BatchTable::{BatchTable, BatchTablePathBased};
use crate::BatchSystem::BatchTablesImplementation::DataSerializing::Serializer;
use crate::BatchSystem::BatchTablesImplementation::DataSerializing::Serializer::SerializerAlgo;


#[derive(Clone)]
pub struct BatchTable_FileSystem{
    pub table_folder: String,
    pub table_name: String,
    pub table_chunk_count: usize,
    pub chunk_shape: Vec<usize>,
    pub memory_mode: MemoryMode,

    pub(crate) first_chunk_size: usize
}

impl BatchTable_FileSystem{
    fn write_file<T:Data, D: Dimension + Serialize>(&mut self, array:&ArrayBase<T, D>, path:PathBuf) where <T as RawData>::Elem: serde::ser::Serialize{
            let mut serialized_data_stream = Serializer::to_vec(array, SerializerAlgo::Postcard);
            serialized_data_stream.seek(SeekFrom::Start(0)).unwrap( );

            let output_file = File::create(path.clone()).expect(&*format!("Creating file failed: {}", path.to_str().unwrap()));
            let mut encoder = EncoderBuilder::new().level(4).build(output_file).unwrap();
            io::copy(&mut serialized_data_stream, &mut encoder).unwrap();
            let (out, res) = encoder.finish();
    }

    fn read_file<'de,T: DeserializeOwned, D: Dimension + DeserializeOwned>(&self, path: PathBuf) -> Array<T, D>{
            let mut decoder = Decoder::new(File::open(path.clone()).unwrap_or_else(|_| panic!("Cannot find Path: {}", path.clone().to_str().unwrap()))).unwrap();
            let mut file = Cursor::new(Vec::new());
            io::copy(&mut decoder, &mut file).unwrap();

            Serializer::from_vec(file, SerializerAlgo::Postcard)
    }
}

impl BatchTablePathBased for BatchTable_FileSystem {
    fn override_chunk_with_file(&mut self, file_path:&str, index:usize){
        let chunk_path = self.get_chunk_path(index);
        fs::remove_file(chunk_path.clone()).unwrap();
        fs::copy(file_path.to_string(), chunk_path).expect("copy of existing chunk fails");
    }

    fn get_chunk_path(&self, index: usize)  -> PathBuf {
        let mut filename = String::new();
        filename = filename.add("chunk");
        filename = filename.add(index.to_string().as_str());
        filename = filename.add(".npz");

        Path::new(self.table_folder.as_str()).join(filename)
    }
}

impl BatchTable for BatchTable_FileSystem {

     fn add_chunk<T:Data, D: Dimension + Serialize>(&mut self, array: &ArrayBase<T,D>) where <T as RawData>::Elem: serde::ser::Serialize{
        if array.shape()[0] > self.chunk_shape[0]{
            panic!("The added chunk is larger than the chunk shape");
        }

        if self.table_chunk_count == 0{
            self.first_chunk_size = array.shape()[0];
        }else if array.shape()[0] > self.first_chunk_size{
            panic!("The added chunk is larger than the chunk before");
        }

        let chunk_path = self.get_chunk_path(self.table_chunk_count);
        self.write_file(array, chunk_path);
        self.table_chunk_count += 1;
    }

    fn load_chunk<T:DeserializeOwned, D: Dimension + DeserializeOwned>(&self, index: usize) -> Array<T,D>{
        let chunk_path = self.get_chunk_path(index);
        self.read_file(chunk_path)
    }

    fn get_size(&self) -> usize {
        self.table_chunk_count
    }

    fn get_shape(&self) -> Vec<usize>{
        return self.chunk_shape.clone();
    }

    fn get_table_chunk_count(&self) -> usize {
        return self.table_chunk_count.clone();
    }

    fn delete(&mut self) {
        fs::remove_dir_all(self.table_folder.as_str()).unwrap_or(());
    }

    fn get_table_name(&self) -> String{
        return self.table_name.clone();
    }

    fn get_memory_mode(&self) -> MemoryMode {
        return self.memory_mode.clone();
    }

    fn override_chunk_with_array<T: Data, D: Dimension + Serialize>(&mut self, array: &ArrayBase<T, D>, index: usize) where <T as RawData>::Elem: serde::ser::Serialize{
        let chunk_path = self.get_chunk_path(index);
        fs::remove_file(chunk_path.clone()).unwrap();
        self.write_file(array, chunk_path);
    }
}