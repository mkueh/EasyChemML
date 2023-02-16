use std::borrow::BorrowMut;
use std::io;
use std::io::{Cursor, Seek, SeekFrom, Write};
use digest::generic_array::arr;
use lz4::{Decoder, EncoderBuilder};
use ndarray::{Array, ArrayBase, Data, Dimension, IxDyn, RawData};
use ndarray_npy::{NpzReader, NpzWriter, ReadableElement, WritableElement};
use serde::de::DeserializeOwned;
use serde::Serialize;
use crate::BatchSystem::BatchHolder::MemoryMode;
use crate::BatchSystem::BatchTablesImplementation::BatchTable::{BatchTable, BatchTableMemoryBased};
use crate::BatchSystem::BatchTablesImplementation::DataSerializing::Serializer;
use crate::BatchSystem::BatchTablesImplementation::DataSerializing::Serializer::SerializerAlgo;

#[derive(Clone)]
pub struct BatchTable_InMemory{
    pub table_folder: String,
    pub table_name: String,
    pub table_chunk_count: usize,
    pub chunk_shape: Vec<usize>,
    pub memory_mode: MemoryMode,

    pub(crate) first_chunk_size: usize,
    pub(crate) data_holder: Vec<Vec<u8>>
}

impl BatchTable_InMemory{

    fn generate_serial_data<T:Data, D: Dimension + Serialize>(&mut self, array: &ArrayBase<T,D>) -> Vec<u8> where <T as RawData>::Elem: serde::ser::Serialize{
        let compress_data_stream = Cursor::new(Vec::new());

        let mut serialized_data_stream = Serializer::to_vec(array, SerializerAlgo::Postcard);
        serialized_data_stream.seek(SeekFrom::Start(0)).unwrap( );

        let mut encoder = EncoderBuilder::new().level(4).build(compress_data_stream).unwrap();
        io::copy(&mut serialized_data_stream, &mut encoder).unwrap();
        let (mut out, res) = encoder.finish();
        let out_vec_pointer= out.into_inner();

        out_vec_pointer
    }
}

impl BatchTable for BatchTable_InMemory {
    fn add_chunk<T:Data, D: Dimension + Serialize>(&mut self, array: &ArrayBase<T,D>) where <T as RawData>::Elem: serde::ser::Serialize{
        if array.shape()[0] > self.chunk_shape[0]{
            panic!("The added chunk is larger than the chunk shape");
        }

        if self.table_chunk_count == 0{
            self.first_chunk_size = array.shape()[0];
        }else if array.shape()[0] > self.first_chunk_size{
            panic!("The added chunk is larger than the chunk before");
        }

        let processed_in_memory_data = self.generate_serial_data(array);
        self.data_holder.push(processed_in_memory_data);
        self.table_chunk_count += 1;
    }

    fn load_chunk<T:DeserializeOwned, D: Dimension + DeserializeOwned>(&self, index: usize) -> Array<T,D>{
        let compress_data_stream = Cursor::new(self.data_holder.get(index).unwrap());

        let mut decoder = Decoder::new(compress_data_stream).unwrap();
        let mut serde_data_stream = Cursor::new(Vec::new());
        io::copy(&mut decoder, &mut serde_data_stream).unwrap();

        Serializer::from_vec(serde_data_stream, SerializerAlgo::Postcard)
    }

    fn get_size(&self) -> usize {
        self.table_chunk_count
    }

    fn get_shape(&self) -> Vec<usize>{
        self.chunk_shape.clone()
    }

    fn get_table_chunk_count(&self) -> usize {
        self.table_chunk_count
    }

    fn delete(&mut self) {
        self.data_holder = vec![];
    }

    fn get_table_name(&self) -> String{
        self.table_name.clone()
    }

    fn get_memory_mode(&self) -> MemoryMode {
        self.memory_mode.clone()
    }

    fn override_chunk_with_array<T:Data, D: Dimension + Serialize>(&mut self, array: &ArrayBase<T, D>, index: usize) where <T as RawData>::Elem: serde::ser::Serialize{
        let processed_in_memory_data = self.generate_serial_data(array);
        self.data_holder[index] = processed_in_memory_data;
    }
}

impl BatchTableMemoryBased for BatchTable_InMemory {

    fn override_chunk_with_pointer(&mut self, vec: Vec<u8>, index: usize) {
        self.data_holder[index] = vec
    }

    fn get_raw_data_pointer(&self, index:usize) -> Vec<u8>{
        self.data_holder[index].iter().cloned().collect()
    }

}