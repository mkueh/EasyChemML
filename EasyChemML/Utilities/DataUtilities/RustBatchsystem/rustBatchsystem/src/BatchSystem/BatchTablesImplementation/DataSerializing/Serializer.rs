use std::io::{Cursor, Seek, SeekFrom};
use ndarray::{Array, ArrayBase, Data, Dimension, RawData};
use serde::de::DeserializeOwned;
use serde::Serialize;
use crate::BatchSystem::BatchTablesImplementation::DataSerializing::Modules::json_serde;
use crate::BatchSystem::BatchTablesImplementation::DataSerializing::Modules::postcard;
use crate::BatchSystem::BatchTablesImplementation::DataSerializing::Modules::flexbuffers;

pub enum SerializerAlgo {
    SerdeJson,
    Postcard,
    Flexbuffers,
}

pub fn to_vec<T:Data, D: Dimension + Serialize>(array:&ArrayBase<T, D>, serializer:SerializerAlgo) -> Cursor<Vec<u8>> where <T as RawData>::Elem: serde::ser::Serialize{
    match serializer {
        SerializerAlgo::SerdeJson => {json_serde::to_vec(array)}
        SerializerAlgo::Postcard => {postcard::to_vec(array)}
        SerializerAlgo::Flexbuffers => {flexbuffers::to_vec(array)}
    }
}

pub fn from_vec<T: DeserializeOwned, D: Dimension + DeserializeOwned>(mut input:Cursor<Vec<u8>>, serializer:SerializerAlgo) -> Array<T, D>{
    input.seek(SeekFrom::Start(0)).unwrap();
    match serializer {
        SerializerAlgo::SerdeJson => {json_serde::from_vec(input)}
        SerializerAlgo::Postcard => {postcard::from_vec(input)}
        SerializerAlgo::Flexbuffers => {flexbuffers::from_vec(input)}
    }
}