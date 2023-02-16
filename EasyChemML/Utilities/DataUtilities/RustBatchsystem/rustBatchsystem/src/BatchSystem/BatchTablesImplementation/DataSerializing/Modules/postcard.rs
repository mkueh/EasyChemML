use std::io::{Cursor, Seek, SeekFrom};
use ndarray::{Array, ArrayBase, Data, Dimension, RawData};
use serde::de::DeserializeOwned;
use serde::Serialize;
use postcard;
use postcard::to_allocvec;



pub(crate) fn to_vec<T:Data, D: Dimension + Serialize>(array:&ArrayBase<T, D>) -> Cursor<Vec<u8>> where <T as RawData>::Elem: serde::ser::Serialize{
    let mut serde_data_stream = Cursor::new(postcard::to_allocvec(array).unwrap());
    serde_data_stream.seek(SeekFrom::Start(0)).unwrap( );
    serde_data_stream
}

pub(crate) fn from_vec<T: DeserializeOwned, D: Dimension + DeserializeOwned>(mut input:Cursor<Vec<u8>>) -> Array<T, D>{
    postcard::from_bytes(input.get_ref()).unwrap()
}