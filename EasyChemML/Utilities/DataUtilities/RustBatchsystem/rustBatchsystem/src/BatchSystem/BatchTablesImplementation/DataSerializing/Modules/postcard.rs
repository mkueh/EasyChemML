use ndarray::{Array, ArrayBase, Data, Dimension, RawData};
use postcard;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::io::{Cursor, Seek, SeekFrom};

pub(crate) fn to_vec<T: Data, D: Dimension + Serialize>(array: &ArrayBase<T, D>) -> Cursor<Vec<u8>>
where
    <T as RawData>::Elem: serde::ser::Serialize,
{
    let mut serde_data_stream = Cursor::new(postcard::to_allocvec(array).unwrap());
    serde_data_stream.seek(SeekFrom::Start(0)).unwrap();
    serde_data_stream
}

pub(crate) fn from_vec<T: DeserializeOwned, D: Dimension + DeserializeOwned>(
    input: Cursor<Vec<u8>>,
) -> Array<T, D> {
    postcard::from_bytes(input.get_ref()).unwrap()
}
