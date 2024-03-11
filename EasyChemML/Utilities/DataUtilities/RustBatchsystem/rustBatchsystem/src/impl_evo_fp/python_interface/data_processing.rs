use crate::impl_evo_fp::get_data;
use crate::BatchSystem::PythonInterfaces::py_rustBatchTable::{
    BatchTableF64Py, BatchTableStringPy,
};
use ndarray::{arr2, Array, Axis, Ix1, Ix2};
use rdkit::ROMol;

pub fn convert_smiles_batch_data(smiles_batch_table: &BatchTableStringPy) -> Vec<Vec<ROMol>> {
    // Returns a reference to the batchtable, Arc<RwLock<BatchTablesTypWrapper>>
    let ground_batchtable = smiles_batch_table.batchtable.clone();
    let mut batchtable = ground_batchtable.read().unwrap();
    let chunk_count = batchtable.get_table_chunk_count();
    let mut converted_chunks: Vec<Vec<ROMol>> = Vec::new();
    for chunk_index in 0..chunk_count {
        let loaded_chunk: Array<String, Ix2> = batchtable.load_chunk(chunk_index);
        let mut mol_data = get_data::convert_data_to_mol(&arr2(&loaded_chunk));
        converted_chunks.append(&mut mol_data);
    }
    converted_chunks
}

pub fn convert_target_batch_data(target_batch_table: &BatchTableF64Py) -> Array<f64, Ix1> {
    let ground_batchtable = target_batch_table.batchtable.clone();
    let mut batchtable = ground_batchtable.read().unwrap();
    let chunk_count = batchtable.get_table_chunk_count();

    let mut target_chunks = Array::zeros(0);
    for chunk_index in 0..chunk_count {
        let loaded_chunk: Array<f64, Ix2> = batchtable.load_chunk(chunk_index);
        let flat_chunk = loaded_chunk.remove_axis(Axis(1));
        target_chunks.append(Axis(0), flat_chunk.view()).unwrap();
    }
    target_chunks
}
