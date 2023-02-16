#[cfg(test)]
mod tests {
    use crate::BatchSystem::BatchHolder::{BatchHolder, MemoryMode};
    use crate::Utilities::array_helper;
    use digest::generic_array::arr;
    use ndarray;
    use ndarray::{Array, Array2, ArrayBase, Dimension, Ix2, OwnedRepr};
    use rand::Rng;
    use rayon::prelude::*;
    use std::env;
    use std::ops::Add;
    use crate::BatchSystem::BatchTablesImplementation::BatchTable::{BatchTable, BatchTablesTypWrapper};

    #[test]
    fn test_create_override_batch() {
        let mut test = env::current_exe().unwrap();
        test.pop();
        test.pop();
        test.pop();
        test.pop();
        test.push("..");
        test.push("test");
        let mut path = test.to_str().unwrap();

        let mut batchholder: BatchHolder = BatchHolder::new(path);
        batchholder.create_batchtable(String::from("test"), vec![100,2048], MemoryMode::DirectIo);

        let mut table = batchholder.get_batchtable(String::from("test"));
        {
            let mut table = table.write().unwrap();

            let first_array_of_fp = create_fingerprints_vector();

            table.add_chunk(&first_array_of_fp.clone().into_dyn());

            let mut loaded_array: Array<i32, Ix2> = table.load_chunk(0);

            let second_array_of_fp = create_fingerprints_vector();
            table.override_chunk_with_array(&second_array_of_fp, 0);

            let mut loaded_array: Array<i32, Ix2> = table.load_chunk(0);
            assert_eq!(loaded_array, second_array_of_fp);
            assert_ne!(loaded_array, first_array_of_fp);
        }
        batchholder.clean();
    }

    #[test]
    fn test_create_load_batchsystem() {
        let mut test = env::current_exe().unwrap();
        test.pop();
        test.pop();
        test.pop();
        test.pop();
        test.push("..");
        test.push("test");
        let mut path = test.to_str().unwrap();

        let mut batchholder: BatchHolder = BatchHolder::new(path);
        batchholder.create_batchtable(String::from("test"), vec![100,2048], MemoryMode::DirectIo);

        {
        let mut table = batchholder.get_batchtable(String::from("test"));
        let mut table = table.write().unwrap();

        let array_of_fp = create_fingerprints_vector();

        table.add_chunk(&array_of_fp.to_owned());

        let mut loaded_array: Array<i32, Ix2> = table.load_chunk(0);
        array_helper::print_array_2D(&loaded_array);

        assert_eq!(loaded_array, array_of_fp);
        }
        batchholder.clean()
    }

    fn create_fingerprints_vector() -> Array2<i32> {
        let width = 2048;
        let height = 100;

        let mut rng = rand::thread_rng();
        let mut array: Array2<i32> = Array2::zeros((height, width));

        for (index_row, item) in array.iter_mut().enumerate() {
            let random_number: i32 = rng.gen_range(0..2);
            *item = random_number
        }
        return array;
    }
}
