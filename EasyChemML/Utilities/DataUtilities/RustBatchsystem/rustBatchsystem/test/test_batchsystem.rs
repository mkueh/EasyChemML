#[cfg(test)]
mod tests {
    use crate::BatchSystem::BatchHolder::{BatchHolder, MemoryMode};

    use ndarray;
    use ndarray::{arr2, Array, Array2, Dimension, Ix2};
    use rand::Rng;
    use std::env;

    use crate::BatchSystem::BatchTablesImplementation::BatchTable::BatchTable;

    #[test]
    fn test_create_override_batch() {
        let mut test = env::current_exe().unwrap();
        test.pop();
        test.pop();
        test.pop();
        test.pop();
        test.push("..");
        test.push("test");
        let path = test.to_str().unwrap();

        let mut batchholder: BatchHolder = BatchHolder::new(path);
        batchholder.create_batchtable(String::from("test"), vec![100, 2048], MemoryMode::DirectIo);

        let table = batchholder.get_batchtable(String::from("test"));
        {
            let mut table = table.write().unwrap();

            let first_array_of_fp = create_fingerprints_vector();

            table.add_chunk(&first_array_of_fp.clone().into_dyn());

            let _loaded_array: Array<i32, Ix2> = table.load_chunk(0);

            let second_array_of_fp = create_fingerprints_vector();
            table.override_chunk_with_array(&second_array_of_fp, 0);

            let loaded_array: Array<i32, Ix2> = table.load_chunk(0);
            assert_eq!(loaded_array, second_array_of_fp);
            assert_ne!(loaded_array, first_array_of_fp);
        }
        batchholder.clean();
    }

    #[test]
    fn test_create_override_string_batch() {
        let mut test = env::current_exe().unwrap();
        test.pop();
        test.pop();
        test.pop();
        test.pop();
        test.push("..");
        test.push("test1");
        let path = test.to_str().unwrap();

        let mut batchholder: BatchHolder = BatchHolder::new(path);
        batchholder.create_batchtable(String::from("test"), vec![11, 1], MemoryMode::DirectIo);
        let table = batchholder.get_batchtable(String::from("test"));
        {
            let mut table = table.write().unwrap();

            let first_array_of_strings = create_string_array();
            table.add_chunk(&first_array_of_strings.clone().into_dyn());

            let _loaded_array: Array<String, Ix2> = table.load_chunk(0);

            let mut second_array_of_strings = first_array_of_strings.clone();
            second_array_of_strings[[0, 0]] = "O".to_string();
            table.override_chunk_with_array(&second_array_of_strings, 0);

            let loaded_array: Array<String, Ix2> = table.load_chunk(0);
            assert_eq!(loaded_array, second_array_of_strings);
            assert_ne!(loaded_array, first_array_of_strings);
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
        test.push("test2");
        let path = test.to_str().unwrap();

        let mut batchholder: BatchHolder = BatchHolder::new(path);
        batchholder.create_batchtable(String::from("test"), vec![100, 2048], MemoryMode::DirectIo);

        {
            let table = batchholder.get_batchtable(String::from("test"));
            let mut table = table.write().unwrap();

            let array_of_fp = create_fingerprints_vector();

            table.add_chunk(&array_of_fp.to_owned());

            let loaded_array: Array<i32, Ix2> = table.load_chunk(0);
            // array_helper::print_array_2D(&loaded_array);

            assert_eq!(loaded_array, array_of_fp);
        }
        batchholder.clean()
    }

    #[test]
    fn test_create_load_string_batchsystem() {
        let mut test = env::current_exe().unwrap();
        test.pop();
        test.pop();
        test.pop();
        test.pop();
        test.push("..");
        test.push("test3");
        let path = test.to_str().unwrap();

        let mut batchholder: BatchHolder = BatchHolder::new(path);
        batchholder.create_batchtable(String::from("test"), vec![11, 1], MemoryMode::DirectIo);

        {
            let table = batchholder.get_batchtable(String::from("test"));
            let mut table = table.write().unwrap();

            let array_of_strings = create_string_array();
            table.add_chunk(&array_of_strings.clone().into_dyn());

            let loaded_array: Array<String, Ix2> = table.load_chunk(0);
            // array_helper::print_array_2D(&loaded_array);

            assert_eq!(loaded_array, array_of_strings);
        }
        batchholder.clean()
    }

    fn create_string_array() -> Array2<String> {
        arr2(&[
            ["Cn1c(CN2CCN(c3ccc(Cl)cc3)CC2)nc2ccccc21".to_string()],
            ["COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1".to_string()],
            ["O=C(NC1Cc2ccccc2N(C[C@@H](O)CO)C1=O)c1cc2cc(Cl)sc2[nH]1".to_string()],
            ["Cc1cccc(C[C@H](NC(=O)c2cc(C(C)(C)C)nn2C)C(=O)NCC#N)c1".to_string()],
            ["OC1(C#Cc2ccc(-c3ccccc3)cc2)CN2CCC1CC2".to_string()],
        ])
    }

    fn create_fingerprints_vector() -> Array2<i32> {
        let width = 2048;
        let height = 100;

        let mut rng = rand::thread_rng();
        let mut array: Array2<i32> = Array2::zeros((height, width));

        for (_index_row, item) in array.iter_mut().enumerate() {
            let random_number: i32 = rng.gen_range(0..2);
            *item = random_number
        }
        return array;
    }
}
