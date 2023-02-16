#[cfg(test)]
mod tests {
    use std::env;
    use std::fs;
    use std::fs::File;
    use std::time::Instant;
    use digest::generic_array::arr;
    use ndarray_npy::{NpzReader, NpzWriter};
    use ndarray;
    use ndarray::{Array2, ArrayBase, Axis, Ix2, OwnedRepr};
    use rand::Rng;
    use rand::seq::index::IndexVec::USize;
    use crate::Utilities::array_helper;
    use rayon::prelude::*;
    use crate::BatchFunctions::{BatchListFunctions_duplicates, BatchSort};
    use crate::BatchFunctions::BatchListFunctions_duplicates::duplicat_result;
    use crate::BatchSystem::BatchHolder::{BatchHolder, MemoryMode};
    use crate::BatchSystem::BatchTablesImplementation::BatchTable::BatchTable;

    #[test]
    fn test_small_array_to_file_find_duplicates() {
        let height = 100;
        let width = 100;
        let chunk_count = 10;

        let mut test = env::current_exe().unwrap();
        test.pop();
        test.pop();
        test.pop();
        test.pop();
        test.push("..");
        test.push("test");
        let mut path = test.to_str().unwrap();

        let mut batchholder: BatchHolder = BatchHolder::new(path);
        batchholder.create_batchtable(String::from("test"), vec![height, width], MemoryMode::InMemory);

        let mut table = batchholder.get_batchtable(String::from("test"));
        let mut test_vec: Vec<Vec<i32>> = Vec::new();

        {
            let mut table = table.write().unwrap();
            for i in 0..chunk_count {
                let ( mut arr, mut vec) = create_fingerprints_vector(width, height, true);
                test_vec.extend(vec);
                table.add_chunk( &arr);
            }
        }

        test_vec.par_sort_unstable();
        //println!("--------------------------");

        let mut test = env::current_exe().unwrap();
        test.pop();
        test.pop();
        test.pop();
        test.pop();
        test.push("..");
        test.push("test");
        test.push("tmp");
        let mut path = test.to_str().unwrap();

        let start = Instant::now();
        BatchSort::sort::<i32>(table.clone(), path);
        let duration = start.elapsed();
        println!("Time elapsed in sort() is: {:?}", duration);

        let start = Instant::now();
        let result: duplicat_result<i32> = BatchListFunctions_duplicates::count_duplicates_on_sorted_list::<i32>(table.clone(), true);
        let duration = start.elapsed();
        println!("Time elapsed in count_duplicates is: {:?}", duration);
        println!("counted_entries: {:?}", result.counted_entries);
        println!("counted_duplicates: {:?}", result.counted_duplicates);
        println!("entry_most_duplicates: {:?}", result.entry_most_duplicates);


        for x in 0..chunk_count {
            let arr: Array2<i32> = table.read().unwrap().load_chunk(x);
            //array_helper::print_array_2D(&arr);
            for (i, mut row) in arr.axis_iter(Axis(0)).enumerate() {
                for (j, col) in row.iter().enumerate() {
                    assert_eq!(test_vec[(x * height) + i][j], *col);
                }
            }
        }

        batchholder.clean()
    }

    #[test]
    fn test_random_inplace() {
        let mut rng = rand::thread_rng();

        for i in 0..1000{
            println!("Test {}", i);
            let height = rng.gen_range(1..20);
            let width = rng.gen_range(1..100);
            let chunk_count = rng.gen_range(1..30);

            let mut test = env::current_exe().unwrap();
            test.pop();
            test.pop();
            test.pop();
            test.pop();
            test.push("..");
            test.push("test");
            let mut path = test.to_str().unwrap();

            let mut batchholder: BatchHolder = BatchHolder::new(path);
            batchholder.create_batchtable(String::from("test"), vec![height, width], MemoryMode::InMemory);

            let mut table = batchholder.get_batchtable(String::from("test"));

            {
                let mut table = table.write().unwrap();
                for i in 0..chunk_count {
                    let ( mut arr, mut vec) = create_fingerprints_vector(width, height, true);
                    table.add_chunk( &arr);
                }
            }

            let mut test = env::current_exe().unwrap();
            test.pop();
            test.pop();
            test.pop();
            test.pop();
            test.push("..");
            test.push("test");
            test.push("tmp");
            let mut path = test.to_str().unwrap();

            let start = Instant::now();
            BatchSort::sort::<i32>(table.clone(), path);
            let duration = start.elapsed();
            println!("Time elapsed in sort() is: {:?}", duration);

            let start = Instant::now();
            let result: duplicat_result<i32> = BatchListFunctions_duplicates::count_duplicates_on_sorted_list::<i32>(table.clone(), true);
            let duration = start.elapsed();
            println!("Time elapsed in count_duplicates is: {:?}", duration);
            println!("counted_entries: {:?}", result.counted_entries);
            println!("counted_duplicates: {:?}", result.counted_duplicates);
            println!("entry_most_duplicates: {:?}", result.entry_most_duplicates);
            batchholder.clean()
        }
    }

    fn create_fingerprints_vector(width:usize, height:usize, with_mw_last_col:bool) -> (Array2<i32>, Vec<Vec<i32>>) {
        let mut rng = rand::thread_rng();
        let mut array: Array2<i32> = Array2::zeros((height, width));
        let mut vec = vec![vec![0;width]; height];

        let mut col_pos = 0;
        for (index_row, item) in array.iter_mut().enumerate() {
            if col_pos < width-1{
                let random_number_bit:i32 = rng.gen_range(0..1);
                *item = random_number_bit;
            }else{
                let random_number_mw:i32 = rng.gen_range(30..60);
                *item = random_number_mw;
                col_pos = 0;
                continue
            }
            col_pos += 1;
        }

        for (i, mut row) in array.axis_iter(Axis(0)).enumerate() {
            for (j, col) in row.iter().enumerate() {
                vec[i][j] = *col;
            }
        }
        return (array, vec);
    }
}