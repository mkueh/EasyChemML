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
    use crate::BatchSystem::BatchTablesImplementation::BatchTable::{BatchTable, BatchTablesTypWrapper};

    #[test]
    fn test_create_small_1Darray(){
        let (mut arr, mut vec) = create_fingerprints_vector(50,100);
        println!("-----------------");
        let start = Instant::now();
        BatchSort::sort_batch(&mut arr);
        let duration = start.elapsed();
        //array_helper::print_array_2D(&arr);
        println!("Time elapsed in expensive_function() is: {:?}", duration);
        vec.par_sort_unstable();

        for (i, mut row) in arr.axis_iter(Axis(0)).enumerate() {
            for (j, col) in row.iter().enumerate() {
                assert_eq!(vec[i][j], *col);
            }
        }
    }

    #[test]
    fn test_medium_array_to_InMemory(){
        let height = 10;
        let width = 10;
        let chunk_count = 10;

        let mut test = env::current_exe().unwrap();
        test.pop(); test.pop(); test.pop(); test.pop();
        test.push("test");
        test.push("tmp");
        let mut path = test.to_str().unwrap();

        let mut batchholder: BatchHolder = BatchHolder::new(path);
        batchholder.create_batchtable(String::from("test"), vec![height, width], MemoryMode::InMemory);

        let mut table = batchholder.get_batchtable(String::from("test"));
        let mut test_vec:Vec<Vec<i32>> = Vec::new();

        {
            let mut table = table.write().unwrap();
            for i in 0..chunk_count{
                let (mut arr, mut vec) = create_fingerprints_vector(width,height);
                test_vec.extend(vec);
                table.add_chunk(&arr);
            }
        }

        test_vec.par_sort_unstable();
        //println!("--------------------------");

        let mut test = env::current_exe().unwrap();
        test.pop(); test.pop(); test.pop(); test.pop();
        test.push("test");
        test.push("tmp_mem");
        let mut path = test.to_str().unwrap();

        let start = Instant::now();
        BatchSort::sort::<i32>(table.clone(), path);
        let duration = start.elapsed();
        println!("Time elapsed in expensive_function() is: {:?}", duration);

        println!("The following print is the sorted array from the batchsorter");
        {
            let table = table.read().unwrap();
            for i in 0..chunk_count{
               let arr:Array2<i32> = table.load_chunk(i);
               //array_helper::print_array_2D(&arr);
            }
        }

        println!("--------------------------");
        println!("The following print is the sorted array from the par_sort_unstable");
        println!("--------------------------");

        for x in 0..chunk_count{
            let arr:Array2<i32> = table.read().unwrap().load_chunk(x);
            for (i, mut row) in arr.axis_iter(Axis(0)).enumerate() {
                for (j, col) in row.iter().enumerate() {
                    assert_eq!(test_vec[(x*height)+i][j], *col);
                }
            }
        }

        batchholder.clean()
    }

       #[test]
    fn test_medium_array_to_InMemory_differentSizes(){
        let height = 10;
        let width = 10;
        let chunk_count = 10;

        let mut test = env::current_exe().unwrap();
        test.pop(); test.pop(); test.pop(); test.pop();
        test.push("test");
        test.push("tmp");
        let mut path = test.to_str().unwrap();

        let mut batchholder: BatchHolder = BatchHolder::new(path);
        batchholder.create_batchtable(String::from("test"), vec![height, width], MemoryMode::InMemory);

        let mut table = batchholder.get_batchtable(String::from("test"));
        let mut test_vec:Vec<Vec<i32>> = Vec::new();

        {
            let mut table = table.write().unwrap();

            let (mut arr, mut vec) = create_fingerprints_vector(width,8);
            test_vec.extend(vec);
            table.add_chunk(&arr);

            for i in 0..chunk_count{
                let (mut arr, mut vec) = create_fingerprints_vector(width,height);
                test_vec.extend(vec);
                table.add_chunk(&arr);
            }


        }

        test_vec.par_sort_unstable();
        //println!("--------------------------");

        let mut test = env::current_exe().unwrap();
        test.pop(); test.pop(); test.pop(); test.pop();
        test.push("test");
        test.push("tmp_mem");
        let mut path = test.to_str().unwrap();

        let start = Instant::now();
        BatchSort::sort::<i32>(table.clone(), path);
        let duration = start.elapsed();
        println!("Time elapsed in expensive_function() is: {:?}", duration);

        println!("The following print is the sorted array from the batchsorter");
        {
            let table = table.read().unwrap();
            for i in 0..chunk_count{
               let arr:Array2<i32> = table.load_chunk(i);
               //array_helper::print_array_2D(&arr);
            }
        }

        println!("--------------------------");
        println!("The following print is the sorted array from the par_sort_unstable");
        println!("--------------------------");

        for x in 0..chunk_count{
            let arr:Array2<i32> = table.read().unwrap().load_chunk(x);
            for (i, mut row) in arr.axis_iter(Axis(0)).enumerate() {
                for (j, col) in row.iter().enumerate() {
                    assert_eq!(test_vec[(x*height)+i][j], *col);
                }
            }
        }

        batchholder.clean()
    }

    #[test]
    fn test_medium_array_to_DirectIO(){
        let height = 10;
        let width = 10;
        let chunk_count = 5000;

        let mut test = env::current_exe().unwrap();
        test.pop(); test.pop(); test.pop(); test.pop();
        test.push("test");
        test.push("tmp");
        let mut path = test.to_str().unwrap();

        let mut batchholder: BatchHolder = BatchHolder::new(path);
        batchholder.create_batchtable(String::from("test"), vec![height, width], MemoryMode::DirectIo);

        let mut table = batchholder.get_batchtable(String::from("test"));
        let mut test_vec:Vec<Vec<i32>> = Vec::new();

        {
            let mut table = table.write().unwrap();
            for i in 0..chunk_count{
                let (mut arr, mut vec) = create_fingerprints_vector(width,height);
                test_vec.extend(vec);
                table.add_chunk(&arr);
            }
        }

        test_vec.par_sort_unstable();
        //println!("--------------------------");

        let mut test = env::current_exe().unwrap();
        test.pop(); test.pop(); test.pop(); test.pop();
        test.push("test");
        test.push("tmp_mem");
        let mut path = test.to_str().unwrap();

        let start = Instant::now();
        BatchSort::sort::<i32>(table.clone(), path);
        let duration = start.elapsed();
        println!("Time elapsed in expensive_function() is: {:?}", duration);

        println!("The following print is the sorted array from the batchsorter");
        for i in 0..chunk_count{
            let arr:Array2<i32> = table.read().unwrap().load_chunk(i);
            //array_helper::print_array_2D(&arr);
        }

        println!("--------------------------");
        println!("The following print is the sorted array from the par_sort_unstable");
        //array_helper::print_vec_2D(&test_vec);
        println!("--------------------------");

        for x in 0..chunk_count{
            let arr:Array2<i32> = table.read().unwrap().load_chunk(x);
            for (i, mut row) in arr.axis_iter(Axis(0)).enumerate() {
                for (j, col) in row.iter().enumerate() {
                    assert_eq!(test_vec[(x*height)+i][j], *col);
                }
            }
        }

        batchholder.clean()
    }

    fn create_fingerprints_vector(width:usize, height:usize) -> (Array2<i32>, Vec<Vec<i32>>) {
        let mut rng = rand::thread_rng();
        let mut array: Array2<i32> = Array2::zeros((height, width));
        let mut vec = vec![vec![0;width]; height];

        for (index_row, item) in array.iter_mut().enumerate() {
            let random_number:i32 = rng.gen_range(0..2);
            *item = random_number;
        }

        for (i, mut row) in array.axis_iter(Axis(0)).enumerate() {
            for (j, col) in row.iter().enumerate() {
                vec[i][j] = *col;
            }
        }
        return (array, vec);
    }
}