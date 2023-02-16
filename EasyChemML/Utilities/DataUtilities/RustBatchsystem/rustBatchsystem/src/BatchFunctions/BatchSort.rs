use std::collections::HashMap;
use std::fmt::Display;
use std::hash::Hash;
use std::collections::VecDeque;
use std::time::Instant;
use std::marker::Send;
use kdam::{tqdm, BarExt};
use std::ops::DerefMut;
use std::sync::{Arc, RwLock, RwLockWriteGuard};

use ndarray::{Array, Array2, Axis, Dimension, Ix2, IxDyn, s};
use ndarray_npy::{ReadableElement};

use crate::BatchSystem::BatchTablesImplementation::BatchTable::{BatchTable, BatchTableMemoryBased, BatchTablePathBased, BatchTablesTypWrapper};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde::de::DeserializeOwned;
use crate::BatchSystem::BatchHolder::{BatchHolder};
use crate::Utilities::array_helper;

pub fn sort<'de,T: Ord + Clone + Hash + Display + Copy + Send + Serialize + DeserializeOwned + ReadableElement>(batchtable: Arc<RwLock<BatchTablesTypWrapper>>, tmp_path:&str){
    let mut batchtable = batchtable.write().unwrap();
    //sort batches
    println!("Sort batch files");
    let mut bar = tqdm!(total = batchtable.get_table_chunk_count());
    let start = Instant::now();
    for index in 0..batchtable.get_table_chunk_count(){
        bar.update(1);
        let arr:&mut Array<T,Ix2> =  &mut batchtable.load_chunk(index);
        sort_batch(arr);
        batchtable.override_chunk_with_array(arr, index);
    }
    let duration = start.elapsed();
    println!("Rust batch sorting takes : {:?}", duration);
    let start = Instant::now();

    //merge batches
    let batch_count = batchtable.get_table_chunk_count();
    let merge_slices: VecDeque<((usize, usize), (usize, usize))> = merge_generate_merge_slices(batch_count);

    let mut tmp_batchholder: BatchHolder = BatchHolder::new(tmp_path);

    println!("Merge sorted files");
    let mut bar = tqdm!(total = merge_slices.len());
    for slice in merge_slices {
        bar.update(1);
        let (slice_left, slice_right) = slice;
        let (slice_left_start, slice_left_end) = slice_left;
        let (slice_right_start, slice_right_end) = slice_right;
        merge_slice::<T>(batchtable.deref_mut(), &mut tmp_batchholder, slice_left, slice_right);

        if slice_left_start == slice_right_start && slice_left_end == slice_right_end {
            continue;
        }

        {
            let current_tmp_table = tmp_batchholder.get_batchtable(batchtable.get_table_name() + "_sort");
            let mut current_tmp_table = current_tmp_table.write().unwrap();
            for (tmp_batchtable_index, slice_index) in (slice_left_start..slice_right_end + 1).enumerate() {
                if batchtable.is_batch_table_pathbased() && current_tmp_table.is_batch_table_pathbased() {
                    let mut cast_currentTMPTable = current_tmp_table.to_batch_table_file_system().unwrap();
                    let cast_batchtable = batchtable.to_batch_table_file_system().unwrap();

                    let tmpTable_path = cast_currentTMPTable.get_chunk_path(tmp_batchtable_index);
                    let slice_tmp_path = tmpTable_path.to_str().unwrap();
                    cast_batchtable.override_chunk_with_file(slice_tmp_path, slice_index);
                } else if batchtable.is_batch_table_memorybased() && current_tmp_table.is_batch_table_memorybased() {
                    let cast_currentTMPTable = current_tmp_table.to_batch_table_in_memory().unwrap();
                    let cast_batchtable = batchtable.to_batch_table_in_memory().unwrap();

                    let vec_pointer = cast_currentTMPTable.get_raw_data_pointer(tmp_batchtable_index);
                    cast_batchtable.override_chunk_with_pointer(vec_pointer, slice_index);
                } else {
                    let arr: &mut Array<T, Ix2> = &mut current_tmp_table.load_chunk(tmp_batchtable_index);
                    batchtable.override_chunk_with_array(arr, slice_index)
                }
            }
        }
        tmp_batchholder.remove_batchtable(batchtable.get_table_name() + "_sort");
    }
    let duration = start.elapsed();
    println!("Rust batch merging takes : {:?}", duration);
    tmp_batchholder.clean();

}

fn merge_slice<'de,T: ReadableElement + Ord + Clone + Hash + Display+ Copy+ Serialize+ DeserializeOwned>(mut batchtable: &mut BatchTablesTypWrapper, tmp_batchholder: &mut BatchHolder, slice_one: (usize, usize), slice_two: (usize, usize)){
    let (slice_one_start, slice_one_end) = slice_one;
    let (slice_two_start, slice_two_end) = slice_two;

    if slice_one_start == slice_two_start && slice_one_end == slice_two_end{
        return;
    }

    tmp_batchholder.create_batchtable(batchtable.get_table_name() + "_sort", batchtable.get_shape(), batchtable.get_memory_mode());
    let mut tmp_batchtable = tmp_batchholder.get_batchtable(batchtable.get_table_name() + "_sort");
    let mut tmp_batchtable = tmp_batchtable.write().unwrap();

    let mut left_arr: Array<T, Ix2> = batchtable.load_chunk(slice_one_start);
    let mut right_arr: Array<T, Ix2> = batchtable.load_chunk(slice_two_start);

    let mut tmp_arr: Array<T, Ix2> = Array::clone(&left_arr);

    let mut left_arr_rows_count:usize = left_arr.nrows();
    let mut right_arr_rows_count:usize = right_arr.nrows();
    let tmp_arr_rows_count:usize = tmp_arr.nrows();

    let mut current_slice_left = slice_one_start;
    let mut current_slice_right = slice_two_start;

    let mut current_tmp_vector_ptr: usize = 0;
    let mut current_right_arr_prt: usize = 0;
    let mut current_left_arr_prt: usize = 0;

    loop {

        if current_tmp_vector_ptr >= tmp_arr_rows_count{
            tmp_batchtable.add_chunk(&tmp_arr);
            current_tmp_vector_ptr = 0;
        }

        if current_left_arr_prt >= left_arr_rows_count{
            current_slice_left += 1;
            if current_slice_left <= slice_one_end{
                left_arr = batchtable.load_chunk(current_slice_left);
                left_arr_rows_count = left_arr.nrows();
                current_left_arr_prt = 0;
            }else{
                merge_copy_rest_of_array(&mut tmp_arr, &mut current_tmp_vector_ptr, batchtable, tmp_batchtable.deref_mut(), current_slice_right, slice_two_end, &mut current_right_arr_prt);
                break;
            }

        }

        if current_right_arr_prt >= right_arr_rows_count{
            current_slice_right += 1;
            if current_slice_right <= slice_two_end{
                right_arr = batchtable.load_chunk(current_slice_right);
                right_arr_rows_count = right_arr.nrows();
                current_right_arr_prt = 0;
            }else{
                merge_copy_rest_of_array(&mut tmp_arr, &mut current_tmp_vector_ptr, batchtable, tmp_batchtable.deref_mut(), current_slice_left, slice_one_end, &mut current_left_arr_prt);
                break;
            }
        }

        if current_slice_left >= slice_one_end && current_slice_right >= slice_two_end && current_left_arr_prt >= left_arr_rows_count && current_right_arr_prt >= right_arr_rows_count {
            break;
        }

        let compare_result = array_helper::merge_compare_arrayView1(left_arr.row(current_left_arr_prt), right_arr.row(current_right_arr_prt), false);

        if compare_result >= 0{
            array_helper::merge_arrayView1_copy(&mut tmp_arr.row_mut(current_tmp_vector_ptr), &left_arr.row(current_left_arr_prt));
            current_tmp_vector_ptr += 1;
            current_left_arr_prt += 1;
        }else {
            array_helper::merge_arrayView1_copy(&mut tmp_arr.row_mut(current_tmp_vector_ptr), &right_arr.row(current_right_arr_prt));
            current_tmp_vector_ptr += 1;
            current_right_arr_prt += 1;
        }
    }

    tmp_batchtable.add_chunk(&tmp_arr.slice(s!(0..current_tmp_vector_ptr, ..)).to_owned());
}

fn merge_copy_rest_of_array<'de,T: ReadableElement + Ord + Clone + Hash + Display + Copy + Serialize+ DeserializeOwned>
                (tmp_array: &mut Array2<T>, mut current_tmp_vector_ptr:&mut usize, mut batchtable: &mut BatchTablesTypWrapper, tmp_batchtable: &mut BatchTablesTypWrapper, start_slice: usize, end_slice:usize, mut current_arr_ptr: &mut usize){

    for i_slice in start_slice .. end_slice+1{
        let rest_array:Array2<T> = batchtable.load_chunk(i_slice);
        for current_ptr in *current_arr_ptr .. rest_array.nrows(){
            if *current_tmp_vector_ptr >= tmp_array.nrows(){
                tmp_batchtable.add_chunk(&tmp_array);
                *current_tmp_vector_ptr = 0;
            }

            array_helper::merge_arrayView1_copy(&mut tmp_array.row_mut(*current_tmp_vector_ptr), &rest_array.row(current_ptr));
            *current_tmp_vector_ptr += 1;
        }
        *current_arr_ptr = 0;
    }
    *current_arr_ptr = tmp_array.nrows()+1;
}

fn merge_generate_merge_slices(batch_count:usize) -> VecDeque<((usize, usize), (usize, usize))> {
    let mut merges: VecDeque<((usize,usize),(usize,usize))> = VecDeque::new();
    let mut last_merges: VecDeque<((usize,usize),(usize,usize))> = VecDeque::new();
    for batch_indexin in (0..batch_count).step_by(2){
        let first_entry = (batch_indexin, batch_indexin);
        let mut second_entry = (batch_indexin+1, batch_indexin+1);

        if batch_indexin+1 >= batch_count {
            second_entry = (batch_indexin, batch_indexin);
        }

        merges.push_back((first_entry,second_entry));
        last_merges.push_back((first_entry,second_entry));
    }

    loop {
        let last_merge_len = last_merges.len();
        let clone_last_merge = last_merges.clone();
        last_merges.clear();
        for (i, (first_entry, second_entry)) in clone_last_merge.
            iter().step_by(2 as usize).enumerate(){

            let (first_left, first_right) = first_entry;
            let (second_left, second_right) = second_entry;

            if (i*2)+1 >= last_merge_len{
                let new_first = (*first_left,*second_right);
                let new_second = (*first_left, *second_right);

                merges.push_back((new_first, new_second));
                last_merges.push_back((new_first, new_second));
            }else{
                let ((next_first_left, next_first_right), (next_second_left, next_second_right)) = clone_last_merge[(i*2)+1];
                let new_first = (*first_left,*second_right);
                let new_second = (next_first_left, next_second_right);

                merges.push_back((new_first, new_second));
                last_merges.push_back((new_first, new_second));
            }

        }

        if last_merges.len() <= 1{
            return merges;
        }
    }
}

pub fn sort_batch<'de,T: ReadableElement + Ord + Clone + Hash + Display + Copy + Send + DeserializeOwned>(mut unsorted_array: &mut Array2<T>){
    let input_shape = unsorted_array.shape().to_owned();
    let mut partitions:VecDeque<(usize, usize,usize)> = VecDeque::new();

    partitions.push_back((0, 0, input_shape[0]));
    while !partitions.is_empty(){
        let (current_deep,start_index,end_index) = partitions.pop_front().unwrap();

        let buckets = sort_batch_count_buckets(unsorted_array,&current_deep, start_index, end_index);

        let mut v: Vec<_> = buckets.keys().cloned().collect();
        v.par_sort_by(|x,y| x.partial_cmp(&y).unwrap());

        let mut buckets_startpoint = sort_batch_start_pointer(buckets, v, start_index);

        let mut current_pos:usize = start_index;
        while current_pos < end_index {
            let current_val = unsorted_array.get((current_pos.clone(), current_deep)).unwrap();
            let (start_position, new_position, end_position) = buckets_startpoint.get_mut(current_val).unwrap();

            if current_pos < *end_position && current_pos >= *start_position{
                current_pos += 1;
                *new_position+= 1;
            }else {
                sort_batch_swap(unsorted_array, current_pos, *new_position);
                *new_position+= 1;
            }
        }


        for (k,(start,_,end)) in buckets_startpoint{
            if start.abs_diff(end) > 1 && current_deep+1 < input_shape[1] {
                partitions.push_back((current_deep+1,start,end));
            }
        }

        //println!("-----------------");
        //array_helper::print_array_2D(&unsorted_array);
        //println!("-----------------");
    }
}

fn sort_batch_swap<T: Clone + Copy>(mut unsorted_array: &mut Array2<T>, index_a: usize, index_b:usize){
    let tmp_a = unsorted_array.row(index_a).to_owned();

    for i in 0..tmp_a.shape()[0]{
        unsorted_array[(index_a, i)] = unsorted_array[(index_b, i)];
        unsorted_array[(index_b, i)] = tmp_a[i];
    }

}

fn sort_batch_count_buckets<T: ReadableElement + Ord + Clone + Hash>(unsorted_array: &Array2<T>,
                                                                                         col_index: &usize, start:usize, end:usize) -> HashMap<T, usize>{
    let mut bucket_map:HashMap<T, usize> = HashMap::new();
    for (current_pos, item) in unsorted_array.axis_iter(Axis(0)).enumerate().skip(start){

        if current_pos >= end{
            return bucket_map
        }

        if !bucket_map.contains_key(&item[*col_index]){
            bucket_map.insert(item[*col_index].clone(),1);
        }else {
            let entry = bucket_map.get_mut(&item[*col_index]).unwrap();
            *entry += 1;
        }
    }
    bucket_map
}

fn sort_batch_start_pointer<T: Ord + Hash + Clone>(buckets: HashMap<T, usize>, keys: Vec<T>, start:usize) -> HashMap<T, (usize, usize, usize)> {
    let mut start_pointer_map:HashMap<T, (usize,usize, usize)> = HashMap::new();
    let mut current_pos: usize = start;

    for key in keys.into_iter() {
        start_pointer_map.insert(key.clone(), (current_pos.clone(), current_pos.clone(), current_pos+buckets.get(&key).unwrap()));
        current_pos += buckets.get(&key).unwrap();
    }
    start_pointer_map
}