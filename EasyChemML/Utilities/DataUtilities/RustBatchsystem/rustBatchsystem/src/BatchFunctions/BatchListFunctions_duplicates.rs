use std::borrow::Borrow;
use std::collections::HashMap;
use std::fmt::Display;
use std::hash::Hash;
use std::sync::{Arc, RwLock};
use ndarray::{Array, Array2, ArrayView2, ArrayViewMut, Axis, Ix1, Ix2, IxDyn, s};
use ndarray_npy::{ReadableElement};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use crate::BatchSystem::BatchTablesImplementation::BatchTable::{BatchTable, BatchTablesTypWrapper};
use crate::Utilities::array_helper;

pub struct duplicat_result<T: ReadableElement + Ord + Clone + Hash + Display+ Copy> {
    pub counted_entries:usize,
    pub counted_duplicates:usize,
    pub entry_most_duplicates:usize,
    pub duplicates_dist: HashMap<usize, usize>,
    pub duplicates_by_last_col: HashMap<T, usize>
}

pub fn count_duplicates_on_sorted_list<'de,T: ReadableElement + Ord + Clone + Hash + Display+ Copy + DeserializeOwned>(batchtable: Arc<RwLock<BatchTablesTypWrapper>>, calc_duplicates_by_last_col:bool) -> duplicat_result<T> {
    let mut batchtable = batchtable.read().unwrap();
    let chunk_count = batchtable.get_table_chunk_count();

    let mut counted_entries:usize = 0;
    let mut counted_duplicates:usize = 0;
    let mut entry_most_duplicates:usize = 0;
    let mut duplicates_dist: HashMap<usize, usize> = HashMap::new();
    let mut duplicates_by_last_col: HashMap<T,usize> = HashMap::new();

    let mut current_entry_duplicated:usize = 0;

    let mut last_entry:Option<Array<T,Ix1>> = None;
    for chunk_index in 0..chunk_count{
        let loaded_chunk: Array<T, Ix2> = batchtable.load_chunk(chunk_index);


        for entry in loaded_chunk.axis_iter(Axis(0)){
            counted_entries += 1;

            if last_entry.is_none(){
                last_entry = Option::from(entry.to_owned());
            }else{
                let last_clone = last_entry.clone().unwrap();

                if array_helper::merge_compare_arrayView1(last_clone.view(), entry, calc_duplicates_by_last_col) == 0{
                    if current_entry_duplicated == 0{
                        counted_duplicates += 2;
                        current_entry_duplicated = 2;
                    }else {
                        counted_duplicates += 1;
                        current_entry_duplicated += 1;
                    }

                    if calc_duplicates_by_last_col{
                        if current_entry_duplicated == 2{

                            let val_last = last_clone.last().unwrap();
                            let val_current = entry.view();
                            let val_current = val_current.last().unwrap();

                            if duplicates_by_last_col.contains_key(val_last.borrow()){
                                let hash_pointer = duplicates_by_last_col.get_mut(val_last.borrow()).unwrap();
                                *hash_pointer += 1;
                            }else {
                                duplicates_by_last_col.insert(*val_last, 1);
                            }

                            if duplicates_by_last_col.contains_key(val_current.borrow()){
                                let hash_pointer = duplicates_by_last_col.get_mut(val_current.borrow()).unwrap();
                                *hash_pointer += 1;
                            }else {
                                duplicates_by_last_col.insert(*val_current, 1);
                            }
                        }else if current_entry_duplicated > 2 {
                            let val_current = entry.view();
                            let val_current = val_current.last().unwrap();

                            if duplicates_by_last_col.contains_key(val_current){
                                let hash_pointer = duplicates_by_last_col.get_mut(val_current).unwrap();
                                *hash_pointer += 1;
                            }else {
                                duplicates_by_last_col.insert(*val_current, 1);
                            }
                        }
                    }

                }else{

                    last_entry = Option::from(entry.to_owned());

                    if current_entry_duplicated >= 2{
                        if duplicates_dist.contains_key(&current_entry_duplicated){
                            *duplicates_dist.get_mut(&current_entry_duplicated).unwrap() += 1;
                        }else {
                            duplicates_dist.insert(current_entry_duplicated, 1);
                        }
                    }



                    if current_entry_duplicated > entry_most_duplicates{
                        entry_most_duplicates = current_entry_duplicated;
                    }

                    current_entry_duplicated = 0;
                }
            }
        }
    }

    if current_entry_duplicated >= 2{
        if duplicates_dist.contains_key(&current_entry_duplicated){
            *duplicates_dist.get_mut(&current_entry_duplicated).unwrap() += 1;
        }else {
            duplicates_dist.insert(current_entry_duplicated, 1);
        }
    }

    duplicat_result{counted_entries, counted_duplicates, entry_most_duplicates, duplicates_dist, duplicates_by_last_col }
}