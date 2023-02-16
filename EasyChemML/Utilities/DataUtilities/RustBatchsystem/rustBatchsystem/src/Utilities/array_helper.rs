use std::cmp::Ordering;
use std::fmt::Display;
use ndarray::{Array2, ArrayView, ArrayView1, ArrayViewMut, ArrayViewMut1, IxDyn};


pub fn print_vec_2D<A: Display>(arr:&Vec<Vec<A>>){
    for row in arr.iter() {
        print!("[");
        for (index_col, col) in row.iter().enumerate(){
            if index_col == row.len(){
                print!("{}", col);
            }else {
                print!("{} ", col);
            }
        }
        print!("] \n");
    }
}

pub fn print_array_2D<A: Display>(arr:&Array2<A>){
    for row in arr.rows(){
        print!("[");
        for (index_col, col) in row.iter().enumerate(){
            if index_col == row.len(){
                print!("{}", col);
            }else {
                print!("{} ", col);
            }
        }
        print!("] \n");
    }
}

pub fn print_vec_1D<A: Display>(arr:Vec<A>){
    print!("[");
    for (index_row, row) in arr.iter().enumerate(){
        if index_row == arr.len(){
            print!("{}", row);
        }else {
            print!("{} ", row);
        }
    }
    print!("] \n");
}

pub(crate) fn merge_arrayView1_copy<T: Clone>(output_arr: &mut ArrayViewMut1<T>, input_view: &ArrayView1<T>){
    for (index, val) in input_view.iter().enumerate(){
        output_arr[index] = val.clone();
    }
}

pub(crate) fn merge_compare_arrayView1<T: PartialOrd>(left_arr: ArrayView1<T>, right_arr: ArrayView1<T>, ignore_last_col:bool) -> i32 {
    for (index,left_val) in left_arr.iter().enumerate(){

        if ignore_last_col && index == left_arr.len()-1{
            return 0;
        }

        let right_val = &right_arr[index];
        match left_val.partial_cmp(right_val).unwrap() {
            Ordering::Less => {return 1}
            Ordering::Equal => {continue}
            Ordering::Greater => {return -1}
        }
    }
    0

}
