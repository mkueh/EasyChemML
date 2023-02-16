pub enum BatchDatatyp {
    NUMPY_STRING,
    PYTHON_OBJECT,
    NUMPY_INT8,
    NUMPY_INT16,
    NUMPY_INT32,
    NUMPY_INT64,
    NUMPY_FLOAT16,
    NUMPY_FLOAT32,
    NUMPY_FLOAT64,
    NUMPY_COMPLEX64,
    NUMPY_COMPLEX128,
    UNKOWN
}

pub fn dtyp_int_in_enum(dtype_int: i32) -> BatchDatatyp {
    match dtype_int {
        -1 => BatchDatatyp::NUMPY_STRING,
        -2 =>BatchDatatyp::PYTHON_OBJECT,
         1 => BatchDatatyp::NUMPY_INT8,
         2 => BatchDatatyp::NUMPY_INT16,
         3 => BatchDatatyp::NUMPY_FLOAT16,
         4 => BatchDatatyp::NUMPY_INT32,
         5 => BatchDatatyp::NUMPY_FLOAT32,
         6 => BatchDatatyp::NUMPY_INT64,
         7 => BatchDatatyp::NUMPY_FLOAT64,
         8 => BatchDatatyp::NUMPY_COMPLEX64,
         9 => BatchDatatyp::NUMPY_COMPLEX128,
        _ => BatchDatatyp::UNKOWN
    }
}

pub fn enum_2_dtyp_int(bd_typ: BatchDatatyp) -> i32{
    match bd_typ {
        BatchDatatyp::NUMPY_STRING => -1,
        BatchDatatyp::PYTHON_OBJECT => -2,
        BatchDatatyp::NUMPY_INT8 =>  1 ,
        BatchDatatyp::NUMPY_INT16 =>  2 ,
        BatchDatatyp::NUMPY_FLOAT16 =>  3,
        BatchDatatyp::NUMPY_INT32 =>  4 ,
        BatchDatatyp::NUMPY_FLOAT32 =>  5 ,
        BatchDatatyp::NUMPY_INT64 =>  6 ,
        BatchDatatyp::NUMPY_FLOAT64 => 7,
        BatchDatatyp::NUMPY_COMPLEX64 => 8,
        BatchDatatyp::NUMPY_COMPLEX128=>  9,
        _ => -100
    }
}