use crate::impl_evo_fp;
use ndarray::{Array1, ArrayBase, Ix1, OwnedRepr};
use rdkit::ROMol;
use std::fs::File;

pub fn dreher_doyle() -> (Vec<Vec<ROMol>>, ArrayBase<OwnedRepr<f64>, Ix1>) {
    let file_path =
        "/home/student/aebberg/rusty-evo-fp/src/chem_datasets/Dreher_and_Doyle_input_data.csv";
    let file = File::open(file_path).unwrap();
    let mut rdr = csv::ReaderBuilder::new().delimiter(b';').from_reader(&file);

    let target_data: Vec<f64> = rdr
        .records()
        .map(|r| {
            r.unwrap()[4]
                .trim()
                .replace(',', ".")
                .parse::<f64>()
                .unwrap()
        })
        .collect::<Vec<_>>();

    let file_two = File::open(file_path).unwrap();
    let mut rdr_two = csv::ReaderBuilder::new()
        .delimiter(b';')
        .from_reader(file_two);

    let data = rdr_two
        .records()
        .map(|r| {
            let mut rec = r.unwrap();
            rec.truncate(4);
            rec.iter()
                .map(|entry| entry.to_string())
                .collect::<Vec<String>>()
        })
        .collect::<Vec<_>>();

    let array_target_data = Array1::from_shape_vec(target_data.len(), target_data).unwrap();
    let mol_data = impl_evo_fp::get_data::bench_convert_data_to_mol(&data);

    (mol_data, array_target_data)
}

pub fn lipophilicity() -> (Vec<Vec<ROMol>>, ArrayBase<OwnedRepr<f64>, Ix1>) {
    let file_path = "/home/student/aebberg/rusty-evo-fp/src/chem_datasets/Lipophilicity.csv";
    let file = File::open(file_path).unwrap();
    let mut rdr = csv::ReaderBuilder::new().delimiter(b';').from_reader(&file);

    let target_data = rdr
        .records()
        .map(|r| {
            r.unwrap()[2]
                .trim()
                .replace(',', ".")
                .parse::<f64>()
                .unwrap()
        })
        .collect::<Vec<_>>();
    let file_two = File::open(file_path).unwrap();
    let mut rdrtwo = csv::ReaderBuilder::new()
        .delimiter(b';')
        .from_reader(file_two);
    let data = rdrtwo
        .records()
        .map(|r| vec![r.unwrap()[1].to_string()])
        .collect::<Vec<_>>();

    let array_target_data = Array1::from_shape_vec(target_data.len(), target_data).unwrap();
    let mol_data = impl_evo_fp::get_data::bench_convert_data_to_mol(&data);

    (mol_data, array_target_data)
}

pub fn hiv() -> (Vec<Vec<ROMol>>, ArrayBase<OwnedRepr<usize>, Ix1>) {
    let file_path = "/home/student/aebberg/rusty-evo-fp/src/chem_datasets/HIV_classify.csv";
    let file = File::open(file_path).unwrap();
    let mut rdr = csv::ReaderBuilder::new().delimiter(b';').from_reader(&file);
    println!("{:?}", rdr.headers());

    let target_data = rdr
        .records()
        .map(|r| {
            r.unwrap()[1]
                .trim()
                .replace(',', ".")
                .parse::<usize>()
                .unwrap()
        })
        .collect::<Vec<_>>();

    let file_two = File::open(file_path).unwrap();
    let mut rdrtwo = csv::ReaderBuilder::new()
        .delimiter(b';')
        .from_reader(file_two);
    let data = rdrtwo
        .records()
        .map(|r| vec![r.unwrap()[0].to_string()])
        .collect::<Vec<_>>();

    let array_target_data = Array1::from_shape_vec(target_data.len(), target_data).unwrap();
    let mol_data = impl_evo_fp::get_data::bench_convert_data_to_mol(&data);

    (mol_data, array_target_data)
}

pub fn hiv_8000() -> (Vec<Vec<ROMol>>, ArrayBase<OwnedRepr<usize>, Ix1>) {
    let file_path = "/home/student/aebberg/rusty-evo-fp/src/chem_datasets/HIV_classify_8000.csv";
    let file = File::open(file_path).unwrap();
    let mut rdr = csv::ReaderBuilder::new().delimiter(b';').from_reader(&file);
    println!("{:?}", rdr.headers());

    let target_data = rdr
        .records()
        .map(|r| {
            r.unwrap()[1]
                .trim()
                .replace(',', ".")
                .parse::<usize>()
                .unwrap()
        })
        .collect::<Vec<_>>();

    let file_two = File::open(file_path).unwrap();
    let mut rdrtwo = csv::ReaderBuilder::new()
        .delimiter(b';')
        .from_reader(file_two);
    let data = rdrtwo
        .records()
        .map(|r| vec![r.unwrap()[0].to_string()])
        .collect::<Vec<_>>();

    let array_target_data = Array1::from_shape_vec(target_data.len(), target_data).unwrap();
    let mol_data = impl_evo_fp::get_data::bench_convert_data_to_mol(&data);

    (mol_data, array_target_data)
}

#[cfg(test)]
mod tests {
    use crate::chem_datasets::csv_import::hiv;

    #[test]
    fn test_dreher_doyle() {
        hiv();
    }
}
