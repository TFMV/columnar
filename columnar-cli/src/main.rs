use std::path::PathBuf;
use std::{env, process};

use columnar_format::ColumnarReader;

fn inspect(path: PathBuf) {
    let mmap = columnar_mmap::MmapFile::open(&path).unwrap();
    let bytes = mmap.as_slice();
    let reader = ColumnarReader::new(bytes).unwrap();

    println!("--- Header ---");
    println!("{:?}", reader.header());

    println!("--- Column Directory ---");
    for i in 0..reader.column_count() {
        println!("Column {}: {:?}", i, reader.column_meta(i).unwrap());
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 || args[1] != "inspect" {
        println!("Usage: columnar-cli inspect <path>");
        process::exit(1);
    }
    let path = PathBuf::from(&args[2]);
    inspect(path);
}
