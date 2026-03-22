use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::{env, process};

use columnar_format::{ColumnarReader, FileHeader, FILE_HEADER_LEN};

fn inspect(path: PathBuf) {
    let mut f = File::open(&path).unwrap();
    let mut buffer = [0u8; FILE_HEADER_LEN];
    f.read_exact(&mut buffer).unwrap();
    let header = FileHeader::deserialize(&buffer).unwrap();
    header.validate().unwrap();

    println!("--- Header ---");
    println!("{:?}", header);

    let mmap = columnar_mmap::MmapFile::open(&path).unwrap();
    let reader = ColumnarReader::new(mmap.as_slice()).unwrap();

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
