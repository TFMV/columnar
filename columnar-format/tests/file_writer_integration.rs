//! Write a `.columnar` file to disk and verify raw layout against format §2–6.

use columnar_format::{
    pad_length, ColumnDirectory, ColumnDirectoryView, ColumnMeta, ColumnarWriter, FileHeader,
    COLUMN_META_LEN, FILE_HEADER_LEN, SECTION_ALIGN, V0_PHYSICAL_FIXED_WIDTH_I64,
    VALUES_BUFFER_ALIGN,
};
use std::fs;
use std::path::PathBuf;

#[test]
fn write_file_read_raw_bytes_verify_layout() {
    let tmp = std::env::temp_dir();
    let path: PathBuf = tmp.join(format!("columnar_test_{}.columnar", std::process::id()));

    let schema = b"arrow-ipc-schema-placeholder";

    let mut w = ColumnarWriter::new();
    w.write_header_placeholder().unwrap();
    w.write_schema_block(schema).unwrap();
    let dir_offset = w.reserve_column_directory(1).unwrap();
    w.pad_to_alignment(VALUES_BUFFER_ALIGN).unwrap();

    let values: Vec<u8> = (0i64..5).flat_map(|v| v.to_le_bytes()).collect();
    let (data_off, data_len) = w.write_fixed_width_values(&values, 8).unwrap();

    let meta = ColumnMeta {
        column_id: 0,
        physical_type: V0_PHYSICAL_FIXED_WIDTH_I64,
        logical_type: 0,
        data_offset: data_off,
        data_length: data_len,
        validity_offset: 0,
        validity_length: 0,
        offsets_offset: 0,
        offsets_length: 0,
        stats_offset: 0,
        stats_length: 0,
    };
    w.patch_column_directory(std::slice::from_ref(&meta))
        .unwrap();
    w.finalize_header().unwrap();
    let bytes = w.into_inner();

    fs::write(&path, &bytes).expect("write temp file");
    let read_back = fs::read(&path).expect("read temp file");
    assert_eq!(read_back, bytes);
    let _ = fs::remove_file(&path);

    // --- Header ---
    assert_eq!(bytes.len() % SECTION_ALIGN, 0);
    let hdr = FileHeader::deserialize(&bytes[0..FILE_HEADER_LEN]).unwrap();
    hdr.validate().unwrap();
    assert_eq!(hdr.schema_offset, FILE_HEADER_LEN as u64);
    assert_eq!(hdr.schema_length as usize, schema.len());
    assert_eq!(hdr.column_dir_offset as usize, dir_offset);
    assert_eq!(hdr.column_dir_length as usize, COLUMN_META_LEN);

    // --- Schema payload (unpadded logical length) ---
    let s0 = hdr.schema_offset as usize;
    let s1 = s0 + hdr.schema_length as usize;
    assert_eq!(&bytes[s0..s1], schema);

    // --- 8-byte padding between schema end and directory ---
    let schema_end = s0 + schema.len();
    let expected_pad = pad_length(schema_end, SECTION_ALIGN);
    assert_eq!(dir_offset, schema_end + expected_pad);
    assert_eq!(dir_offset % SECTION_ALIGN, 0);

    // --- Directory ---
    let d0 = hdr.column_dir_offset as usize;
    let d1 = d0 + hdr.column_dir_length as usize;
    let dir = ColumnDirectory::deserialize(&bytes[d0..d1]).unwrap();
    assert_eq!(dir.len(), 1);
    assert_eq!(dir.as_slice()[0], meta);

    // --- Values: 64-byte aligned start, contiguous i64 LE ---
    assert_eq!(data_off as usize % VALUES_BUFFER_ALIGN, 0);
    let v0 = data_off as usize;
    let v1 = v0 + data_len as usize;
    assert_eq!(&bytes[v0..v1], values.as_slice());

    // --- Order: header < schema < directory < values ---
    assert!(FILE_HEADER_LEN <= s0);
    assert!(s1 <= d0);
    assert!(d1 <= v0);

    ColumnDirectoryView::new(&bytes[d0..d1])
        .unwrap()
        .validate(bytes.len() as u64)
        .unwrap();
}
