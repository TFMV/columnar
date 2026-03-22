use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow_array::{Array, Int64Array, StringArray};
use columnar_arrow::{build_int64_array, build_utf8_array};
use columnar_format::{ColumnarReader, ColumnarWriter, Int64Stats};
use columnar_mmap::MmapFile;

enum TestColumn<'a> {
    Int64(&'a Int64Array),
    Utf8(&'a StringArray),
}

static NEXT_TEST_ID: AtomicUsize = AtomicUsize::new(0);

fn unique_temp_path(label: &str) -> PathBuf {
    let unique = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "{label}_{}_{}.columnar",
        std::process::id(),
        unique
    ))
}

fn encode_int64_values(array: &Int64Array) -> Vec<u8> {
    let mut out = Vec::with_capacity(array.len() * std::mem::size_of::<i64>());
    for value in array.values().inner().as_slice() {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn encode_utf8_offsets(array: &StringArray) -> Vec<u8> {
    let mut out = Vec::with_capacity(std::mem::size_of_val(array.value_offsets()));
    for offset in array.value_offsets() {
        out.extend_from_slice(&offset.to_le_bytes());
    }
    out
}

fn encode_validity(array: &dyn Array) -> Option<Vec<u8>> {
    let nulls = array.nulls()?;
    let needed_bytes = array.len().div_ceil(8);
    Some(nulls.buffer().as_slice()[..needed_bytes].to_vec())
}

fn int64_stats(array: &Int64Array) -> Int64Stats {
    let mut min: Option<i64> = None;
    let mut max: Option<i64> = None;
    let mut distinct = std::collections::BTreeSet::new();
    let mut null_count = 0u64;

    for value in array.iter() {
        match value {
            Some(value) => {
                min = Some(min.map_or(value, |current| current.min(value)));
                max = Some(max.map_or(value, |current| current.max(value)));
                distinct.insert(value);
            }
            None => null_count += 1,
        }
    }

    Int64Stats {
        min,
        max,
        null_count,
        distinct_count: Some(distinct.len() as u64),
    }
}

fn write_chunks(chunks: &[Vec<TestColumn<'_>>]) -> Vec<u8> {
    assert!(!chunks.is_empty());
    let column_count = chunks[0].len();
    assert!(column_count > 0);
    for chunk in chunks {
        assert_eq!(chunk.len(), column_count);
    }

    let mut writer = ColumnarWriter::new();
    writer.write_header_placeholder().expect("header");
    writer
        .write_schema_block(b"arrow-roundtrip")
        .expect("schema");
    writer
        .reserve_column_directory(chunks.len() * column_count)
        .expect("reserve directory");

    let mut metas = Vec::with_capacity(chunks.len() * column_count);
    for chunk in chunks {
        for (column_id, column) in chunk.iter().enumerate() {
            let meta = match column {
                TestColumn::Int64(array) => writer
                    .write_int64_column_chunk(
                        column_id as u32,
                        0,
                        &encode_int64_values(array),
                        encode_validity(*array).as_deref(),
                        Some(int64_stats(array)),
                    )
                    .expect("write int64 chunk"),
                TestColumn::Utf8(array) => writer
                    .write_utf8_i32_column_chunk(
                        column_id as u32,
                        0,
                        &encode_utf8_offsets(array),
                        array.value_data(),
                        encode_validity(*array).as_deref(),
                        None,
                    )
                    .expect("write utf8 chunk"),
            };
            metas.push(meta);
        }
    }

    writer
        .patch_column_directory(&metas)
        .expect("patch directory");
    writer.finalize_header().expect("finalize header");
    writer.into_inner()
}

fn assert_int64_round_trip(
    reader: &ColumnarReader<'_>,
    mmap: &MmapFile,
    chunk_index: usize,
    column_index: usize,
    expected: &Int64Array,
) {
    let values = reader
        .chunk_column_values(chunk_index, column_index)
        .expect("values");
    assert_eq!(values, encode_int64_values(expected).as_slice());

    let validity = reader
        .chunk_column_validity(chunk_index, column_index)
        .expect("validity");
    assert_eq!(
        validity.map(|bytes| bytes.to_vec()),
        encode_validity(expected)
    );

    let round_tripped = build_int64_array(mmap.mmap_arc(), values, validity).expect("array");
    assert_eq!(round_tripped.len(), expected.len());
    for index in 0..expected.len() {
        assert_eq!(round_tripped.is_null(index), expected.is_null(index));
        if expected.is_valid(index) {
            assert_eq!(round_tripped.value(index), expected.value(index));
        }
    }
}

fn assert_utf8_round_trip(
    reader: &ColumnarReader<'_>,
    mmap: &MmapFile,
    chunk_index: usize,
    column_index: usize,
    expected: &StringArray,
) {
    let variable = reader
        .chunk_variable_column_buffers(chunk_index, column_index)
        .expect("variable buffers");
    assert_eq!(variable.offsets, encode_utf8_offsets(expected).as_slice());
    assert_eq!(variable.values, expected.value_data());
    assert_eq!(
        variable.validity.map(|bytes| bytes.to_vec()),
        encode_validity(expected)
    );

    let round_tripped = build_utf8_array(
        mmap.mmap_arc(),
        variable.offsets,
        variable.values,
        variable.validity,
    )
    .expect("utf8 array");
    assert_eq!(round_tripped.len(), expected.len());
    for index in 0..expected.len() {
        assert_eq!(round_tripped.is_null(index), expected.is_null(index));
        if expected.is_valid(index) {
            assert_eq!(round_tripped.value(index), expected.value(index));
        }
    }
}

#[test]
fn round_trip_null_heavy_int64_column_preserves_bytes() {
    let array = Int64Array::from(vec![
        None,
        Some(11),
        None,
        None,
        Some(-7),
        None,
        Some(0),
        None,
    ]);

    let bytes = write_chunks(&[vec![TestColumn::Int64(&array)]]);
    let path = unique_temp_path("columnar_roundtrip_null_heavy");
    fs::write(&path, &bytes).expect("write file");

    let mmap = MmapFile::open(&path).expect("mmap");
    let reader = ColumnarReader::new(mmap.as_slice()).expect("reader");
    assert_int64_round_trip(&reader, &mmap, 0, 0, &array);

    let _ = fs::remove_file(&path);
}

#[test]
fn round_trip_mixed_type_chunks_preserve_buffer_bytes() {
    let int_chunk0 = Int64Array::from(vec![Some(1), Some(2), None, Some(4)]);
    let str_chunk0 = StringArray::from(vec![Some("alpha"), None, Some(""), Some("beta")]);
    let int_chunk1 = Int64Array::from(vec![None, Some(8), Some(13)]);
    let str_chunk1 = StringArray::from(vec![Some("omega"), Some("delta"), None]);

    let bytes = write_chunks(&[
        vec![
            TestColumn::Int64(&int_chunk0),
            TestColumn::Utf8(&str_chunk0),
        ],
        vec![
            TestColumn::Int64(&int_chunk1),
            TestColumn::Utf8(&str_chunk1),
        ],
    ]);
    let path = unique_temp_path("columnar_roundtrip_mixed");
    fs::write(&path, &bytes).expect("write file");

    let mmap = MmapFile::open(&path).expect("mmap");
    let reader = ColumnarReader::new(mmap.as_slice()).expect("reader");
    assert_eq!(reader.chunk_count(), 2);
    assert_eq!(reader.column_count(), 2);
    assert_int64_round_trip(&reader, &mmap, 0, 0, &int_chunk0);
    assert_utf8_round_trip(&reader, &mmap, 0, 1, &str_chunk0);
    assert_int64_round_trip(&reader, &mmap, 1, 0, &int_chunk1);
    assert_utf8_round_trip(&reader, &mmap, 1, 1, &str_chunk1);

    let _ = fs::remove_file(&path);
}

#[test]
fn round_trip_large_multi_chunk_dataset_preserves_counts_and_bytes() {
    let rows_per_chunk = 4_096usize;
    let chunk_count = 3usize;
    let mut int_chunks = Vec::with_capacity(chunk_count);
    let mut utf8_chunks = Vec::with_capacity(chunk_count);

    for chunk_index in 0..chunk_count {
        let base = (chunk_index * rows_per_chunk) as i64;
        let int_values = (0..rows_per_chunk)
            .map(|row| {
                if row % 5 == 0 {
                    None
                } else {
                    Some(base + row as i64)
                }
            })
            .collect::<Vec<_>>();
        let utf8_values = (0..rows_per_chunk)
            .map(|row| {
                if row % 7 == 0 {
                    None
                } else if row % 11 == 0 {
                    Some(String::new())
                } else {
                    Some(format!("chunk{chunk_index}-row{row}"))
                }
            })
            .collect::<Vec<_>>();

        int_chunks.push(Int64Array::from(int_values));
        utf8_chunks.push(StringArray::from(utf8_values));
    }

    let chunks = (0..chunk_count)
        .map(|index| {
            vec![
                TestColumn::Int64(&int_chunks[index]),
                TestColumn::Utf8(&utf8_chunks[index]),
            ]
        })
        .collect::<Vec<_>>();

    let bytes = write_chunks(&chunks);
    let path = unique_temp_path("columnar_roundtrip_large");
    fs::write(&path, &bytes).expect("write file");

    let mmap = MmapFile::open(&path).expect("mmap");
    let reader = ColumnarReader::new(mmap.as_slice()).expect("reader");
    assert_eq!(reader.chunk_count(), chunk_count);
    assert_eq!(reader.column_count(), 2);

    let mut total_rows = 0usize;
    for chunk_index in 0..chunk_count {
        total_rows += reader.chunk_row_count(chunk_index).expect("row count");
        assert_int64_round_trip(&reader, &mmap, chunk_index, 0, &int_chunks[chunk_index]);
        assert_utf8_round_trip(&reader, &mmap, chunk_index, 1, &utf8_chunks[chunk_index]);
    }
    assert_eq!(total_rows, rows_per_chunk * chunk_count);

    let _ = fs::remove_file(&path);
}
