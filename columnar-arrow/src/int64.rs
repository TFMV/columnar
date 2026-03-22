use arrow_array::Int64Array;
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer};
use arrow_data::ArrayData;
use arrow_schema::DataType;
use memmap2::Mmap;
use std::sync::Arc;

use crate::buffer::{ArrowBuildError, MmapBuffer};
use columnar_format::MIN_BUFFER_ALIGN;

fn make_values_buffer(mmap: Arc<Mmap>, values: &[u8]) -> Result<(Buffer, usize), ArrowBuildError> {
    if values.len() % 8 != 0 {
        return Err(ArrowBuildError::InvalidValuesLength { got: values.len() });
    }
    let len = values.len() / 8;
    let ptr = values.as_ptr() as usize;
    let alignment = std::mem::align_of::<i64>();
    if ptr % alignment != 0 {
        return Err(ArrowBuildError::ValuesMisaligned { ptr, alignment });
    }
    let buf = MmapBuffer::try_new(mmap, values)?.into_arrow_buffer();
    Ok((buf, len))
}

fn make_nulls(
    mmap: Arc<Mmap>,
    validity: Option<&[u8]>,
    len: usize,
) -> Result<Option<NullBuffer>, ArrowBuildError> {
    let Some(validity) = validity else {
        return Ok(None);
    };
    if len == 0 {
        return Ok(None);
    }
    let needed_bytes = (len + 7) / 8;
    if validity.len() < needed_bytes {
        return Err(ArrowBuildError::ValidityTooShort {
            got_bytes: validity.len(),
            needed_bytes,
        });
    }
    let validity_ptr = validity.as_ptr() as usize;
    let alignment = MIN_BUFFER_ALIGN as usize;
    if validity.len() > 0 && validity_ptr % alignment != 0 {
        return Err(ArrowBuildError::ValidityMisaligned {
            ptr: validity_ptr,
            alignment,
        });
    }
    let validity_buffer = MmapBuffer::try_new(mmap, validity)?.into_arrow_buffer();
    let boolean_buffer = BooleanBuffer::new(validity_buffer, 0, len);
    Ok(Some(NullBuffer::new(boolean_buffer)))
}

/// Build an Arrow `ArrayData` for an `Int64` column using zero-copy Arrow buffers.
///
/// The returned buffers retain an internal `Arc<Mmap>` so the array remains valid even if the
/// caller drops its `MmapFile`.
pub fn int64_array_data_from_slices(
    mmap: Arc<Mmap>,
    values: &[u8],
    validity: Option<&[u8]>,
) -> Result<ArrayData, ArrowBuildError> {
    let (values_buffer, len) = make_values_buffer(mmap.clone(), values)?;
    let nulls = make_nulls(mmap, validity, len)?;

    let builder = ArrayData::builder(DataType::Int64)
        .len(len)
        .add_buffer(values_buffer)
        .nulls(nulls);
    builder.build().map_err(ArrowBuildError::Arrow)
}

/// Convenience wrapper returning an `Int64Array`.
pub fn build_int64_array_from_mmap(
    mmap: Arc<Mmap>,
    values: &[u8],
    validity: Option<&[u8]>,
) -> Result<Int64Array, ArrowBuildError> {
    let data = int64_array_data_from_slices(mmap, values, validity)?;
    Ok(Int64Array::from(data))
}

/// Convenience wrapper returning an `ArrayData` (alias).
pub fn build_int64_array_data_from_mmap(
    mmap: Arc<Mmap>,
    values: &[u8],
    validity: Option<&[u8]>,
) -> Result<ArrayData, ArrowBuildError> {
    int64_array_data_from_slices(mmap, values, validity)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Array;
    use columnar_format::{
        align_offset, ColumnMeta, ColumnarReader, ColumnarWriter, FileHeader, COLUMN_META_LEN,
        FILE_HEADER_LEN, FILE_HEADER_MAGIC, FILE_HEADER_ON_DISK_SIZE, V0_PHYSICAL_FIXED_WIDTH_I64,
        VALUES_BUFFER_ALIGN,
    };
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_TEST_ID: AtomicUsize = AtomicUsize::new(0);

    #[derive(Clone)]
    struct TestColumn {
        values: Vec<i64>,
        validity: Option<Vec<bool>>,
    }

    #[derive(Clone)]
    struct RetainedArray {
        values: Vec<i64>,
        validity: Option<Vec<bool>>,
        array: Int64Array,
    }

    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u64(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            self.state
        }

        fn next_usize(&mut self, upper_exclusive: usize) -> usize {
            assert!(upper_exclusive > 0);
            (self.next_u64() % upper_exclusive as u64) as usize
        }
    }

    fn encode_i64(values: &[i64]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * std::mem::size_of::<i64>());
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }

    fn encode_validity(mask: &[bool]) -> Vec<u8> {
        let mut bitmap = vec![0u8; mask.len().div_ceil(8)];
        for (index, valid) in mask.iter().copied().enumerate() {
            if valid {
                bitmap[index / 8] |= 1u8 << (index % 8);
            }
        }
        bitmap
    }

    fn unique_temp_path(label: &str) -> std::path::PathBuf {
        let unique = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "{label}_{}_{}.columnar",
            std::process::id(),
            unique
        ))
    }

    fn write_multi_column_file(columns: &[TestColumn]) -> Vec<u8> {
        let mut writer = ColumnarWriter::new();
        writer.write_header_placeholder().expect("header");
        writer.write_schema_block(b"ipc-schema").expect("schema");
        writer
            .reserve_column_directory(columns.len())
            .expect("reserve directory");

        let mut metas = Vec::with_capacity(columns.len());
        for (index, column) in columns.iter().enumerate() {
            let (validity_offset, validity_length) = if let Some(validity) = &column.validity {
                assert_eq!(validity.len(), column.values.len());
                writer
                    .pad_to_alignment(MIN_BUFFER_ALIGN as usize)
                    .expect("align validity");
                let bitmap = encode_validity(validity);
                writer
                    .write_fixed_width_values(&bitmap, 1)
                    .expect("write validity")
            } else {
                (0, 0)
            };

            writer
                .pad_to_alignment(VALUES_BUFFER_ALIGN)
                .expect("align values");
            let encoded = encode_i64(&column.values);
            let (data_offset, data_length) = writer
                .write_fixed_width_values(&encoded, std::mem::size_of::<i64>())
                .expect("write values");

            metas.push(ColumnMeta {
                column_id: index as u32,
                physical_type: V0_PHYSICAL_FIXED_WIDTH_I64,
                logical_type: 0,
                data_offset,
                data_length,
                validity_offset,
                validity_length,
                offsets_offset: 0,
                offsets_length: 0,
                stats_offset: 0,
                stats_length: 0,
            });
        }

        writer
            .patch_column_directory(&metas)
            .expect("patch directory");
        writer.finalize_header().expect("finalize header");
        writer.into_inner()
    }

    fn write_test_file(values: &[i64], validity: Option<&[bool]>) -> (Vec<u8>, usize) {
        let rows = values.len();
        let schema = b"ipc-schema";

        let schema_offset = FILE_HEADER_LEN as u64;
        let schema_length = schema.len() as u64;

        let mut file = vec![0u8; FILE_HEADER_LEN];
        file.extend_from_slice(schema);

        let schema_end = file.len();
        let schema_padded_end = align_offset(schema_end, MIN_BUFFER_ALIGN as usize);
        file.resize(schema_padded_end, 0);

        let dir_offset = file.len();
        file.resize(dir_offset + COLUMN_META_LEN, 0);

        // Chunk begins after directory; align to 8 bytes.
        if file.len() % (MIN_BUFFER_ALIGN as usize) != 0 {
            let new_len = align_offset(file.len(), MIN_BUFFER_ALIGN as usize);
            file.resize(new_len, 0);
        }

        let (validity_offset, validity_length) = if let Some(mask) = validity {
            assert_eq!(mask.len(), rows);
            let bitmap_len_bytes = (rows + 7) / 8;
            let mut bitmap = vec![0u8; bitmap_len_bytes];
            for (i, valid) in mask.iter().copied().enumerate() {
                if valid {
                    bitmap[i / 8] |= 1u8 << (i % 8);
                }
            }

            let v_off = file.len() as u64;
            let padded = align_offset(bitmap.len(), MIN_BUFFER_ALIGN as usize);
            file.extend_from_slice(&bitmap);
            file.resize(file.len() + (padded - bitmap.len()), 0);
            (v_off, padded as u64)
        } else {
            (0u64, 0u64)
        };

        // Values must be aligned independently (64-byte preferred).
        let values_offset = align_offset(file.len(), 64);
        if values_offset > file.len() {
            file.resize(values_offset, 0);
        }
        let values_start = values_offset;
        let data_offset = values_start as u64;
        for v in values {
            file.extend_from_slice(&v.to_le_bytes());
        }
        let data_length = (rows * 8) as u64;

        let meta = ColumnMeta {
            column_id: 0,
            physical_type: V0_PHYSICAL_FIXED_WIDTH_I64,
            logical_type: 0,
            data_offset,
            data_length,
            validity_offset,
            validity_length,
            offsets_offset: 0,
            offsets_length: 0,
            stats_offset: 0,
            stats_length: 0,
        };
        let dir_bytes = meta.serialize();
        file[dir_offset..dir_offset + COLUMN_META_LEN].copy_from_slice(&dir_bytes);

        let hdr = FileHeader {
            magic: FILE_HEADER_MAGIC,
            version: 1,
            flags: 0,
            header_size: FILE_HEADER_ON_DISK_SIZE,
            schema_offset,
            schema_length,
            column_dir_offset: dir_offset as u64,
            column_dir_length: COLUMN_META_LEN as u64,
            reserved: [0u8; 8],
        };
        file[0..FILE_HEADER_LEN].copy_from_slice(&hdr.serialize());

        (file, values_start)
    }

    #[test]
    fn build_array_from_mmap_zero_copy_and_drop_handle() {
        let values = [10i64, -2, 33, 4];
        let (file, values_start) = write_test_file(&values, None);

        let path =
            std::env::temp_dir().join(format!("col_arrow_drop_{}.columnar", std::process::id()));
        fs::write(&path, &file).unwrap();

        let array = {
            let mmap = columnar_mmap::MmapFile::open(&path).unwrap();
            let arc = mmap.mmap_arc();
            let bytes = mmap.as_slice();
            let values_slice = &bytes[values_start..values_start + values.len() * 8];

            let arr = build_int64_array_from_mmap(arc.clone(), values_slice, None).unwrap();
            assert_eq!(
                arr.values().inner().as_slice().as_ptr(),
                values_slice.as_ptr()
            );
            arr
        };

        for (i, expected) in values.iter().enumerate() {
            assert_eq!(array.value(i), *expected);
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn build_array_with_optional_validity_bitmap() {
        let values = [1i64, 2, 3, 4, 5];
        let mask = [true, false, true, true, false];
        let (file, _values_start) = write_test_file(&values, Some(&mask));

        let path =
            std::env::temp_dir().join(format!("col_arrow_valid_{}.columnar", std::process::id()));
        fs::write(&path, &file).unwrap();

        let array = {
            let mmap = columnar_mmap::MmapFile::open(&path).unwrap();
            let arc = mmap.mmap_arc();
            let bytes = mmap.as_slice();

            let reader = ColumnarReader::new(bytes).unwrap();
            let values_slice = reader.column_values(0).unwrap();
            let validity_slice = reader.column_validity(0).unwrap();

            build_int64_array_from_mmap(arc, values_slice, validity_slice).unwrap()
        };

        for i in 0..values.len() {
            if mask[i] {
                assert!(array.is_valid(i));
                assert_eq!(array.value(i), values[i]);
            } else {
                assert!(array.is_null(i));
            }
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn misaligned_values_slice_rejected() {
        let values = [1i64, 2, 3];
        let (mut file, values_start) = write_test_file(&values, None);
        // Add trailing slack so the misaligned slice stays in bounds.
        file.push(0);

        let path = std::env::temp_dir().join(format!(
            "col_arrow_misalign_{}.columnar",
            std::process::id()
        ));
        fs::write(&path, &file).unwrap();

        let err = {
            let mmap = columnar_mmap::MmapFile::open(&path).unwrap();
            let arc = mmap.mmap_arc();
            let bytes = mmap.as_slice();

            let len_bytes = values.len() * 8;
            let misaligned = &bytes[values_start + 1..values_start + 1 + len_bytes];
            build_int64_array_from_mmap(arc, misaligned, None).unwrap_err()
        };

        assert!(matches!(err, ArrowBuildError::ValuesMisaligned { .. }));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn mmap_backed_buffer_survives_original_handle_drop() {
        let values = [11i64, 22, 33];
        let (file, values_start) = write_test_file(&values, None);

        let path = std::env::temp_dir().join(format!(
            "col_arrow_buffer_drop_{}.columnar",
            std::process::id()
        ));
        fs::write(&path, &file).unwrap();

        let buffer = {
            let mmap = columnar_mmap::MmapFile::open(&path).unwrap();
            let arc = mmap.mmap_arc();
            MmapBuffer::from_mmap_range(arc, values_start, values.len() * 8)
                .unwrap()
                .into_arrow_buffer()
        };

        let typed = buffer.typed_data::<i64>();
        assert_eq!(typed, values.as_slice());
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn randomized_valid_windows_survive_mmap_handle_drop() {
        let values: Vec<i64> = (0..256).map(|index| (index as i64 * 17) - 9_000).collect();
        let (file, values_start) = write_test_file(&values, None);
        let path = unique_temp_path("col_arrow_random_windows");
        fs::write(&path, &file).unwrap();

        let windows = {
            let mmap = columnar_mmap::MmapFile::open(&path).unwrap();
            let arc = mmap.mmap_arc();
            let bytes = mmap.as_slice();
            let values_slice = &bytes[values_start..values_start + values.len() * 8];
            let mut rng = Lcg::new(0x5eed_cafe_d00d_beef);
            let mut windows = Vec::with_capacity(128);

            for _ in 0..128 {
                let start = rng.next_usize(values.len());
                let remaining = values.len() - start;
                let len = rng.next_usize(remaining) + 1;
                let byte_start = start * std::mem::size_of::<i64>();
                let byte_end = byte_start + len * std::mem::size_of::<i64>();
                let array = build_int64_array_from_mmap(
                    arc.clone(),
                    &values_slice[byte_start..byte_end],
                    None,
                )
                .unwrap();
                windows.push((start, len, array));
            }

            windows
        };

        for (start, len, array) in windows {
            assert_eq!(array.len(), len);
            for index in 0..len {
                assert_eq!(array.value(index), values[start + index]);
            }
        }

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn stress_multiple_columns_and_batches_survive_handle_drop() {
        let mut retained = Vec::new();
        let mut paths = Vec::new();

        for batch_index in 0..6 {
            let columns: Vec<TestColumn> = (0..4)
                .map(|column_index| {
                    let rows = 48 + batch_index * 7;
                    let values: Vec<i64> = (0..rows)
                        .map(|row| {
                            (batch_index as i64 * 10_000)
                                + (column_index as i64 * 1_000)
                                + (row as i64 * 3)
                                - 77
                        })
                        .collect();
                    let validity = if column_index % 2 == 0 {
                        Some(
                            (0..rows)
                                .map(|row| (row + batch_index + column_index) % 3 != 1)
                                .collect(),
                        )
                    } else {
                        None
                    };
                    TestColumn { values, validity }
                })
                .collect();

            let file = write_multi_column_file(&columns);
            let path = unique_temp_path(&format!("col_arrow_stress_batch_{batch_index}"));
            fs::write(&path, &file).unwrap();
            paths.push(path.clone());

            let batch_arrays = {
                let mmap = columnar_mmap::MmapFile::open(&path).unwrap();
                let arc = mmap.mmap_arc();
                let reader = ColumnarReader::new(mmap.as_slice()).unwrap();
                let mut arrays = Vec::with_capacity(columns.len());

                for (column_index, column) in columns.iter().enumerate() {
                    let array = build_int64_array_from_mmap(
                        arc.clone(),
                        reader.column_values(column_index).unwrap(),
                        reader.column_validity(column_index).unwrap(),
                    )
                    .unwrap();
                    arrays.push(RetainedArray {
                        values: column.values.clone(),
                        validity: column.validity.clone(),
                        array,
                    });
                }

                arrays
            };

            retained.extend(batch_arrays);
        }

        for retained_array in retained {
            assert_eq!(retained_array.array.len(), retained_array.values.len());
            match retained_array.validity.as_ref() {
                Some(validity) => {
                    for (index, expected_valid) in validity.iter().copied().enumerate() {
                        if expected_valid {
                            assert!(retained_array.array.is_valid(index));
                            assert_eq!(
                                retained_array.array.value(index),
                                retained_array.values[index]
                            );
                        } else {
                            assert!(retained_array.array.is_null(index));
                        }
                    }
                }
                None => {
                    for (index, expected) in retained_array.values.iter().copied().enumerate() {
                        assert!(retained_array.array.is_valid(index));
                        assert_eq!(retained_array.array.value(index), expected);
                    }
                }
            }
        }

        for path in paths {
            let _ = fs::remove_file(path);
        }
    }
}
