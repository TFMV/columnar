use std::sync::Arc;

use arrow_array::{LargeStringArray, StringArray};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer};
use arrow_data::ArrayData;
use arrow_schema::DataType;
use memmap2::Mmap;

use crate::buffer::{ArrowBuildError, MmapBuffer};
use columnar_format::MIN_BUFFER_ALIGN;

fn make_values_buffer(mmap: Arc<Mmap>, values: &[u8]) -> Result<Buffer, ArrowBuildError> {
    Ok(MmapBuffer::try_new(mmap, values)?.into_arrow_buffer())
}

fn validate_offsets_i32(offsets: &[u8], values_len: usize) -> Result<usize, ArrowBuildError> {
    let mut chunks = offsets.chunks_exact(std::mem::size_of::<i32>());
    let Some(first_chunk) = chunks.next() else {
        return Err(ArrowBuildError::OffsetsTooShort {
            got_bytes: offsets.len(),
            needed_bytes: std::mem::size_of::<i32>(),
        });
    };
    let first = i32::from_le_bytes(first_chunk.try_into().expect("i32 chunk")) as i64;
    if first != 0 {
        return Err(ArrowBuildError::OffsetsMustStartAtZero { got: first });
    }

    let mut previous = first;
    let mut len = 0usize;
    for (index, chunk) in chunks.enumerate() {
        let current = i32::from_le_bytes(chunk.try_into().expect("i32 chunk")) as i64;
        if current < previous {
            return Err(ArrowBuildError::OffsetsNotMonotonic {
                index: index + 1,
                previous,
                current,
            });
        }
        if current as usize > values_len {
            return Err(ArrowBuildError::OffsetOutOfBounds {
                index: index + 1,
                offset: current,
                values_len,
            });
        }
        previous = current;
        len += 1;
    }
    Ok(len)
}

fn validate_offsets_i64(offsets: &[u8], values_len: usize) -> Result<usize, ArrowBuildError> {
    let mut chunks = offsets.chunks_exact(std::mem::size_of::<i64>());
    let Some(first_chunk) = chunks.next() else {
        return Err(ArrowBuildError::OffsetsTooShort {
            got_bytes: offsets.len(),
            needed_bytes: std::mem::size_of::<i64>(),
        });
    };
    let first = i64::from_le_bytes(first_chunk.try_into().expect("i64 chunk"));
    if first != 0 {
        return Err(ArrowBuildError::OffsetsMustStartAtZero { got: first });
    }

    let mut previous = first;
    let mut len = 0usize;
    for (index, chunk) in chunks.enumerate() {
        let current = i64::from_le_bytes(chunk.try_into().expect("i64 chunk"));
        if current < previous {
            return Err(ArrowBuildError::OffsetsNotMonotonic {
                index: index + 1,
                previous,
                current,
            });
        }
        if current as usize > values_len {
            return Err(ArrowBuildError::OffsetOutOfBounds {
                index: index + 1,
                offset: current,
                values_len,
            });
        }
        previous = current;
        len += 1;
    }
    Ok(len)
}

fn make_offsets_buffer_i32(
    mmap: Arc<Mmap>,
    offsets: &[u8],
    values_len: usize,
) -> Result<(Buffer, usize), ArrowBuildError> {
    const WIDTH: usize = std::mem::size_of::<i32>();
    if offsets.len() % WIDTH != 0 {
        return Err(ArrowBuildError::InvalidOffsetsLength {
            got: offsets.len(),
            offset_width: WIDTH,
        });
    }
    if offsets.len() < WIDTH {
        return Err(ArrowBuildError::OffsetsTooShort {
            got_bytes: offsets.len(),
            needed_bytes: WIDTH,
        });
    }
    let ptr = offsets.as_ptr() as usize;
    let alignment = std::mem::align_of::<i32>();
    if ptr % alignment != 0 {
        return Err(ArrowBuildError::OffsetsMisaligned { ptr, alignment });
    }
    let len = validate_offsets_i32(offsets, values_len)?;
    let buffer = MmapBuffer::try_new(mmap, offsets)?.into_arrow_buffer();
    Ok((buffer, len))
}

fn make_offsets_buffer_i64(
    mmap: Arc<Mmap>,
    offsets: &[u8],
    values_len: usize,
) -> Result<(Buffer, usize), ArrowBuildError> {
    const WIDTH: usize = std::mem::size_of::<i64>();
    if offsets.len() % WIDTH != 0 {
        return Err(ArrowBuildError::InvalidOffsetsLength {
            got: offsets.len(),
            offset_width: WIDTH,
        });
    }
    if offsets.len() < WIDTH {
        return Err(ArrowBuildError::OffsetsTooShort {
            got_bytes: offsets.len(),
            needed_bytes: WIDTH,
        });
    }
    let ptr = offsets.as_ptr() as usize;
    let alignment = std::mem::align_of::<i64>();
    if ptr % alignment != 0 {
        return Err(ArrowBuildError::OffsetsMisaligned { ptr, alignment });
    }
    let len = validate_offsets_i64(offsets, values_len)?;
    let buffer = MmapBuffer::try_new(mmap, offsets)?.into_arrow_buffer();
    Ok((buffer, len))
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
    let needed_bytes = len.div_ceil(8);
    if validity.len() < needed_bytes {
        return Err(ArrowBuildError::ValidityTooShort {
            got_bytes: validity.len(),
            needed_bytes,
        });
    }
    let ptr = validity.as_ptr() as usize;
    let alignment = MIN_BUFFER_ALIGN as usize;
    if !validity.is_empty() && ptr % alignment != 0 {
        return Err(ArrowBuildError::ValidityMisaligned { ptr, alignment });
    }
    let validity_buffer = MmapBuffer::try_new(mmap, validity)?.into_arrow_buffer();
    Ok(Some(NullBuffer::new(BooleanBuffer::new(
        validity_buffer,
        0,
        len,
    ))))
}

pub fn build_utf8_array(
    mmap: Arc<Mmap>,
    offsets: &[u8],
    values: &[u8],
    validity: Option<&[u8]>,
) -> Result<StringArray, ArrowBuildError> {
    let (offsets_buffer, len) = make_offsets_buffer_i32(mmap.clone(), offsets, values.len())?;
    let values_buffer = make_values_buffer(mmap.clone(), values)?;
    let nulls = make_nulls(mmap, validity, len)?;
    let array_data = ArrayData::builder(DataType::Utf8)
        .len(len)
        .add_buffer(offsets_buffer)
        .add_buffer(values_buffer)
        .nulls(nulls)
        .build()
        .map_err(ArrowBuildError::Arrow)?;
    Ok(StringArray::from(array_data))
}

pub fn build_large_utf8_array(
    mmap: Arc<Mmap>,
    offsets: &[u8],
    values: &[u8],
    validity: Option<&[u8]>,
) -> Result<LargeStringArray, ArrowBuildError> {
    let (offsets_buffer, len) = make_offsets_buffer_i64(mmap.clone(), offsets, values.len())?;
    let values_buffer = make_values_buffer(mmap.clone(), values)?;
    let nulls = make_nulls(mmap, validity, len)?;
    let array_data = ArrayData::builder(DataType::LargeUtf8)
        .len(len)
        .add_buffer(offsets_buffer)
        .add_buffer(values_buffer)
        .nulls(nulls)
        .build()
        .map_err(ArrowBuildError::Arrow)?;
    Ok(LargeStringArray::from(array_data))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Array;
    use columnar_format::{
        ColumnMeta, ColumnarReader, ColumnarWriter, MIN_BUFFER_ALIGN, V0_PHYSICAL_UTF8_I32,
        V0_PHYSICAL_UTF8_I64, VALUES_BUFFER_ALIGN,
    };
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_TEST_ID: AtomicUsize = AtomicUsize::new(0);

    fn unique_temp_path(label: &str) -> std::path::PathBuf {
        let unique = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "{label}_{}_{}.columnar",
            std::process::id(),
            unique
        ))
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

    fn encode_i32_offsets(offsets: &[i32]) -> Vec<u8> {
        #[cfg(target_endian = "little")]
        {
            // SAFETY: `i32` has a defined layout, and we are on a little-endian system, so
            // a simple memory copy is correct.
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    offsets.as_ptr() as *const u8,
                    offsets.len() * std::mem::size_of::<i32>(),
                )
            };
            bytes.to_vec()
        }

        #[cfg(not(target_endian = "little"))]
        {
            let mut out = Vec::with_capacity(std::mem::size_of_val(offsets));
            for offset in offsets {
                out.extend_from_slice(&offset.to_le_bytes());
            }
            out
        }
    }

    fn encode_i64_offsets(offsets: &[i64]) -> Vec<u8> {
        #[cfg(target_endian = "little")]
        {
            // SAFETY: `i64` has a defined layout, and we are on a little-endian system, so
            // a simple memory copy is correct.
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    offsets.as_ptr() as *const u8,
                    offsets.len() * std::mem::size_of::<i64>(),
                )
            };
            bytes.to_vec()
        }

        #[cfg(not(target_endian = "little"))]
        {
            let mut out = Vec::with_capacity(std::mem::size_of_val(offsets));
            for offset in offsets {
                out.extend_from_slice(&offset.to_le_bytes());
            }
            out
        }
    }

    fn write_utf8_test_file(
        physical_type: u32,
        offsets: &[u8],
        values: &[u8],
        validity: Option<&[bool]>,
    ) -> Vec<u8> {
        let mut writer = ColumnarWriter::new();
        writer.write_header_placeholder().unwrap();
        writer.write_schema_block(b"utf8-schema").unwrap();
        writer.reserve_column_directory(1).unwrap();

        let (validity_offset, validity_length) = if let Some(validity) = validity {
            writer
                .pad_to_alignment(MIN_BUFFER_ALIGN as usize)
                .expect("align validity");
            let bitmap = encode_validity(validity);
            writer.write_fixed_width_values(&bitmap, 1).unwrap()
        } else {
            (0, 0)
        };

        writer
            .pad_to_alignment(MIN_BUFFER_ALIGN as usize)
            .expect("align offsets");
        let (offsets_offset, offsets_length) = writer.write_fixed_width_values(offsets, 1).unwrap();

        writer
            .pad_to_alignment(VALUES_BUFFER_ALIGN)
            .expect("align values");
        let (data_offset, data_length) = writer.write_fixed_width_values(values, 1).unwrap();

        let meta = ColumnMeta {
            column_id: 0,
            physical_type,
            logical_type: 0,
            data_offset,
            data_length,
            validity_offset,
            validity_length,
            offsets_offset,
            offsets_length,
            stats_offset: 0,
            stats_length: 0,
        };
        writer
            .patch_column_directory(std::slice::from_ref(&meta))
            .unwrap();
        writer.finalize_header().unwrap();
        writer.into_inner()
    }

    #[test]
    fn utf8_array_zero_copy_handles_multiple_and_empty_strings() {
        let offsets = [0i32, 5, 5, 10, 14];
        let values = b"alphabetaomega";
        let file = write_utf8_test_file(
            V0_PHYSICAL_UTF8_I32,
            &encode_i32_offsets(&offsets),
            values,
            None,
        );
        let path = unique_temp_path("col_arrow_utf8_zero_copy");
        fs::write(&path, &file).unwrap();

        let array = {
            let mmap = columnar_mmap::MmapFile::open(&path).unwrap();
            let arc = mmap.mmap_arc();
            let reader = ColumnarReader::new(mmap.as_slice()).unwrap();
            let variable = reader.variable_column_buffers(0).unwrap();
            let array = build_utf8_array(
                arc,
                variable.offsets,
                variable.values,
                variable.validity,
            )
            .unwrap();
            assert_eq!(array.value_data().as_ptr(), variable.values.as_ptr());
            array
        };

        assert_eq!(array.value(0), "alpha");
        assert_eq!(array.value(1), "");
        assert_eq!(array.value(2), "betao");
        assert_eq!(array.value(3), "mega");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn utf8_array_preserves_nulls_after_handle_drop() {
        let offsets = [0i32, 5, 5, 9];
        let values = b"alphabeta";
        let validity = [true, false, true];
        let file = write_utf8_test_file(
            V0_PHYSICAL_UTF8_I32,
            &encode_i32_offsets(&offsets),
            values,
            Some(&validity),
        );
        let path = unique_temp_path("col_arrow_utf8_nulls");
        fs::write(&path, &file).unwrap();

        let array = {
            let mmap = columnar_mmap::MmapFile::open(&path).unwrap();
            let arc = mmap.mmap_arc();
            let reader = ColumnarReader::new(mmap.as_slice()).unwrap();
            let variable = reader.variable_column_buffers(0).unwrap();
            build_utf8_array(arc, variable.offsets, variable.values, variable.validity).unwrap()
        };

        assert_eq!(array.value(0), "alpha");
        assert!(array.is_null(1));
        assert_eq!(array.value(2), "beta");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn malformed_i32_offsets_are_rejected_explicitly() {
        let values = b"alphabet";
        let cases: [(&[i32], ArrowBuildError); 4] = [
            (
                &[1, 5, 8],
                ArrowBuildError::OffsetsMustStartAtZero { got: 1 },
            ),
            (
                &[0, 6, 4],
                ArrowBuildError::OffsetsNotMonotonic {
                    index: 2,
                    previous: 6,
                    current: 4,
                },
            ),
            (
                &[0, 3, 9],
                ArrowBuildError::OffsetOutOfBounds {
                    index: 2,
                    offset: 9,
                    values_len: values.len(),
                },
            ),
            (
                &[0, 8, 7],
                ArrowBuildError::OffsetsNotMonotonic {
                    index: 2,
                    previous: 8,
                    current: 7,
                },
            ),
        ];

        for (offsets, expected) in cases {
            let file = write_utf8_test_file(
                V0_PHYSICAL_UTF8_I32,
                &encode_i32_offsets(offsets),
                values,
                None,
            );
            let path = unique_temp_path("col_arrow_utf8_bad_i32");
            fs::write(&path, &file).unwrap();

            let err = {
                let mmap = columnar_mmap::MmapFile::open(&path).unwrap();
                let arc = mmap.mmap_arc();
                let reader = ColumnarReader::new(mmap.as_slice()).unwrap();
                let variable = reader.variable_column_buffers(0).unwrap();
                build_utf8_array(
                    arc,
                    variable.offsets,
                    variable.values,
                    variable.validity,
                )
                .unwrap_err()
            };

            assert_eq!(format!("{err}"), format!("{expected}"));
            let _ = fs::remove_file(&path);
        }
    }

    #[test]
    fn malformed_i64_offsets_are_rejected_explicitly() {
        let values = b"helloworld";
        let cases: [(&[i64], ArrowBuildError); 3] = [
            (
                &[2, 5, 10],
                ArrowBuildError::OffsetsMustStartAtZero { got: 2 },
            ),
            (
                &[0, 7, 6],
                ArrowBuildError::OffsetsNotMonotonic {
                    index: 2,
                    previous: 7,
                    current: 6,
                },
            ),
            (
                &[0, 5, 11],
                ArrowBuildError::OffsetOutOfBounds {
                    index: 2,
                    offset: 11,
                    values_len: values.len(),
                },
            ),
        ];

        for (offsets, expected) in cases {
            let file = write_utf8_test_file(
                V0_PHYSICAL_UTF8_I64,
                &encode_i64_offsets(offsets),
                values,
                None,
            );
            let path = unique_temp_path("col_arrow_utf8_bad_i64");
            fs::write(&path, &file).unwrap();

            let err = {
                let mmap = columnar_mmap::MmapFile::open(&path).unwrap();
                let arc = mmap.mmap_arc();
                let reader = ColumnarReader::new(mmap.as_slice()).unwrap();
                let variable = reader.variable_column_buffers(0).unwrap();
                build_large_utf8_array(
                    arc,
                    variable.offsets,
                    variable.values,
                    variable.validity,
                )
                .unwrap_err()
            };

            assert_eq!(format!("{err}"), format!("{expected}"));
            let _ = fs::remove_file(&path);
        }
    }

    #[test]
    fn fuzz_style_offset_mutations_never_escape_explicit_validation() {
        let base = [0i32, 2, 5, 9];
        let values = b"alphabetz";
        let mutations = [
            [1, 2, 5, 9],
            [0, 5, 4, 9],
            [0, 2, 5, 10],
            [0, 3, 2, 9],
            [0, 2, 9, 8],
        ];

        for offsets in mutations {
            let file = write_utf8_test_file(
                V0_PHYSICAL_UTF8_I32,
                &encode_i32_offsets(&offsets),
                values,
                None,
            );
            let path = unique_temp_path("col_arrow_utf8_fuzz");
            fs::write(&path, &file).unwrap();

            let err = {
                let mmap = columnar_mmap::MmapFile::open(&path).unwrap();
                let arc = mmap.mmap_arc();
                let reader = ColumnarReader::new(mmap.as_slice()).unwrap();
                let variable = reader.variable_column_buffers(0).unwrap();
                build_utf8_array(
                    arc,
                    variable.offsets,
                    variable.values,
                    variable.validity,
                )
                .unwrap_err()
            };

            assert!(!matches!(err, ArrowBuildError::Arrow(_)));
            assert!(
                matches!(
                    err,
                    ArrowBuildError::OffsetsMustStartAtZero { .. }
                        | ArrowBuildError::OffsetsNotMonotonic { .. }
                        | ArrowBuildError::OffsetOutOfBounds { .. }
                ),
                "unexpected malformed-offset error for base {:?}: {err:?}",
                base
            );
            let _ = fs::remove_file(&path);
        }
    }

    #[test]
    fn large_utf8_array_uses_i64_offsets_zero_copy() {
        let offsets = [0i64, 0, 5, 11];
        let values = b"helloworld!";
        let file = write_utf8_test_file(
            V0_PHYSICAL_UTF8_I64,
            &encode_i64_offsets(&offsets),
            values,
            None,
        );
        let path = unique_temp_path("col_arrow_large_utf8");
        fs::write(&path, &file).unwrap();

        let array = {
            let mmap = columnar_mmap::MmapFile::open(&path).unwrap();
            let arc = mmap.mmap_arc();
            let reader = ColumnarReader::new(mmap.as_slice()).unwrap();
            let variable = reader.variable_column_buffers(0).unwrap();
            build_large_utf8_array(
                arc,
                variable.offsets,
                variable.values,
                variable.validity,
            )
            .unwrap()
        };

        assert_eq!(array.value(0), "");
        assert_eq!(array.value(1), "hello");
        assert_eq!(array.value(2), "world!");
        let _ = fs::remove_file(&path);
    }
}
