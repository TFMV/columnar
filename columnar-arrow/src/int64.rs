use std::sync::Arc;

use arrow_array::Int64Array;
use arrow_buffer::alloc::Allocation as ArrowAllocation;
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer};
use arrow_data::ArrayData;
use arrow_schema::{ArrowError, DataType};
use memmap2::Mmap;

use columnar_format::MIN_BUFFER_ALIGN;

/// Errors while building an Arrow Int64 array from Columnar zero-copy slices.
#[derive(Debug)]
pub enum ArrowBuildError {
    /// `values.len()` was not a multiple of 8 bytes.
    InvalidValuesLength { got: usize },
    /// `values` pointer not aligned for `i64`.
    ValuesMisaligned { ptr: usize, alignment: usize },
    /// `validity` length too small for `len` rows.
    ValidityTooShort { got_bytes: usize, needed_bytes: usize },
    /// `validity` pointer not aligned to Columnar alignment.
    ValidityMisaligned { ptr: usize, alignment: usize },
    /// The provided slice pointer does not fall within the provided mmap address range.
    SliceNotWithinMmap {
        slice_ptr: usize,
        slice_len: usize,
        mmap_len: usize,
    },
    /// `arrow_data` validation failed.
    Arrow(ArrowError),
}

impl std::fmt::Display for ArrowBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArrowBuildError::InvalidValuesLength { got } => {
                write!(f, "values length {got} is not a multiple of 8")
            }
            ArrowBuildError::ValuesMisaligned { ptr, alignment } => write!(
                f,
                "values buffer pointer {ptr} is not aligned to {alignment}"
            ),
            ArrowBuildError::ValidityTooShort {
                got_bytes,
                needed_bytes,
            } => write!(
                f,
                "validity length {got_bytes} bytes is too small, need {needed_bytes}"
            ),
            ArrowBuildError::ValidityMisaligned { ptr, alignment } => write!(
                f,
                "validity buffer pointer {ptr} is not aligned to {alignment}"
            ),
            ArrowBuildError::SliceNotWithinMmap {
                slice_ptr,
                slice_len,
                mmap_len,
            } => write!(
                f,
                "slice [{slice_ptr}, {}) is not within mmap length {mmap_len}",
                slice_ptr + slice_len
            ),
            ArrowBuildError::Arrow(e) => write!(f, "arrow error: {e}"),
        }
    }
}

impl std::error::Error for ArrowBuildError {}

#[derive(Clone)]
struct MmapOwner {
    _mmap: Arc<Mmap>,
}

// `MmapOwner` only stores an immutable `Arc<Mmap>`. For our zero-copy reads, this is unwind-safe.
impl std::panic::RefUnwindSafe for MmapOwner {}

fn slice_within_mmap(mmap: &Mmap, slice: &[u8]) -> Result<(), ArrowBuildError> {
    let base_ptr = mmap.as_ref().as_ptr() as usize;
    let slice_ptr = slice.as_ptr() as usize;
    let slice_len = slice.len();
    let mmap_len = mmap.len();

    if slice_len == 0 {
        return Ok(());
    }
    if slice_ptr < base_ptr {
        return Err(ArrowBuildError::SliceNotWithinMmap {
            slice_ptr,
            slice_len,
            mmap_len,
        });
    }
    let off = slice_ptr - base_ptr;
    let end = off
        .checked_add(slice_len)
        .ok_or(ArrowBuildError::SliceNotWithinMmap {
            slice_ptr,
            slice_len,
            mmap_len,
        })?;
    if end > mmap_len {
        return Err(ArrowBuildError::SliceNotWithinMmap {
            slice_ptr,
            slice_len,
            mmap_len,
        });
    }
    Ok(())
}

/// Safety: `ptr..ptr+len` must point into the live mmap region, and the returned `Buffer` keeps
/// the mmap alive by storing an internal `Arc<Mmap>` in `owner`.
unsafe fn buffer_from_mmap_slice(
    mmap: Arc<Mmap>,
    slice: &[u8],
) -> Result<Buffer, ArrowBuildError> {
    if slice.is_empty() {
        return Ok(Buffer::from_vec(Vec::<u8>::new()));
    }

    let ptr = std::ptr::NonNull::new(slice.as_ptr() as *mut u8).ok_or(
        ArrowBuildError::SliceNotWithinMmap {
            slice_ptr: slice.as_ptr() as usize,
            slice_len: slice.len(),
            mmap_len: mmap.len(),
        },
    )?;
    let len = slice.len();
    let owner: Arc<dyn ArrowAllocation> = Arc::new(MmapOwner { _mmap: mmap });
    Ok(Buffer::from_custom_allocation(ptr, len, owner))
}

fn make_values_buffer(
    mmap: Arc<Mmap>,
    values: &[u8],
) -> Result<(Buffer, usize), ArrowBuildError> {
    if values.len() % 8 != 0 {
        return Err(ArrowBuildError::InvalidValuesLength { got: values.len() });
    }
    let len = values.len() / 8;
    let ptr = values.as_ptr() as usize;
    let alignment = std::mem::align_of::<i64>();
    if ptr % alignment != 0 {
        return Err(ArrowBuildError::ValuesMisaligned { ptr, alignment });
    }
    slice_within_mmap(&mmap, values)?;
    // SAFETY: checks above ensure `values` points into `mmap` and is aligned for `i64`.
    let buf = unsafe { buffer_from_mmap_slice(mmap, values)? };
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
    slice_within_mmap(&mmap, validity)?;

    // SAFETY: `validity` points into mmap, and `NullBuffer` consumes it as a packed bitmask.
    let validity_buffer = unsafe { buffer_from_mmap_slice(mmap, validity)? };
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
        ColumnMeta, ColumnarReader, FileHeader, FILE_HEADER_MAGIC, FILE_HEADER_LEN,
        FILE_HEADER_ON_DISK_SIZE, V0_PHYSICAL_FIXED_WIDTH_I64, COLUMN_META_LEN, align_offset,
    };
    use std::fs;

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

        let path = std::env::temp_dir().join(format!(
            "col_arrow_drop_{}.columnar",
            std::process::id()
        ));
        fs::write(&path, &file).unwrap();

        let array = {
            let mmap = columnar_mmap::MmapFile::open(&path).unwrap();
            let arc = mmap.mmap_arc();
            let bytes = mmap.as_slice();
            let values_slice = &bytes[values_start..values_start + values.len() * 8];

            let arr = build_int64_array_from_mmap(arc.clone(), values_slice, None).unwrap();
            assert_eq!(arr.values().inner().as_slice().as_ptr(), values_slice.as_ptr());
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

        let path = std::env::temp_dir().join(format!(
            "col_arrow_valid_{}.columnar",
            std::process::id()
        ));
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
}

