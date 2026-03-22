use arrow_array::Int64Array;
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer};
use arrow_data::ArrayData;
use arrow_schema::DataType;
use memmap2::Mmap;
use std::sync::Arc;

use crate::buffer::{mmap_buffer, ArrowBuildError};
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
    let offset = values.as_ptr() as usize - mmap.as_ptr() as usize;
    let buf = unsafe { mmap_buffer(mmap, offset, values.len()) };
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
    if !validity.is_empty() && validity_ptr % alignment != 0 {
        return Err(ArrowBuildError::ValidityMisaligned {
            ptr: validity_ptr,
            alignment,
        });
    }
    let offset = validity.as_ptr() as usize - mmap.as_ptr() as usize;
    let validity_buffer = unsafe { mmap_buffer(mmap, offset, validity.len()) };
    let boolean_buffer = BooleanBuffer::new(validity_buffer, 0, len);
    Ok(Some(NullBuffer::new(boolean_buffer)))
}

/// Build an Arrow `Int64Array` from zero-copy buffers over a shared `Mmap` region.
///
/// The returned array and its buffers retain an internal `Arc<Mmap>` so the data remains valid
/// even if the caller drops its original `MmapFile` handle.
pub fn build_int64_array(
    mmap: Arc<Mmap>,
    values: &[u8],
    validity: Option<&[u8]>,
) -> Result<Int64Array, ArrowBuildError> {
    let (values_buffer, len) = make_values_buffer(mmap.clone(), values)?;
    let nulls = make_nulls(mmap, validity, len)?;

    let builder = ArrayData::builder(DataType::Int64)
        .len(len)
        .add_buffer(values_buffer)
        .nulls(nulls);
    let array_data = builder.build()?;
    Ok(Int64Array::from(array_data))
}
