use std::sync::Arc;

use arrow_array::{LargeStringArray, StringArray};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer};
use arrow_data::ArrayData;
use arrow_schema::DataType;
use memmap2::Mmap;

use crate::buffer::{mmap_buffer, ArrowBuildError};
use columnar_format::MIN_BUFFER_ALIGN;

fn make_values_buffer(mmap: Arc<Mmap>, values: &[u8]) -> Result<Buffer, ArrowBuildError> {
    let offset = values.as_ptr() as usize - mmap.as_ptr() as usize;
    Ok(unsafe { mmap_buffer(mmap, offset, values.len()) })
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
    let offset = offsets.as_ptr() as usize - mmap.as_ptr() as usize;
    let buffer = unsafe { mmap_buffer(mmap, offset, offsets.len()) };
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
    let offset = offsets.as_ptr() as usize - mmap.as_ptr() as usize;
    let buffer = unsafe { mmap_buffer(mmap, offset, offsets.len()) };
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
    let needed_bytes = (len + 7) / 8;
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
    let offset = validity.as_ptr() as usize - mmap.as_ptr() as usize;
    let validity_buffer = unsafe { mmap_buffer(mmap, offset, validity.len()) };
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
        .build()?;
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
        .build()?;
    Ok(LargeStringArray::from(array_data))
}
