use std::ptr::NonNull;
use std::sync::Arc;

use arrow_buffer::Buffer;
use memmap2::Mmap;

#[derive(Debug, PartialEq, Eq)]
pub enum ArrowBuildError {
    InvalidValuesLength { got: usize },
    ValuesMisaligned { ptr: usize, alignment: usize },
    ValidityTooShort { got_bytes: usize, needed_bytes: usize },
    ValidityMisaligned { ptr: usize, alignment: usize },
    InvalidOffsetsLength { got: usize, offset_width: usize },
    OffsetsTooShort { got_bytes: usize, needed_bytes: usize },
    OffsetsMisaligned { ptr: usize, alignment: usize },
    OffsetsMustStartAtZero { got: i64 },
    OffsetsNotMonotonic { index: usize, previous: i64, current: i64 },
    OffsetOutOfBounds { index: usize, offset: i64, values_len: usize },
    Arrow(String),
}

impl From<arrow_schema::ArrowError> for ArrowBuildError {
    fn from(value: arrow_schema::ArrowError) -> Self {
        Self::Arrow(value.to_string())
    }
}

impl std::fmt::Display for ArrowBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for ArrowBuildError {}

/// Creates an Arrow `Buffer` from a sub-region of a memory-mapped file.
///
/// # Safety
///
/// This function is unsafe because it creates a `Buffer` that directly references
/// a memory-mapped region. The caller must ensure that the `mmap` lives as long as
/// the `Buffer` to avoid use-after-free. The buffer holds an Arc to the Mmap
/// to ensure this, but it's the caller's responsibility to construct it correctly.
pub unsafe fn mmap_buffer(mmap: Arc<Mmap>, offset: usize, len: usize) -> Buffer {
    let ptr = mmap.as_ptr().add(offset);
    Buffer::from_custom_allocation(
        NonNull::new(ptr as *mut u8).expect("pointer must not be null"),
        len,
        mmap,
    )
}
