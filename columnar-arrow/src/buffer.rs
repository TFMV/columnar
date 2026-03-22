use std::sync::Arc;

use arrow_buffer::alloc::Allocation as ArrowAllocation;
use arrow_buffer::Buffer;
use arrow_schema::ArrowError;
use memmap2::Mmap;

/// Errors while building Arrow arrays from Columnar zero-copy slices.
#[derive(Debug)]
pub enum ArrowBuildError {
    InvalidValuesLength {
        got: usize,
    },
    ValuesMisaligned {
        ptr: usize,
        alignment: usize,
    },
    InvalidOffsetsLength {
        got: usize,
        offset_width: usize,
    },
    OffsetsTooShort {
        got_bytes: usize,
        needed_bytes: usize,
    },
    OffsetsMisaligned {
        ptr: usize,
        alignment: usize,
    },
    MissingOffsets,
    OffsetsMustStartAtZero {
        got: i64,
    },
    OffsetsNotMonotonic {
        index: usize,
        previous: i64,
        current: i64,
    },
    OffsetOutOfBounds {
        index: usize,
        offset: i64,
        values_len: usize,
    },
    ValidityTooShort {
        got_bytes: usize,
        needed_bytes: usize,
    },
    ValidityMisaligned {
        ptr: usize,
        alignment: usize,
    },
    SliceNotWithinMmap {
        slice_ptr: usize,
        slice_len: usize,
        mmap_len: usize,
    },
    Arrow(ArrowError),
}

impl std::fmt::Display for ArrowBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidValuesLength { got } => {
                write!(f, "values length {got} is not a multiple of 8")
            }
            Self::ValuesMisaligned { ptr, alignment } => {
                write!(
                    f,
                    "values buffer pointer {ptr} is not aligned to {alignment}"
                )
            }
            Self::InvalidOffsetsLength { got, offset_width } => write!(
                f,
                "offsets length {got} is not a multiple of offset width {offset_width}"
            ),
            Self::OffsetsTooShort {
                got_bytes,
                needed_bytes,
            } => write!(
                f,
                "offsets length {got_bytes} bytes is too small, need at least {needed_bytes}"
            ),
            Self::OffsetsMisaligned { ptr, alignment } => {
                write!(
                    f,
                    "offsets buffer pointer {ptr} is not aligned to {alignment}"
                )
            }
            Self::MissingOffsets => write!(f, "variable-length column requires an offsets buffer"),
            Self::OffsetsMustStartAtZero { got } => {
                write!(f, "variable-length offsets must start at 0, got {got}")
            }
            Self::OffsetsNotMonotonic {
                index,
                previous,
                current,
            } => write!(
                f,
                "offsets are not monotonic at index {index}: previous {previous}, current {current}"
            ),
            Self::OffsetOutOfBounds {
                index,
                offset,
                values_len,
            } => write!(
                f,
                "offset {offset} at index {index} exceeds values buffer length {values_len}"
            ),
            Self::ValidityTooShort {
                got_bytes,
                needed_bytes,
            } => write!(
                f,
                "validity length {got_bytes} bytes is too small, need {needed_bytes}"
            ),
            Self::ValidityMisaligned { ptr, alignment } => write!(
                f,
                "validity buffer pointer {ptr} is not aligned to {alignment}"
            ),
            Self::SliceNotWithinMmap {
                slice_ptr,
                slice_len,
                mmap_len,
            } => write!(
                f,
                "slice [{slice_ptr}, {}) is not within mmap length {mmap_len}",
                slice_ptr + slice_len
            ),
            Self::Arrow(error) => write!(f, "arrow error: {error}"),
        }
    }
}

impl std::error::Error for ArrowBuildError {}

/// Zero-copy buffer view backed by a retained `Arc<Mmap>`.
#[derive(Clone)]
pub struct MmapBuffer {
    arc: Arc<Mmap>,
    ptr: *const u8,
    len: usize,
}

#[derive(Clone)]
struct MmapOwner {
    _mmap: Arc<Mmap>,
}

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

fn mmap_range(
    mmap: &Mmap,
    offset: usize,
    len: usize,
) -> Result<(*const u8, usize), ArrowBuildError> {
    let end = offset
        .checked_add(len)
        .ok_or(ArrowBuildError::SliceNotWithinMmap {
            slice_ptr: mmap.as_ref().as_ptr() as usize + offset,
            slice_len: len,
            mmap_len: mmap.len(),
        })?;
    if end > mmap.len() {
        return Err(ArrowBuildError::SliceNotWithinMmap {
            slice_ptr: mmap.as_ref().as_ptr() as usize + offset,
            slice_len: len,
            mmap_len: mmap.len(),
        });
    }

    let ptr = if len == 0 {
        if mmap.is_empty() {
            std::ptr::NonNull::<u8>::dangling().as_ptr()
        } else {
            mmap.as_ref().as_ptr()
        }
    } else {
        // SAFETY: `end <= mmap.len()` was checked above, so `offset` lies within the mapping and
        // `base.add(offset)` produces a pointer to the first byte of the requested range.
        unsafe { mmap.as_ref().as_ptr().add(offset) }
    };

    Ok((ptr, len))
}

impl MmapBuffer {
    pub fn try_new(arc: Arc<Mmap>, slice: &[u8]) -> Result<Self, ArrowBuildError> {
        slice_within_mmap(&arc, slice)?;
        let base = arc.as_ref().as_ptr() as usize;
        let ptr = slice.as_ptr() as usize;
        let offset = ptr.saturating_sub(base);
        Self::from_mmap_range(arc, offset, slice.len())
    }

    pub(crate) fn from_mmap_range(
        arc: Arc<Mmap>,
        offset: usize,
        len: usize,
    ) -> Result<Self, ArrowBuildError> {
        let (ptr, len) = mmap_range(&arc, offset, len)?;
        Ok(Self { arc, ptr, len })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn into_arrow_buffer(self) -> Buffer {
        let owner: Arc<dyn ArrowAllocation> = Arc::new(MmapOwner { _mmap: self.arc });
        let ptr =
            std::ptr::NonNull::new(self.ptr as *mut u8).unwrap_or_else(std::ptr::NonNull::dangling);

        // SAFETY:
        // - `ptr` is non-null. For `len > 0`, it comes from a range validated to lie wholly
        //   within the live mmap region.
        // - Bounds were checked in `MmapBuffer::try_new` / `from_mmap_range`, so `ptr..ptr+len`
        //   is valid for reads.
        // - Typed callers validate alignment before constructing typed Arrow arrays.
        // - The returned `Buffer` co-owns the underlying mapping through `owner`, which holds an
        //   `Arc<Mmap>`, so the backing memory outlives all Arrow buffer clones and slices.
        // - For `len == 0`, Arrow never dereferences the pointer; a dangling non-null pointer is valid.
        unsafe { Buffer::from_custom_allocation(ptr, self.len, owner) }
    }
}
