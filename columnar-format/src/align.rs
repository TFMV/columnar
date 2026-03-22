//_! Alignment helpers for columnar format structures.

/// Minimum alignment for all buffers.
pub const MIN_BUFFER_ALIGN: usize = 8;

/// Preferred alignment for fixed-width **values** buffers (format §1.3).
pub const VALUES_BUFFER_ALIGN: usize = 64;

/// Minimum alignment for header, schema tail, and directory placement (format §1.3 / §2.1).
pub const SECTION_ALIGN: usize = 8;

/// Returns `offset` rounded up to the nearest multiple of `alignment`.
/// `alignment` must be a power of two.
#[inline]
pub const fn align_offset(offset: usize, alignment: usize) -> usize {
    offset.wrapping_add(alignment - 1) & !(alignment - 1)
}

/// Returns the number of bytes needed to pad `len` to `alignment`.
/// `alignment` must be a power of two.
#[inline]
pub const fn pad_length(len: usize, alignment: usize) -> usize {
    align_offset(len, alignment) - len
}
