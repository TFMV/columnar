//! Construction of Arrow `ArrayData` and schemas from Columnar-mapped buffers.

mod buffer;
mod int64;
mod utf8;

pub use buffer::{ArrowBuildError, MmapBuffer};
pub use int64::{
    build_int64_array_data_from_mmap, build_int64_array_from_mmap, int64_array_data_from_slices,
};
pub use utf8::{
    build_large_utf8_array_from_mmap, build_utf8_array_from_mmap,
    large_utf8_array_data_from_slices, utf8_array_data_from_slices,
};
