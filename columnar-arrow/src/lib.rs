//! Construction of Arrow `ArrayData` and schemas from Columnar-mapped buffers.

mod int64;

pub use int64::{
    ArrowBuildError, build_int64_array_data_from_mmap, build_int64_array_from_mmap,
    int64_array_data_from_slices,
};
