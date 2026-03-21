//! On-disk format layout, headers, and related structs for Columnar.

mod align;
mod header;

pub use align::{align_offset, pad_length};
pub use header::{
    FileHeader, FileHeaderError, FILE_HEADER_LEN, FILE_HEADER_MAGIC, FILE_HEADER_ON_DISK_SIZE,
    FILE_HEADER_STRUCT_LEN, FILE_HEADER_VERSION,
};
