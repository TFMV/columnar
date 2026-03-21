//! On-disk format layout, headers, and related structs for Columnar.

mod align;
mod directory;
mod header;
mod reader;
mod writer;

pub use align::{align_offset, pad_length};
pub use directory::{
    BufferField, ColumnDirectory, ColumnDirectoryError, ColumnDirectoryView, ColumnMeta,
    COLUMN_META_LEN, MIN_BUFFER_ALIGN,
};
pub use header::{
    FileHeader, FileHeaderError, FILE_HEADER_LEN, FILE_HEADER_MAGIC, FILE_HEADER_ON_DISK_SIZE,
    FILE_HEADER_STRUCT_LEN, FILE_HEADER_VERSION,
};
pub use reader::{ColumnBufferSlices, ColumnarReadError, ColumnarReader};
pub use writer::{
    ColumnarWriteError, ColumnarWriter, SECTION_ALIGN, VALUES_BUFFER_ALIGN,
    V0_PHYSICAL_FIXED_WIDTH_I64,
};
