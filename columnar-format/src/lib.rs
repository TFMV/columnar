//! On-disk format layout, headers, and related structs for Columnar.

mod align;
mod directory;
mod header;
mod reader;
mod stats;
mod writer;

pub use align::{align_offset, pad_length};
pub use directory::{
    BufferField, ColumnDirectory, ColumnDirectoryError, ColumnDirectoryView, ColumnMeta,
    COLUMN_META_LEN, MIN_BUFFER_ALIGN,
};
pub use header::{
    FileHeader, FileHeaderError, FILE_FLAG_VALUES_ALIGNED_64, FILE_HEADER_LEN, FILE_HEADER_MAGIC,
    FILE_HEADER_ON_DISK_SIZE, FILE_HEADER_STRUCT_LEN, FILE_HEADER_VERSION,
};
pub use reader::{
    ColumnBufferSlices, ColumnarReadError, ColumnarReader, VariableColumnBufferSlices,
};
pub use stats::{ColumnStats, Int64Stats, StatsBlockError};
pub use writer::{
    ColumnarWriteError, ColumnarWriter, ValueAlignmentStrategy, SECTION_ALIGN,
    V0_PHYSICAL_FIXED_WIDTH_I64, V0_PHYSICAL_UTF8_I32, V0_PHYSICAL_UTF8_I64, VALUES_BUFFER_ALIGN,
};
