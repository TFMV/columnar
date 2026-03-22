#![allow(dead_code)]

pub mod directory;
pub mod error;
pub mod header;
pub mod stats;
pub mod types;
pub mod writer;
pub mod align;

pub use directory::{ColumnMeta, COLUMN_META_LEN};
pub use error::{ColumnarError, ColumnarErrorType};
pub use header::{FileHeader, FILE_HEADER_LEN};
pub use stats::{ColumnStats, Int64Stats, StatsBlockError};
pub use types::{ColumnarType, UnsupportedArrowType};
pub use writer::{ColumnarWriter, ValueAlignmentStrategy};

pub use align::{pad_length, MIN_BUFFER_ALIGN, SECTION_ALIGN, VALUES_BUFFER_ALIGN};

mod reader;
pub use reader::{ColumnarReadError, ColumnarReader};
