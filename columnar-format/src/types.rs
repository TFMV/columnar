
//! Centralized type system for Columnar.

use arrow_schema::DataType as ArrowDataType;

/// The physical type of a column in a Columnar file.
///
/// This enum maps directly to Arrow data types and is serialized as a `u32`
/// in the file metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ColumnarType {
    // Current types
    Int64 = 0,
    Utf8 = 1,
    LargeUtf8 = 2,

    // Planned types
    Float64 = 10,
    Boolean = 11,
    // Other types like Timestamp, Date, etc. can be added here.
}

impl ColumnarType {
    /// The number of bytes for the offset type in a variable-length column.
    pub fn offset_width(&self) -> Option<usize> {
        match self {
            Self::Utf8 => Some(std::mem::size_of::<i32>()),
            Self::LargeUtf8 => Some(std::mem::size_of::<i64>()),
            _ => None,
        }
    }
}

impl From<ColumnarType> for ArrowDataType {
    fn from(value: ColumnarType) -> Self {
        match value {
            ColumnarType::Int64 => ArrowDataType::Int64,
            ColumnarType::Utf8 => ArrowDataType::Utf8,
            ColumnarType::LargeUtf8 => ArrowDataType::LargeUtf8,
            ColumnarType::Float64 => ArrowDataType::Float64,
            ColumnarType::Boolean => ArrowDataType::Boolean,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct UnsupportedArrowType(pub ArrowDataType);

impl std::fmt::Display for UnsupportedArrowType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "unsupported arrow data type: {}", self.0)
    }
}

impl std::error::Error for UnsupportedArrowType {}

impl TryFrom<&ArrowDataType> for ColumnarType {
    type Error = UnsupportedArrowType;

    fn try_from(value: &ArrowDataType) -> Result<Self, Self::Error> {
        match value {
            ArrowDataType::Int64 => Ok(Self::Int64),
            ArrowDataType::Utf8 => Ok(Self::Utf8),
            ArrowDataType::LargeUtf8 => Ok(Self::LargeUtf8),
            ArrowDataType::Float64 => Ok(Self::Float64),
            ArrowDataType::Boolean => Ok(Self::Boolean),
            dt => Err(UnsupportedArrowType(dt.clone())),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct InvalidColumnarType(pub u32);

impl std::fmt::Display for InvalidColumnarType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid columnar type code: {}", self.0)
    }
}

impl std::error::Error for InvalidColumnarType {}

impl TryFrom<u32> for ColumnarType {
    type Error = InvalidColumnarType;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Int64),
            1 => Ok(Self::Utf8),
            2 => Ok(Self::LargeUtf8),
            10 => Ok(Self::Float64),
            11 => Ok(Self::Boolean),
            _ => Err(InvalidColumnarType(value)),
        }
    }
}
