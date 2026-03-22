//! Columnar error types.

use core::fmt;
use crate::types::InvalidColumnarType;

/// The general error type for the Columnar format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnarError {
    kind: ColumnarErrorType,
    message: String,
}

impl ColumnarError {
    pub fn new(kind: ColumnarErrorType, message: String) -> Self {
        Self { kind, message }
    }
}

impl fmt::Display for ColumnarError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for ColumnarError {}

impl From<InvalidColumnarType> for ColumnarError {
    fn from(value: InvalidColumnarType) -> Self {
        ColumnarError::new(ColumnarErrorType::Corrupt, value.to_string())
    }
}

/// The type of a [`ColumnarError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnarErrorType {
    /// The file is corrupt.
    Corrupt,
    /// The file is not a valid Columnar file.
    Invalid,
    /// The operation is not supported.
    Unsupported,
    /// An IO error occurred.
    Io,
}

/// A convenience wrapper for `Result<T, ColumnarError>`.
pub type Result<T, E = ColumnarError> = std::result::Result<T, E>;

#[cfg(feature = "arrow-conversion")]
mod arrow_conversion {
    use super::*;
    use crate::types::UnsupportedArrowType;

    impl From<UnsupportedArrowType> for ColumnarError {
        fn from(value: UnsupportedArrowType) -> Self {
            ColumnarError::new(ColumnarErrorType::Unsupported, value.to_string())
        }
    }
}
