//! Build a v0 `.columnar` file: header, schema, column directory, then fixed-width column values.
//!
//! On-disk section order matches format §2.1: **header → schema → directory → chunks**. The writer
//! reserves the directory after the schema, aligns for the values buffer (64-byte preferred per
//! §1.3), writes payload, then **patches** directory bytes and the header.

use crate::align::align_offset;
use crate::directory::{ColumnMeta, COLUMN_META_LEN};
use crate::header::{
    FileHeader, FILE_HEADER_LEN, FILE_HEADER_MAGIC, FILE_HEADER_ON_DISK_SIZE, FILE_HEADER_VERSION,
};

/// Preferred alignment for fixed-width **values** buffers (format §1.3).
pub const VALUES_BUFFER_ALIGN: usize = 64;

/// Minimum alignment for header, schema tail, and directory placement (format §1.3 / §2.1).
pub const SECTION_ALIGN: usize = 8;

/// Opaque v0 discriminator for an 8-byte fixed-width signed integer column (Arrow Int64-sized).
pub const V0_PHYSICAL_FIXED_WIDTH_I64: u32 = 1;

/// Errors while building a Columnar file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ColumnarWriteError {
    /// Methods were called out of order or the buffer was not in the expected state.
    State(&'static str),
    /// `reserve_column_directory` was given `columns == 0`.
    ZeroColumns,
    /// `patch_column_directory` entry count does not match the reserved size.
    DirectoryColumnMismatch {
        reserved: usize,
        got: usize,
    },
    /// `write_fixed_width_values` when `values.len()` is not a multiple of `element_width`.
    ValuesLengthNotMultiple {
        len: usize,
        element_width: usize,
    },
    /// `element_width == 0` in `write_fixed_width_values`.
    ZeroElementWidth,
    /// `usize` overflow sizing the directory.
    SizeOverflow,
}

impl core::fmt::Display for ColumnarWriteError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ColumnarWriteError::State(s) => write!(f, "{s}"),
            ColumnarWriteError::ZeroColumns => write!(f, "column directory must have at least one column"),
            ColumnarWriteError::DirectoryColumnMismatch { reserved, got } => {
                write!(f, "directory patch: reserved {reserved} columns, got {got}")
            }
            ColumnarWriteError::ValuesLengthNotMultiple { len, element_width } => write!(
                f,
                "values length {len} is not a multiple of element width {element_width}"
            ),
            ColumnarWriteError::ZeroElementWidth => write!(f, "element width must be non-zero"),
            ColumnarWriteError::SizeOverflow => write!(f, "size calculation overflow"),
        }
    }
}

impl std::error::Error for ColumnarWriteError {}

/// Incremental writer producing a contiguous in-memory `.columnar` blob.
#[derive(Debug, Default)]
pub struct ColumnarWriter {
    buf: Vec<u8>,
    schema_raw_len: u64,
    dir_start: usize,
    dir_columns: usize,
    header_written: bool,
    schema_written: bool,
    dir_reserved: bool,
    values_written: bool,
    dir_patched: bool,
    finalized: bool,
}

impl ColumnarWriter {
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            schema_raw_len: 0,
            dir_start: 0,
            dir_columns: 0,
            header_written: false,
            schema_written: false,
            dir_reserved: false,
            values_written: false,
            dir_patched: false,
            finalized: false,
        }
    }

    /// Writes a 64-byte **placeholder** header (zeros). Must be called on an empty writer.
    pub fn write_header_placeholder(&mut self) -> Result<(), ColumnarWriteError> {
        if self.header_written {
            return Err(ColumnarWriteError::State("header placeholder already written"));
        }
        if !self.buf.is_empty() {
            return Err(ColumnarWriteError::State("buffer must be empty before header"));
        }
        self.buf.resize(FILE_HEADER_LEN, 0);
        self.header_written = true;
        Ok(())
    }

    /// Appends the raw Arrow IPC schema bytes, then pads the file to [`SECTION_ALIGN`].
    pub fn write_schema_block(&mut self, schema: &[u8]) -> Result<(), ColumnarWriteError> {
        if !self.header_written {
            return Err(ColumnarWriteError::State("call write_header_placeholder first"));
        }
        if self.schema_written {
            return Err(ColumnarWriteError::State("schema block already written"));
        }
        if self.buf.len() != FILE_HEADER_LEN {
            return Err(ColumnarWriteError::State("corrupt buffer length before schema"));
        }
        self.buf.extend_from_slice(schema);
        self.schema_raw_len = schema.len() as u64;
        let padded = align_offset(self.buf.len(), SECTION_ALIGN);
        self.buf.resize(padded, 0);
        self.schema_written = true;
        Ok(())
    }

    /// Reserves a zero-filled column directory for `columns` entries at the current offset.
    /// The directory begins at an [`SECTION_ALIGN`]-byte aligned offset (enforced by prior padding).
    pub fn reserve_column_directory(&mut self, columns: usize) -> Result<usize, ColumnarWriteError> {
        if !self.schema_written {
            return Err(ColumnarWriteError::State("call write_schema_block first"));
        }
        if self.dir_reserved {
            return Err(ColumnarWriteError::State("column directory already reserved"));
        }
        if columns == 0 {
            return Err(ColumnarWriteError::ZeroColumns);
        }
        if self.buf.len() % SECTION_ALIGN != 0 {
            return Err(ColumnarWriteError::State(
                "buffer not 8-byte aligned before directory reservation",
            ));
        }
        let dir_start = self.buf.len();
        let add = columns
            .checked_mul(COLUMN_META_LEN)
            .ok_or(ColumnarWriteError::SizeOverflow)?;
        self.buf
            .try_reserve(add)
            .map_err(|_| ColumnarWriteError::SizeOverflow)?;
        self.buf.resize(dir_start + add, 0);
        self.dir_start = dir_start;
        self.dir_columns = columns;
        self.dir_reserved = true;
        self.values_written = false;
        self.dir_patched = false;
        Ok(dir_start)
    }

    /// Pads with zeros until the file length is a multiple of `alignment` (power of two).
    pub fn pad_to_alignment(&mut self, alignment: usize) -> Result<(), ColumnarWriteError> {
        if !self.dir_reserved {
            return Err(ColumnarWriteError::State(
                "call reserve_column_directory before padding for column chunk",
            ));
        }
        let padded = align_offset(self.buf.len(), alignment);
        self.buf.resize(padded, 0);
        Ok(())
    }

    /// Appends raw fixed-width values (no validity / offsets / stats). Returns `(data_offset, data_length)`.
    pub fn write_fixed_width_values(
        &mut self,
        values: &[u8],
        element_width: usize,
    ) -> Result<(u64, u64), ColumnarWriteError> {
        if !self.dir_reserved {
            return Err(ColumnarWriteError::State(
                "reserve column directory before writing values",
            ));
        }
        if element_width == 0 {
            return Err(ColumnarWriteError::ZeroElementWidth);
        }
        if values.len() % element_width != 0 {
            return Err(ColumnarWriteError::ValuesLengthNotMultiple {
                len: values.len(),
                element_width,
            });
        }
        let offset = self.buf.len() as u64;
        self.buf.extend_from_slice(values);
        let length = values.len() as u64;
        self.values_written = true;
        Ok((offset, length))
    }

    /// Overwrites the reserved directory region with serialized [`ColumnMeta`] entries.
    pub fn patch_column_directory(&mut self, entries: &[ColumnMeta]) -> Result<(), ColumnarWriteError> {
        if !self.dir_reserved {
            return Err(ColumnarWriteError::State("directory not reserved"));
        }
        if !self.values_written {
            return Err(ColumnarWriteError::State(
                "write_fixed_width_values before patch_column_directory",
            ));
        }
        if entries.len() != self.dir_columns {
            return Err(ColumnarWriteError::DirectoryColumnMismatch {
                reserved: self.dir_columns,
                got: entries.len(),
            });
        }
        let need = self.dir_columns * COLUMN_META_LEN;
        let end = self
            .dir_start
            .checked_add(need)
            .ok_or(ColumnarWriteError::SizeOverflow)?;
        if end > self.buf.len() {
            return Err(ColumnarWriteError::State("directory patch past end of buffer"));
        }
        for (index, entry) in entries.iter().enumerate() {
            let start = self.dir_start + index * COLUMN_META_LEN;
            let entry_end = start + COLUMN_META_LEN;
            entry
                .serialize_into(&mut self.buf[start..entry_end])
                .map_err(|_| ColumnarWriteError::State("directory serialize_into failed"))?;
        }
        self.dir_patched = true;
        Ok(())
    }

    /// Writes the final header at offset 0 (magic, version, offsets, lengths).
    pub fn finalize_header(&mut self) -> Result<(), ColumnarWriteError> {
        if self.finalized {
            return Err(ColumnarWriteError::State("already finalized"));
        }
        if !self.header_written || !self.schema_written || !self.dir_reserved || !self.dir_patched {
            return Err(ColumnarWriteError::State(
                "header, schema, directory reservation, and patch_column_directory are required before finalize",
            ));
        }
        if self.buf.len() < FILE_HEADER_LEN {
            return Err(ColumnarWriteError::State("buffer shorter than header"));
        }

        let dir_byte_len = self
            .dir_columns
            .checked_mul(COLUMN_META_LEN)
            .ok_or(ColumnarWriteError::SizeOverflow)?;
        let column_dir_length: u64 = dir_byte_len
            .try_into()
            .map_err(|_| ColumnarWriteError::SizeOverflow)?;

        let header = FileHeader {
            magic: FILE_HEADER_MAGIC,
            version: FILE_HEADER_VERSION,
            flags: 0,
            header_size: FILE_HEADER_ON_DISK_SIZE,
            schema_offset: FILE_HEADER_LEN as u64,
            schema_length: self.schema_raw_len,
            column_dir_offset: self.dir_start as u64,
            column_dir_length,
            reserved: [0u8; 8],
        };
        header
            .validate()
            .map_err(|_| ColumnarWriteError::State("constructed header failed validate"))?;
        self.buf[0..FILE_HEADER_LEN].copy_from_slice(&header.serialize());
        self.finalized = true;
        Ok(())
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.buf
    }

    #[inline]
    pub fn into_inner(self) -> Vec<u8> {
        self.buf
    }

    #[inline]
    pub fn directory_start_offset(&self) -> usize {
        self.dir_start
    }

    #[inline]
    pub fn directory_column_count(&self) -> usize {
        self.dir_columns
    }
}
