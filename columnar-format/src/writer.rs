//! Build a v0 `.columnar` file: header, schema, column directory, then column buffers.
//!
//! On-disk section order matches format §2.1: **header → schema → directory → chunks**. The writer
//! reserves the directory after the schema, aligns for the values buffer (64-byte preferred per
//! §1.3), writes payload, then **patches** directory bytes and the header.

use crate::align::align_offset;
use crate::directory::{ColumnMeta, COLUMN_META_LEN};
use crate::header::{
    FileHeader, FILE_FLAG_VALUES_ALIGNED_64, FILE_HEADER_LEN, FILE_HEADER_MAGIC,
    FILE_HEADER_ON_DISK_SIZE, FILE_HEADER_VERSION,
};
use crate::stats::Int64Stats;

/// Preferred alignment for fixed-width **values** buffers (format §1.3).
pub const VALUES_BUFFER_ALIGN: usize = 64;

/// Minimum alignment for header, schema tail, and directory placement (format §1.3 / §2.1).
pub const SECTION_ALIGN: usize = 8;

/// Opaque v0 discriminator for an 8-byte fixed-width signed integer column (Arrow Int64-sized).
pub const V0_PHYSICAL_FIXED_WIDTH_I64: u32 = 1;
/// Opaque v0 discriminator for a UTF-8 column with 4-byte offsets (Arrow Utf8).
pub const V0_PHYSICAL_UTF8_I32: u32 = 2;
/// Opaque v0 discriminator for a UTF-8 column with 8-byte offsets (Arrow LargeUtf8).
pub const V0_PHYSICAL_UTF8_I64: u32 = 3;

/// Values-buffer alignment strategy recorded in the file header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ValueAlignmentStrategy {
    #[default]
    Minimum8,
    Align64,
}

impl ValueAlignmentStrategy {
    #[inline]
    pub const fn alignment(self) -> usize {
        match self {
            Self::Minimum8 => SECTION_ALIGN,
            Self::Align64 => VALUES_BUFFER_ALIGN,
        }
    }

    #[inline]
    const fn header_flags(self) -> u16 {
        match self {
            Self::Minimum8 => 0,
            Self::Align64 => FILE_FLAG_VALUES_ALIGNED_64,
        }
    }
}

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
    InvalidValidityLength {
        rows: usize,
        got_bytes: usize,
        needed_bytes: usize,
    },
    InvalidOffsetsLength {
        got: usize,
        offset_width: usize,
    },
    OffsetsTooShort {
        got_bytes: usize,
        needed_bytes: usize,
    },
    OffsetsMustStartAtZero {
        got: i64,
    },
    OffsetsNotMonotonic {
        index: usize,
        previous: i64,
        current: i64,
    },
    OffsetOutOfBounds {
        index: usize,
        offset: i64,
        values_len: usize,
    },
    /// `data_offset` in a directory entry does not satisfy the configured values alignment.
    MisalignedDataOffset {
        column_index: usize,
        offset: u64,
        required_alignment: usize,
    },
}

impl core::fmt::Display for ColumnarWriteError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ColumnarWriteError::State(s) => write!(f, "{s}"),
            ColumnarWriteError::ZeroColumns => {
                write!(f, "column directory must have at least one column")
            }
            ColumnarWriteError::DirectoryColumnMismatch { reserved, got } => {
                write!(f, "directory patch: reserved {reserved} columns, got {got}")
            }
            ColumnarWriteError::ValuesLengthNotMultiple { len, element_width } => write!(
                f,
                "values length {len} is not a multiple of element width {element_width}"
            ),
            ColumnarWriteError::ZeroElementWidth => write!(f, "element width must be non-zero"),
            ColumnarWriteError::SizeOverflow => write!(f, "size calculation overflow"),
            ColumnarWriteError::InvalidValidityLength {
                rows,
                got_bytes,
                needed_bytes,
            } => write!(
                f,
                "validity bitmap for {rows} rows is too short: got {got_bytes} bytes, need at least {needed_bytes}"
            ),
            ColumnarWriteError::InvalidOffsetsLength { got, offset_width } => write!(
                f,
                "offsets length {got} is not a multiple of offset width {offset_width}"
            ),
            ColumnarWriteError::OffsetsTooShort {
                got_bytes,
                needed_bytes,
            } => write!(
                f,
                "offsets buffer is too short: got {got_bytes} bytes, need at least {needed_bytes}"
            ),
            ColumnarWriteError::OffsetsMustStartAtZero { got } => {
                write!(f, "offsets must start at zero, got {got}")
            }
            ColumnarWriteError::OffsetsNotMonotonic {
                index,
                previous,
                current,
            } => write!(
                f,
                "offset {index} is not monotonic: previous {previous}, current {current}"
            ),
            ColumnarWriteError::OffsetOutOfBounds {
                index,
                offset,
                values_len,
            } => write!(
                f,
                "offset {index} value {offset} exceeds values length {values_len}"
            ),
            ColumnarWriteError::MisalignedDataOffset {
                column_index,
                offset,
                required_alignment,
            } => write!(
                f,
                "column {column_index} data offset {offset} is not aligned to {required_alignment}"
            ),
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
    logical_column_count: usize,
    chunk_count: usize,
    values_alignment: ValueAlignmentStrategy,
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
            logical_column_count: 0,
            chunk_count: 0,
            values_alignment: ValueAlignmentStrategy::Minimum8,
            header_written: false,
            schema_written: false,
            dir_reserved: false,
            values_written: false,
            dir_patched: false,
            finalized: false,
        }
    }

    #[inline]
    pub fn with_value_alignment(mut self, values_alignment: ValueAlignmentStrategy) -> Self {
        self.values_alignment = values_alignment;
        self
    }

    #[inline]
    pub fn value_alignment_strategy(&self) -> ValueAlignmentStrategy {
        self.values_alignment
    }

    /// Writes a 64-byte **placeholder** header (zeros). Must be called on an empty writer.
    pub fn write_header_placeholder(&mut self) -> Result<(), ColumnarWriteError> {
        if self.header_written {
            return Err(ColumnarWriteError::State(
                "header placeholder already written",
            ));
        }
        if !self.buf.is_empty() {
            return Err(ColumnarWriteError::State(
                "buffer must be empty before header",
            ));
        }
        self.buf.resize(FILE_HEADER_LEN, 0);
        self.header_written = true;
        Ok(())
    }

    /// Appends the raw Arrow IPC schema bytes, then pads the file to [`SECTION_ALIGN`].
    pub fn write_schema_block(&mut self, schema: &[u8]) -> Result<(), ColumnarWriteError> {
        if !self.header_written {
            return Err(ColumnarWriteError::State(
                "call write_header_placeholder first",
            ));
        }
        if self.schema_written {
            return Err(ColumnarWriteError::State("schema block already written"));
        }
        if self.buf.len() != FILE_HEADER_LEN {
            return Err(ColumnarWriteError::State(
                "corrupt buffer length before schema",
            ));
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
    pub fn reserve_column_directory(
        &mut self,
        columns: usize,
    ) -> Result<usize, ColumnarWriteError> {
        if !self.schema_written {
            return Err(ColumnarWriteError::State("call write_schema_block first"));
        }
        if self.dir_reserved {
            return Err(ColumnarWriteError::State(
                "column directory already reserved",
            ));
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
        self.logical_column_count = 0;
        self.chunk_count = 0;
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

    /// Aligns and appends a column values buffer according to the configured strategy.
    pub fn write_values_buffer(
        &mut self,
        values: &[u8],
        element_width: usize,
    ) -> Result<(u64, u64), ColumnarWriteError> {
        self.pad_to_alignment(self.values_alignment.alignment())?;
        self.write_fixed_width_values(values, element_width)
    }

    /// Writes a single `Int64` column chunk, including optional stats and validity.
    pub fn write_int64_column_chunk(
        &mut self,
        column_id: u32,
        logical_type: u32,
        values: &[u8],
        validity: Option<&[u8]>,
        stats: Option<Int64Stats>,
    ) -> Result<ColumnMeta, ColumnarWriteError> {
        if values.len() % std::mem::size_of::<i64>() != 0 {
            return Err(ColumnarWriteError::ValuesLengthNotMultiple {
                len: values.len(),
                element_width: std::mem::size_of::<i64>(),
            });
        }
        let rows = values.len() / std::mem::size_of::<i64>();
        validate_validity(validity, rows)?;

        let (stats_offset, stats_length) = if let Some(stats) = stats {
            self.write_optional_buffer(Some(&stats.serialize()), std::mem::size_of::<u64>())?
        } else {
            (0, 0)
        };
        let (validity_offset, validity_length) = self.write_optional_buffer(validity, 1)?;
        let (data_offset, data_length) =
            self.write_values_buffer(values, std::mem::size_of::<i64>())?;

        Ok(ColumnMeta {
            column_id,
            physical_type: V0_PHYSICAL_FIXED_WIDTH_I64,
            logical_type,
            data_offset,
            data_length,
            validity_offset,
            validity_length,
            offsets_offset: 0,
            offsets_length: 0,
            stats_offset,
            stats_length,
        })
    }

    /// Writes a single `Utf8` column chunk with 4-byte offsets, plus optional validity and stats.
    pub fn write_utf8_i32_column_chunk(
        &mut self,
        column_id: u32,
        logical_type: u32,
        offsets: &[u8],
        values: &[u8],
        validity: Option<&[u8]>,
        stats: Option<&[u8]>,
    ) -> Result<ColumnMeta, ColumnarWriteError> {
        self.write_utf8_column_chunk(
            column_id,
            logical_type,
            V0_PHYSICAL_UTF8_I32,
            offsets,
            std::mem::size_of::<i32>(),
            values,
            validity,
            stats,
        )
    }

    /// Writes a single `LargeUtf8` column chunk with 8-byte offsets, plus optional validity and stats.
    pub fn write_utf8_i64_column_chunk(
        &mut self,
        column_id: u32,
        logical_type: u32,
        offsets: &[u8],
        values: &[u8],
        validity: Option<&[u8]>,
        stats: Option<&[u8]>,
    ) -> Result<ColumnMeta, ColumnarWriteError> {
        self.write_utf8_column_chunk(
            column_id,
            logical_type,
            V0_PHYSICAL_UTF8_I64,
            offsets,
            std::mem::size_of::<i64>(),
            values,
            validity,
            stats,
        )
    }

    /// Overwrites the reserved directory region with serialized [`ColumnMeta`] entries.
    pub fn patch_column_directory(
        &mut self,
        entries: &[ColumnMeta],
    ) -> Result<(), ColumnarWriteError> {
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
            return Err(ColumnarWriteError::State(
                "directory patch past end of buffer",
            ));
        }
        let required_alignment = self.values_alignment.alignment();
        let (logical_column_count, chunk_count) =
            infer_chunk_layout(entries).map_err(ColumnarWriteError::State)?;
        for (index, entry) in entries.iter().enumerate() {
            if entry.data_length > 0 && entry.data_offset % required_alignment as u64 != 0 {
                return Err(ColumnarWriteError::MisalignedDataOffset {
                    column_index: index,
                    offset: entry.data_offset,
                    required_alignment,
                });
            }
            let start = self.dir_start + index * COLUMN_META_LEN;
            let entry_end = start + COLUMN_META_LEN;
            entry
                .serialize_into(&mut self.buf[start..entry_end])
                .map_err(|_| ColumnarWriteError::State("directory serialize_into failed"))?;
        }
        self.logical_column_count = logical_column_count;
        self.chunk_count = chunk_count;
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
            flags: self.values_alignment.header_flags(),
            header_size: FILE_HEADER_ON_DISK_SIZE,
            schema_offset: FILE_HEADER_LEN as u64,
            schema_length: self.schema_raw_len,
            column_dir_offset: self.dir_start as u64,
            column_dir_length,
            reserved: [0u8; 8],
        }
        .with_chunk_layout(self.logical_column_count as u32, self.chunk_count as u32);
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

    fn write_utf8_column_chunk(
        &mut self,
        column_id: u32,
        logical_type: u32,
        physical_type: u32,
        offsets: &[u8],
        offset_width: usize,
        values: &[u8],
        validity: Option<&[u8]>,
        stats: Option<&[u8]>,
    ) -> Result<ColumnMeta, ColumnarWriteError> {
        let rows = validate_offsets(offsets, offset_width, values.len())?;
        validate_validity(validity, rows)?;

        let (stats_offset, stats_length) =
            self.write_optional_buffer(stats, std::mem::size_of::<u64>())?;
        let (validity_offset, validity_length) = self.write_optional_buffer(validity, 1)?;
        let (offsets_offset, offsets_length) =
            self.write_optional_aligned_buffer(offsets, offset_width)?;
        let (data_offset, data_length) = self.write_values_buffer(values, 1)?;

        Ok(ColumnMeta {
            column_id,
            physical_type,
            logical_type,
            data_offset,
            data_length,
            validity_offset,
            validity_length,
            offsets_offset,
            offsets_length,
            stats_offset,
            stats_length,
        })
    }

    fn write_optional_buffer(
        &mut self,
        bytes: Option<&[u8]>,
        element_width: usize,
    ) -> Result<(u64, u64), ColumnarWriteError> {
        let Some(bytes) = bytes else {
            return Ok((0, 0));
        };
        self.write_optional_aligned_buffer(bytes, element_width)
    }

    fn write_optional_aligned_buffer(
        &mut self,
        bytes: &[u8],
        element_width: usize,
    ) -> Result<(u64, u64), ColumnarWriteError> {
        self.pad_to_alignment(SECTION_ALIGN)?;
        self.write_fixed_width_values(bytes, element_width)
    }
}

fn validate_validity(validity: Option<&[u8]>, rows: usize) -> Result<(), ColumnarWriteError> {
    let Some(validity) = validity else {
        return Ok(());
    };
    let needed_bytes = rows.div_ceil(8);
    if validity.len() < needed_bytes {
        return Err(ColumnarWriteError::InvalidValidityLength {
            rows,
            got_bytes: validity.len(),
            needed_bytes,
        });
    }
    Ok(())
}

fn validate_offsets(
    offsets: &[u8],
    offset_width: usize,
    values_len: usize,
) -> Result<usize, ColumnarWriteError> {
    if offsets.len() % offset_width != 0 {
        return Err(ColumnarWriteError::InvalidOffsetsLength {
            got: offsets.len(),
            offset_width,
        });
    }
    if offsets.len() < offset_width {
        return Err(ColumnarWriteError::OffsetsTooShort {
            got_bytes: offsets.len(),
            needed_bytes: offset_width,
        });
    }

    match offset_width {
        4 => validate_i32_offsets(offsets, values_len),
        8 => validate_i64_offsets(offsets, values_len),
        _ => Err(ColumnarWriteError::State("unsupported offset width")),
    }
}

fn validate_i32_offsets(offsets: &[u8], values_len: usize) -> Result<usize, ColumnarWriteError> {
    let mut chunks = offsets.chunks_exact(std::mem::size_of::<i32>());
    let first = i32::from_le_bytes(
        chunks
            .next()
            .expect("validated offsets length")
            .try_into()
            .expect("i32 chunk"),
    ) as i64;
    if first != 0 {
        return Err(ColumnarWriteError::OffsetsMustStartAtZero { got: first });
    }
    let mut previous = first;
    let mut rows = 0usize;
    for (index, chunk) in chunks.enumerate() {
        let current = i32::from_le_bytes(chunk.try_into().expect("i32 chunk")) as i64;
        if current < previous {
            return Err(ColumnarWriteError::OffsetsNotMonotonic {
                index: index + 1,
                previous,
                current,
            });
        }
        if current as usize > values_len {
            return Err(ColumnarWriteError::OffsetOutOfBounds {
                index: index + 1,
                offset: current,
                values_len,
            });
        }
        previous = current;
        rows += 1;
    }
    Ok(rows)
}

fn validate_i64_offsets(offsets: &[u8], values_len: usize) -> Result<usize, ColumnarWriteError> {
    let mut chunks = offsets.chunks_exact(std::mem::size_of::<i64>());
    let first = i64::from_le_bytes(
        chunks
            .next()
            .expect("validated offsets length")
            .try_into()
            .expect("i64 chunk"),
    );
    if first != 0 {
        return Err(ColumnarWriteError::OffsetsMustStartAtZero { got: first });
    }
    let mut previous = first;
    let mut rows = 0usize;
    for (index, chunk) in chunks.enumerate() {
        let current = i64::from_le_bytes(chunk.try_into().expect("i64 chunk"));
        if current < previous {
            return Err(ColumnarWriteError::OffsetsNotMonotonic {
                index: index + 1,
                previous,
                current,
            });
        }
        if current as usize > values_len {
            return Err(ColumnarWriteError::OffsetOutOfBounds {
                index: index + 1,
                offset: current,
                values_len,
            });
        }
        previous = current;
        rows += 1;
    }
    Ok(rows)
}

fn infer_chunk_layout(entries: &[ColumnMeta]) -> Result<(usize, usize), &'static str> {
    if entries.is_empty() {
        return Err("directory must contain at least one entry");
    }

    let mut max_column_id = 0u32;
    for entry in entries {
        max_column_id = max_column_id.max(entry.column_id);
    }
    let logical_column_count = max_column_id as usize + 1;
    if entries.len() % logical_column_count != 0 {
        return Err("directory entries do not form whole chunks");
    }
    for (index, entry) in entries.iter().enumerate() {
        let expected_column_id = (index % logical_column_count) as u32;
        if entry.column_id != expected_column_id {
            return Err("directory entries are not laid out chunk-major");
        }
    }
    Ok((logical_column_count, entries.len() / logical_column_count))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ColumnarReader, Int64Stats};

    fn encode_i64(values: &[i64]) -> Vec<u8> {
        #[cfg(target_endian = "little")]
        {
            // SAFETY: `i64` has a defined layout, and we are on a little-endian system, so
            // a simple memory copy is correct.
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    values.as_ptr() as *const u8,
                    values.len() * std::mem::size_of::<i64>(),
                )
            };
            bytes.to_vec()
        }

        #[cfg(not(target_endian = "little"))]
        {
            let mut out = Vec::with_capacity(values.len() * std::mem::size_of::<i64>());
            for value in values {
                out.extend_from_slice(&value.to_le_bytes());
            }
            out
        }
    }

    fn encode_i32(values: &[i32]) -> Vec<u8> {
        #[cfg(target_endian = "little")]
        {
            // SAFETY: `i32` has a defined layout, and we are on a little-endian system, so
            // a simple memory copy is correct.
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    values.as_ptr() as *const u8,
                    values.len() * std::mem::size_of::<i32>(),
                )
            };
            bytes.to_vec()
        }

        #[cfg(not(target_endian = "little"))]
        {
            let mut out = Vec::with_capacity(values.len() * std::mem::size_of::<i32>());
            for value in values {
                out.extend_from_slice(&value.to_le_bytes());
            }
            out
        }
    }

    fn encode_validity(mask: &[bool]) -> Vec<u8> {
        let mut bitmap = vec![0u8; mask.len().div_ceil(8)];
        for (index, valid) in mask.iter().copied().enumerate() {
            if valid {
                bitmap[index / 8] |= 1u8 << (index % 8);
            }
        }
        bitmap
    }

    #[test]
    fn write_int64_column_chunk_round_trips_validity_and_stats() {
        let mut writer = ColumnarWriter::new();
        writer.write_header_placeholder().unwrap();
        writer.write_schema_block(b"schema").unwrap();
        writer.reserve_column_directory(1).unwrap();

        let values = encode_i64(&[1, 2, 3]);
        let validity = encode_validity(&[true, false, true]);
        let meta = writer
            .write_int64_column_chunk(
                0,
                0,
                &values,
                Some(&validity),
                Some(Int64Stats {
                    min: Some(1),
                    max: Some(3),
                    null_count: 1,
                    distinct_count: Some(2),
                }),
            )
            .unwrap();

        writer
            .patch_column_directory(std::slice::from_ref(&meta))
            .unwrap();
        writer.finalize_header().unwrap();

        let reader = ColumnarReader::new(writer.as_slice()).unwrap();
        assert_eq!(reader.column_values(0).unwrap(), values.as_slice());
        assert_eq!(
            reader.column_validity(0).unwrap().unwrap(),
            validity.as_slice()
        );
        assert_eq!(
            reader.column_int64_stats(0).unwrap().unwrap(),
            Int64Stats {
                min: Some(1),
                max: Some(3),
                null_count: 1,
                distinct_count: Some(2),
            }
        );
    }

    #[test]
    fn write_utf8_i32_column_chunk_round_trips_offsets_values_and_nulls() {
        let mut writer = ColumnarWriter::new();
        writer.write_header_placeholder().unwrap();
        writer.write_schema_block(b"schema").unwrap();
        writer.reserve_column_directory(1).unwrap();

        let offsets = encode_i32(&[0, 5, 5, 9]);
        let values = b"alphabeta";
        let validity = encode_validity(&[true, false, true]);

        let meta = writer
            .write_utf8_i32_column_chunk(0, 0, &offsets, values, Some(&validity), None)
            .unwrap();

        writer
            .patch_column_directory(std::slice::from_ref(&meta))
            .unwrap();
        writer.finalize_header().unwrap();

        let reader = ColumnarReader::new(writer.as_slice()).unwrap();
        let column = reader.variable_column_buffers(0).unwrap();
        assert_eq!(column.offsets, offsets.as_slice());
        assert_eq!(column.values, values);
        assert_eq!(column.validity.unwrap(), validity.as_slice());
    }

    #[test]
    fn write_utf8_chunk_rejects_out_of_bounds_offsets() {
        let mut writer = ColumnarWriter::new();
        writer.write_header_placeholder().unwrap();
        writer.write_schema_block(b"schema").unwrap();
        writer.reserve_column_directory(1).unwrap();

        let offsets = encode_i32(&[0, 99]);
        let err = writer
            .write_utf8_i32_column_chunk(0, 0, &offsets, b"abc", None, None)
            .unwrap_err();
        assert!(matches!(err, ColumnarWriteError::OffsetOutOfBounds { .. }));
    }
}
