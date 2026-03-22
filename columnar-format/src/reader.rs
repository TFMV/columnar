//! Zero-copy read access to a Columnar file backed by a single byte slice (e.g. mmap).

use core::fmt;

use crate::directory::{
    BufferField, ColumnDirectoryError, ColumnDirectoryView, ColumnMeta, COLUMN_META_LEN,
    MIN_BUFFER_ALIGN,
};
use crate::header::{FileHeader, FileHeaderError, FILE_HEADER_LEN};
use crate::{V0_PHYSICAL_FIXED_WIDTH_I64, V0_PHYSICAL_UTF8_I32, V0_PHYSICAL_UTF8_I64};

/// Parsed file with all payload views borrowing `bytes` (typically memory-mapped).
#[derive(Debug, Clone, Copy)]
pub struct ColumnarReader<'a> {
    bytes: &'a [u8],
    header: FileHeader,
    directory: ColumnDirectoryView<'a>,
    logical_column_count: usize,
    chunk_count: usize,
    schema_start: usize,
    schema_end: usize,
    dir_start: usize,
    dir_end: usize,
}

/// Per-column buffer views into the original file slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColumnBufferSlices<'a> {
    pub values: &'a [u8],
    pub validity: Option<&'a [u8]>,
    pub offsets: Option<&'a [u8]>,
    pub stats: Option<&'a [u8]>,
}

/// Per-column buffer views for a variable-length column, where offsets are mandatory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VariableColumnBufferSlices<'a> {
    pub values: &'a [u8],
    pub validity: Option<&'a [u8]>,
    pub offsets: &'a [u8],
    pub stats: Option<&'a [u8]>,
}

/// Errors from opening or walking a Columnar file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ColumnarReadError {
    TooSmall {
        need: usize,
        got: usize,
    },
    Header(FileHeaderError),
    Directory(ColumnDirectoryError),
    RangeOverflow,
    FileTooLargeForPlatform,
    OutOfBounds {
        offset: u64,
        length: u64,
        file_len: u64,
    },
    SchemaOffsetTooSmall {
        offset: u64,
        min: u64,
    },
    SchemaStartNotAligned {
        offset: u64,
    },
    DirectoryStartNotAligned {
        offset: u64,
    },
    SectionOrdering {
        msg: &'static str,
    },
    UnalignedBuffer {
        column_index: usize,
        field: BufferField,
        offset: u64,
        required_alignment: u64,
    },
    BufferOverlapsStructure {
        column_index: usize,
        field: BufferField,
        start: u64,
        end: u64,
    },
    MissingOffsets {
        column_index: usize,
    },
    EmptyDirectory,
    InvalidChunkLayout {
        entry_index: usize,
        expected_column_id: u32,
        got_column_id: u32,
    },
    IncompleteChunkSet {
        entries: usize,
        logical_columns: usize,
    },
    ChunkRowCountMismatch {
        chunk_index: usize,
        expected_rows: usize,
        column_index: usize,
        got_rows: usize,
    },
    UnsupportedPhysicalType {
        column_index: usize,
        physical_type: u32,
    },
}

impl fmt::Display for ColumnarReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ColumnarReadError::TooSmall { need, got } => {
                write!(f, "file too small: need at least {need} bytes, got {got}")
            }
            ColumnarReadError::Header(e) => write!(f, "header: {e}"),
            ColumnarReadError::Directory(e) => write!(f, "directory: {e}"),
            ColumnarReadError::RangeOverflow => write!(f, "offset/length overflow"),
            ColumnarReadError::FileTooLargeForPlatform => {
                write!(
                    f,
                    "file size does not fit in address space on this platform"
                )
            }
            ColumnarReadError::OutOfBounds {
                offset,
                length,
                file_len,
            } => write!(
                f,
                "range [{offset}, {}) exceeds file length {file_len}",
                offset.saturating_add(*length)
            ),
            ColumnarReadError::SchemaOffsetTooSmall { offset, min } => {
                write!(f, "schema_offset {offset} is before end of header ({min})")
            }
            ColumnarReadError::SchemaStartNotAligned { offset } => {
                write!(
                    f,
                    "schema_offset {offset} is not aligned to {MIN_BUFFER_ALIGN}"
                )
            }
            ColumnarReadError::DirectoryStartNotAligned { offset } => {
                write!(
                    f,
                    "column_dir_offset {offset} is not aligned to {MIN_BUFFER_ALIGN}"
                )
            }
            ColumnarReadError::SectionOrdering { msg } => write!(f, "{msg}"),
            ColumnarReadError::UnalignedBuffer {
                column_index,
                field,
                offset,
                required_alignment,
            } => write!(
                f,
                "column {column_index} {field} offset {offset} is not aligned to {required_alignment}"
            ),
            ColumnarReadError::BufferOverlapsStructure {
                column_index,
                field,
                start,
                end,
            } => write!(
                f,
                "column {column_index} {field} range [{start}, {end}) overlaps a structural region"
            ),
            ColumnarReadError::MissingOffsets { column_index } => {
                write!(
                    f,
                    "column {column_index} is missing required offsets buffer"
                )
            }
            ColumnarReadError::EmptyDirectory => {
                write!(f, "column directory must contain at least one entry")
            }
            ColumnarReadError::InvalidChunkLayout {
                entry_index,
                expected_column_id,
                got_column_id,
            } => write!(
                f,
                "directory entry {entry_index} expected column_id {expected_column_id}, got {got_column_id}"
            ),
            ColumnarReadError::IncompleteChunkSet {
                entries,
                logical_columns,
            } => write!(
                f,
                "directory has {entries} entries, which is not a whole number of {logical_columns}-column chunks"
            ),
            ColumnarReadError::ChunkRowCountMismatch {
                chunk_index,
                expected_rows,
                column_index,
                got_rows,
            } => write!(
                f,
                "chunk {chunk_index} expected {expected_rows} rows, but column {column_index} has {got_rows}"
            ),
            ColumnarReadError::UnsupportedPhysicalType {
                column_index,
                physical_type,
            } => write!(
                f,
                "column {column_index} uses unsupported physical type {physical_type}"
            ),
        }
    }
}

impl std::error::Error for ColumnarReadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ColumnarReadError::Header(e) => Some(e),
            _ => None,
        }
    }
}

impl From<FileHeaderError> for ColumnarReadError {
    fn from(value: FileHeaderError) -> Self {
        ColumnarReadError::Header(value)
    }
}

impl From<ColumnDirectoryError> for ColumnarReadError {
    fn from(value: ColumnDirectoryError) -> Self {
        ColumnarReadError::Directory(value)
    }
}

impl<'a> ColumnarReader<'a> {
    /// Parse `bytes` (e.g. `MmapFile::as_slice()`), validating header, schema/directory placement,
    /// directory buffer graph (bounds, alignment, non-overlap), and isolation from structural regions.
    pub fn new(bytes: &'a [u8]) -> Result<Self, ColumnarReadError> {
        if bytes.len() < FILE_HEADER_LEN {
            return Err(ColumnarReadError::TooSmall {
                need: FILE_HEADER_LEN,
                got: bytes.len(),
            });
        }
        let header = FileHeader::deserialize(&bytes[..FILE_HEADER_LEN])?;
        header.validate()?;

        let file_len =
            u64::try_from(bytes.len()).map_err(|_| ColumnarReadError::FileTooLargeForPlatform)?;

        let (schema_start, schema_end) =
            u64_range_to_usize(header.schema_offset, header.schema_length, file_len)?;

        if header.schema_offset < FILE_HEADER_LEN as u64 {
            return Err(ColumnarReadError::SchemaOffsetTooSmall {
                offset: header.schema_offset,
                min: FILE_HEADER_LEN as u64,
            });
        }
        if header.schema_offset % MIN_BUFFER_ALIGN != 0 {
            return Err(ColumnarReadError::SchemaStartNotAligned {
                offset: header.schema_offset,
            });
        }

        let (dir_start, dir_end) =
            u64_range_to_usize(header.column_dir_offset, header.column_dir_length, file_len)?;

        if header.column_dir_offset % MIN_BUFFER_ALIGN != 0 {
            return Err(ColumnarReadError::DirectoryStartNotAligned {
                offset: header.column_dir_offset,
            });
        }

        if (dir_end - dir_start) % COLUMN_META_LEN != 0 {
            return Err(ColumnarReadError::Directory(
                ColumnDirectoryError::WrongDirectoryLength {
                    got: dir_end - dir_start,
                    multiple_of: COLUMN_META_LEN,
                },
            ));
        }

        // Layout: schema payload precedes column directory (format §2.1).
        let schema_payload_end = header
            .schema_offset
            .checked_add(header.schema_length)
            .ok_or(ColumnarReadError::RangeOverflow)?;
        if header.column_dir_offset < schema_payload_end {
            return Err(ColumnarReadError::SectionOrdering {
                msg: "column_dir_offset must not start before end of schema payload",
            });
        }

        let directory = ColumnDirectoryView::new(&bytes[dir_start..dir_end])?;
        directory.validate(file_len)?;

        let (logical_column_count, chunk_count) = validate_chunk_layout(header, directory)?;

        let reader = Self {
            bytes,
            header,
            directory,
            logical_column_count,
            chunk_count,
            schema_start,
            schema_end,
            dir_start,
            dir_end,
        };
        reader.validate_buffers_vs_structure(file_len)?;
        reader.validate_chunk_row_counts()?;
        Ok(reader)
    }

    #[inline]
    pub fn header(&self) -> &FileHeader {
        &self.header
    }

    /// Raw Arrow IPC schema bytes (format §4). Borrows the underlying file/mmap slice.
    #[inline]
    pub fn schema_bytes(&self) -> &'a [u8] {
        &self.bytes[self.schema_start..self.schema_end]
    }

    #[inline]
    pub fn directory_view(&self) -> ColumnDirectoryView<'a> {
        self.directory
    }

    #[inline]
    pub fn column_count(&self) -> usize {
        self.logical_column_count
    }

    #[inline]
    pub fn chunk_count(&self) -> usize {
        self.chunk_count
    }

    #[inline]
    pub fn column_meta(&self, index: usize) -> Result<ColumnMeta, ColumnarReadError> {
        self.chunk_column_meta(0, index)
    }

    #[inline]
    pub fn chunk_column_meta(
        &self,
        chunk_index: usize,
        column_index: usize,
    ) -> Result<ColumnMeta, ColumnarReadError> {
        let entry_index = self.entry_index(chunk_index, column_index)?;
        Ok(self.directory.get(entry_index)?)
    }

    /// Values buffer for column `index` (may be empty if `data_length == 0`).
    pub fn column_values(&self, index: usize) -> Result<&'a [u8], ColumnarReadError> {
        let m = self.column_meta(index)?;
        self.slice_buffer(index, BufferField::Data, m.data_offset, m.data_length)
    }

    pub fn chunk_column_values(
        &self,
        chunk_index: usize,
        column_index: usize,
    ) -> Result<&'a [u8], ColumnarReadError> {
        let m = self.chunk_column_meta(chunk_index, column_index)?;
        self.slice_buffer(
            column_index,
            BufferField::Data,
            m.data_offset,
            m.data_length,
        )
    }

    pub fn column_validity(&self, index: usize) -> Result<Option<&'a [u8]>, ColumnarReadError> {
        let m = self.column_meta(index)?;
        self.optional_slice_buffer(
            index,
            BufferField::Validity,
            m.validity_offset,
            m.validity_length,
        )
    }

    pub fn chunk_column_validity(
        &self,
        chunk_index: usize,
        column_index: usize,
    ) -> Result<Option<&'a [u8]>, ColumnarReadError> {
        let m = self.chunk_column_meta(chunk_index, column_index)?;
        self.optional_slice_buffer(
            column_index,
            BufferField::Validity,
            m.validity_offset,
            m.validity_length,
        )
    }

    pub fn column_offsets(&self, index: usize) -> Result<Option<&'a [u8]>, ColumnarReadError> {
        let m = self.column_meta(index)?;
        self.optional_slice_buffer(
            index,
            BufferField::Offsets,
            m.offsets_offset,
            m.offsets_length,
        )
    }

    pub fn chunk_column_offsets(
        &self,
        chunk_index: usize,
        column_index: usize,
    ) -> Result<Option<&'a [u8]>, ColumnarReadError> {
        let m = self.chunk_column_meta(chunk_index, column_index)?;
        self.optional_slice_buffer(
            column_index,
            BufferField::Offsets,
            m.offsets_offset,
            m.offsets_length,
        )
    }

    pub fn column_stats(&self, index: usize) -> Result<Option<&'a [u8]>, ColumnarReadError> {
        let m = self.column_meta(index)?;
        self.optional_slice_buffer(index, BufferField::Stats, m.stats_offset, m.stats_length)
    }

    pub fn chunk_column_stats(
        &self,
        chunk_index: usize,
        column_index: usize,
    ) -> Result<Option<&'a [u8]>, ColumnarReadError> {
        let m = self.chunk_column_meta(chunk_index, column_index)?;
        self.optional_slice_buffer(
            column_index,
            BufferField::Stats,
            m.stats_offset,
            m.stats_length,
        )
    }

    pub fn column_buffers(
        &self,
        index: usize,
    ) -> Result<ColumnBufferSlices<'a>, ColumnarReadError> {
        Ok(ColumnBufferSlices {
            values: self.column_values(index)?,
            validity: self.column_validity(index)?,
            offsets: self.column_offsets(index)?,
            stats: self.column_stats(index)?,
        })
    }

    pub fn chunk_column_buffers(
        &self,
        chunk_index: usize,
        column_index: usize,
    ) -> Result<ColumnBufferSlices<'a>, ColumnarReadError> {
        Ok(ColumnBufferSlices {
            values: self.chunk_column_values(chunk_index, column_index)?,
            validity: self.chunk_column_validity(chunk_index, column_index)?,
            offsets: self.chunk_column_offsets(chunk_index, column_index)?,
            stats: self.chunk_column_stats(chunk_index, column_index)?,
        })
    }

    /// Buffer views for a variable-length column. Offsets are required and borrowed from the file.
    pub fn variable_column_buffers(
        &self,
        index: usize,
    ) -> Result<VariableColumnBufferSlices<'a>, ColumnarReadError> {
        let offsets = self
            .column_offsets(index)?
            .ok_or(ColumnarReadError::MissingOffsets {
                column_index: index,
            })?;
        Ok(VariableColumnBufferSlices {
            values: self.column_values(index)?,
            validity: self.column_validity(index)?,
            offsets,
            stats: self.column_stats(index)?,
        })
    }

    pub fn chunk_variable_column_buffers(
        &self,
        chunk_index: usize,
        column_index: usize,
    ) -> Result<VariableColumnBufferSlices<'a>, ColumnarReadError> {
        let offsets = self
            .chunk_column_offsets(chunk_index, column_index)?
            .ok_or(ColumnarReadError::MissingOffsets { column_index })?;
        Ok(VariableColumnBufferSlices {
            values: self.chunk_column_values(chunk_index, column_index)?,
            validity: self.chunk_column_validity(chunk_index, column_index)?,
            offsets,
            stats: self.chunk_column_stats(chunk_index, column_index)?,
        })
    }

    #[inline]
    pub fn file_bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Byte range `[start, end)` of the schema payload in the file.
    #[inline]
    pub fn schema_range(&self) -> (usize, usize) {
        (self.schema_start, self.schema_end)
    }

    /// Byte range `[start, end)` of the on-disk column directory.
    #[inline]
    pub fn directory_range(&self) -> (usize, usize) {
        (self.dir_start, self.dir_end)
    }

    fn validate_buffers_vs_structure(&self, file_len: u64) -> Result<(), ColumnarReadError> {
        let hdr_end = FILE_HEADER_LEN as u64;
        let schema0 = self.header.schema_offset;
        let schema1 = schema0.saturating_add(self.header.schema_length);
        let dir0 = self.header.column_dir_offset;
        let dir1 = dir0.saturating_add(self.header.column_dir_length);

        let structural = [(0u64, hdr_end), (schema0, schema1), (dir0, dir1)];

        for i in 0..self.directory.len() {
            let m = self.directory.get(i)?;
            for (field, off, len) in [
                (BufferField::Data, m.data_offset, m.data_length),
                (BufferField::Validity, m.validity_offset, m.validity_length),
                (BufferField::Offsets, m.offsets_offset, m.offsets_length),
                (BufferField::Stats, m.stats_offset, m.stats_length),
            ] {
                if len == 0 {
                    continue;
                }
                let end = off
                    .checked_add(len)
                    .ok_or(ColumnarReadError::RangeOverflow)?;
                if end > file_len {
                    return Err(ColumnarReadError::OutOfBounds {
                        offset: off,
                        length: len,
                        file_len,
                    });
                }
                let required_alignment = self.required_alignment_for(field);
                if off % required_alignment != 0 {
                    return Err(ColumnarReadError::UnalignedBuffer {
                        column_index: i,
                        field,
                        offset: off,
                        required_alignment,
                    });
                }
                for (s0, s1) in structural {
                    if ranges_overlap(off, end, s0, s1) {
                        return Err(ColumnarReadError::BufferOverlapsStructure {
                            column_index: i,
                            field,
                            start: off,
                            end,
                        });
                    }
                }
            }
        }
        Ok(())
    }

    fn slice_buffer(
        &self,
        column_index: usize,
        field: BufferField,
        offset: u64,
        length: u64,
    ) -> Result<&'a [u8], ColumnarReadError> {
        if length == 0 {
            return Ok(&[]);
        }
        let (start, end) = usize_range(offset, length, self.bytes.len())?;
        let required_alignment = self.required_alignment_for(field);
        if offset % required_alignment != 0 {
            return Err(ColumnarReadError::UnalignedBuffer {
                column_index,
                field,
                offset,
                required_alignment,
            });
        }
        Ok(&self.bytes[start..end])
    }

    fn optional_slice_buffer(
        &self,
        column_index: usize,
        field: BufferField,
        offset: u64,
        length: u64,
    ) -> Result<Option<&'a [u8]>, ColumnarReadError> {
        if length == 0 {
            return Ok(None);
        }
        Ok(Some(self.slice_buffer(
            column_index,
            field,
            offset,
            length,
        )?))
    }

    fn entry_index(
        &self,
        chunk_index: usize,
        column_index: usize,
    ) -> Result<usize, ColumnarReadError> {
        if column_index >= self.logical_column_count {
            return Err(ColumnarReadError::Directory(
                ColumnDirectoryError::InvalidColumnIndex {
                    index: column_index,
                    len: self.logical_column_count,
                },
            ));
        }
        if chunk_index >= self.chunk_count {
            return Err(ColumnarReadError::Directory(
                ColumnDirectoryError::InvalidColumnIndex {
                    index: chunk_index,
                    len: self.chunk_count,
                },
            ));
        }
        Ok(chunk_index * self.logical_column_count + column_index)
    }

    #[inline]
    fn required_alignment_for(&self, field: BufferField) -> u64 {
        match field {
            BufferField::Data if self.header.values_are_64_aligned() => 64,
            _ => MIN_BUFFER_ALIGN,
        }
    }

    fn validate_chunk_row_counts(&self) -> Result<(), ColumnarReadError> {
        for chunk_index in 0..self.chunk_count {
            let expected_rows = self.chunk_column_row_count(chunk_index, 0)?;
            for column_index in 1..self.logical_column_count {
                let got_rows = self.chunk_column_row_count(chunk_index, column_index)?;
                if got_rows != expected_rows {
                    return Err(ColumnarReadError::ChunkRowCountMismatch {
                        chunk_index,
                        expected_rows,
                        column_index,
                        got_rows,
                    });
                }
            }
        }
        Ok(())
    }

    #[inline]
    pub fn chunk_row_count(&self, chunk_index: usize) -> Result<usize, ColumnarReadError> {
        self.chunk_column_row_count(chunk_index, 0)
    }

    fn chunk_column_row_count(
        &self,
        chunk_index: usize,
        column_index: usize,
    ) -> Result<usize, ColumnarReadError> {
        let meta = self.chunk_column_meta(chunk_index, column_index)?;
        match meta.physical_type {
            V0_PHYSICAL_FIXED_WIDTH_I64 => Ok((meta.data_length / 8) as usize),
            V0_PHYSICAL_UTF8_I32 => {
                let offsets_len = meta.offsets_length as usize;
                if offsets_len < std::mem::size_of::<i32>() {
                    return Ok(0);
                }
                Ok(offsets_len / std::mem::size_of::<i32>() - 1)
            }
            V0_PHYSICAL_UTF8_I64 => {
                let offsets_len = meta.offsets_length as usize;
                if offsets_len < std::mem::size_of::<i64>() {
                    return Ok(0);
                }
                Ok(offsets_len / std::mem::size_of::<i64>() - 1)
            }
            physical_type => Err(ColumnarReadError::UnsupportedPhysicalType {
                column_index,
                physical_type,
            }),
        }
    }
}

fn validate_chunk_layout(
    header: FileHeader,
    directory: ColumnDirectoryView<'_>,
) -> Result<(usize, usize), ColumnarReadError> {
    if directory.is_empty() {
        return Err(ColumnarReadError::EmptyDirectory);
    }

    let (logical_column_count, chunk_count) =
        if header.logical_column_count() > 0 && header.chunk_count() > 0 {
            (
                header.logical_column_count() as usize,
                header.chunk_count() as usize,
            )
        } else {
            let mut max_column_id = 0u32;
            for entry_index in 0..directory.len() {
                max_column_id = max_column_id.max(directory.get(entry_index)?.column_id);
            }
            let logical_column_count = max_column_id as usize + 1;
            if directory.len() % logical_column_count != 0 {
                return Err(ColumnarReadError::IncompleteChunkSet {
                    entries: directory.len(),
                    logical_columns: logical_column_count,
                });
            }
            (logical_column_count, directory.len() / logical_column_count)
        };

    if directory.len() != logical_column_count * chunk_count {
        return Err(ColumnarReadError::IncompleteChunkSet {
            entries: directory.len(),
            logical_columns: logical_column_count,
        });
    }

    for entry_index in 0..directory.len() {
        let expected_column_id = (entry_index % logical_column_count) as u32;
        let got_column_id = directory.get(entry_index)?.column_id;
        if got_column_id != expected_column_id {
            return Err(ColumnarReadError::InvalidChunkLayout {
                entry_index,
                expected_column_id,
                got_column_id,
            });
        }
    }

    Ok((logical_column_count, chunk_count))
}

fn ranges_overlap(a0: u64, a1: u64, b0: u64, b1: u64) -> bool {
    a0 < b1 && b0 < a1
}

fn u64_range_to_usize(
    offset: u64,
    length: u64,
    file_len: u64,
) -> Result<(usize, usize), ColumnarReadError> {
    let end_u = offset
        .checked_add(length)
        .ok_or(ColumnarReadError::RangeOverflow)?;
    if end_u > file_len {
        return Err(ColumnarReadError::OutOfBounds {
            offset,
            length,
            file_len,
        });
    }
    let start = usize::try_from(offset).map_err(|_| ColumnarReadError::FileTooLargeForPlatform)?;
    let end = usize::try_from(end_u).map_err(|_| ColumnarReadError::FileTooLargeForPlatform)?;
    Ok((start, end))
}

fn usize_range(
    offset: u64,
    length: u64,
    file_len: usize,
) -> Result<(usize, usize), ColumnarReadError> {
    let file_len_u64 =
        u64::try_from(file_len).map_err(|_| ColumnarReadError::FileTooLargeForPlatform)?;
    let (start, end) = u64_range_to_usize(offset, length, file_len_u64)?;
    Ok((start, end))
}

#[cfg(test)]
mod tests {
    use super::ColumnarReader;
    use crate::{
        ColumnMeta, ColumnarWriter, FileHeader, ValueAlignmentStrategy, FILE_HEADER_LEN,
        MIN_BUFFER_ALIGN, V0_PHYSICAL_FIXED_WIDTH_I64, V0_PHYSICAL_UTF8_I32, VALUES_BUFFER_ALIGN,
    };

    #[test]
    fn writer_roundtrip_zero_copy_views() {
        let schema = b"ipc-schema-bytes";
        let mut w = ColumnarWriter::new();
        w.write_header_placeholder().unwrap();
        w.write_schema_block(schema).unwrap();
        w.reserve_column_directory(1).unwrap();
        w.pad_to_alignment(VALUES_BUFFER_ALIGN).unwrap();
        let values: Vec<u8> = (1i64..=3).flat_map(|v| v.to_le_bytes()).collect();
        let (data_off, data_len) = w.write_fixed_width_values(&values, 8).unwrap();
        let meta = ColumnMeta {
            column_id: 0,
            physical_type: V0_PHYSICAL_FIXED_WIDTH_I64,
            logical_type: 0,
            data_offset: data_off,
            data_length: data_len,
            validity_offset: 0,
            validity_length: 0,
            offsets_offset: 0,
            offsets_length: 0,
            stats_offset: 0,
            stats_length: 0,
        };
        w.patch_column_directory(std::slice::from_ref(&meta))
            .unwrap();
        w.finalize_header().unwrap();
        let file = w.into_inner();

        let r = ColumnarReader::new(&file).expect("parse");
        assert_eq!(r.header().magic, *b"COLUMNAR");
        assert_eq!(r.schema_bytes(), schema);
        assert_eq!(
            r.schema_bytes().as_ptr(),
            file[r.header().schema_offset as usize..].as_ptr()
        );
        assert_eq!(r.column_count(), 1);
        assert_eq!(r.column_values(0).unwrap(), values.as_slice());
        assert_eq!(
            r.column_values(0).unwrap().as_ptr(),
            file[data_off as usize..].as_ptr()
        );
    }

    #[test]
    fn multi_chunk_reader_exposes_chunked_column_views() {
        let schema = b"ipc-schema-bytes";
        let mut w = ColumnarWriter::new();
        w.write_header_placeholder().unwrap();
        w.write_schema_block(schema).unwrap();
        w.reserve_column_directory(4).unwrap();

        let chunk0_col0: Vec<u8> = (1i64..=2).flat_map(|v| v.to_le_bytes()).collect();
        let chunk0_col1: Vec<u8> = (10i64..=11).flat_map(|v| v.to_le_bytes()).collect();
        let chunk1_col0: Vec<u8> = (3i64..=4).flat_map(|v| v.to_le_bytes()).collect();
        let chunk1_col1: Vec<u8> = (12i64..=13).flat_map(|v| v.to_le_bytes()).collect();

        w.pad_to_alignment(VALUES_BUFFER_ALIGN).unwrap();
        let (chunk0_col0_offset, chunk0_col0_len) =
            w.write_fixed_width_values(&chunk0_col0, 8).unwrap();
        w.pad_to_alignment(VALUES_BUFFER_ALIGN).unwrap();
        let (chunk0_col1_offset, chunk0_col1_len) =
            w.write_fixed_width_values(&chunk0_col1, 8).unwrap();
        w.pad_to_alignment(VALUES_BUFFER_ALIGN).unwrap();
        let (chunk1_col0_offset, chunk1_col0_len) =
            w.write_fixed_width_values(&chunk1_col0, 8).unwrap();
        w.pad_to_alignment(VALUES_BUFFER_ALIGN).unwrap();
        let (chunk1_col1_offset, chunk1_col1_len) =
            w.write_fixed_width_values(&chunk1_col1, 8).unwrap();

        let metas = [
            ColumnMeta {
                column_id: 0,
                physical_type: V0_PHYSICAL_FIXED_WIDTH_I64,
                logical_type: 0,
                data_offset: chunk0_col0_offset,
                data_length: chunk0_col0_len,
                validity_offset: 0,
                validity_length: 0,
                offsets_offset: 0,
                offsets_length: 0,
                stats_offset: 0,
                stats_length: 0,
            },
            ColumnMeta {
                column_id: 1,
                physical_type: V0_PHYSICAL_FIXED_WIDTH_I64,
                logical_type: 0,
                data_offset: chunk0_col1_offset,
                data_length: chunk0_col1_len,
                validity_offset: 0,
                validity_length: 0,
                offsets_offset: 0,
                offsets_length: 0,
                stats_offset: 0,
                stats_length: 0,
            },
            ColumnMeta {
                column_id: 0,
                physical_type: V0_PHYSICAL_FIXED_WIDTH_I64,
                logical_type: 0,
                data_offset: chunk1_col0_offset,
                data_length: chunk1_col0_len,
                validity_offset: 0,
                validity_length: 0,
                offsets_offset: 0,
                offsets_length: 0,
                stats_offset: 0,
                stats_length: 0,
            },
            ColumnMeta {
                column_id: 1,
                physical_type: V0_PHYSICAL_FIXED_WIDTH_I64,
                logical_type: 0,
                data_offset: chunk1_col1_offset,
                data_length: chunk1_col1_len,
                validity_offset: 0,
                validity_length: 0,
                offsets_offset: 0,
                offsets_length: 0,
                stats_offset: 0,
                stats_length: 0,
            },
        ];

        w.patch_column_directory(&metas).unwrap();
        w.finalize_header().unwrap();
        let file = w.into_inner();

        let r = ColumnarReader::new(&file).unwrap();
        assert_eq!(r.column_count(), 2);
        assert_eq!(r.chunk_count(), 2);
        assert_eq!(r.chunk_column_values(0, 0).unwrap(), chunk0_col0.as_slice());
        assert_eq!(r.chunk_column_values(0, 1).unwrap(), chunk0_col1.as_slice());
        assert_eq!(r.chunk_column_values(1, 0).unwrap(), chunk1_col0.as_slice());
        assert_eq!(r.chunk_column_values(1, 1).unwrap(), chunk1_col1.as_slice());
        assert_eq!(r.header().logical_column_count(), 2);
        assert_eq!(r.header().chunk_count(), 2);
    }

    #[test]
    fn mismatched_chunk_row_counts_are_rejected_by_reader() {
        let schema = b"ipc-schema-bytes";
        let mut w = ColumnarWriter::new();
        w.write_header_placeholder().unwrap();
        w.write_schema_block(schema).unwrap();
        w.reserve_column_directory(2).unwrap();

        let col0: Vec<u8> = (1i64..=2).flat_map(|v| v.to_le_bytes()).collect();
        let col1: Vec<u8> = (10i64..=12).flat_map(|v| v.to_le_bytes()).collect();
        let (col0_offset, col0_len) = w.write_values_buffer(&col0, 8).unwrap();
        let (col1_offset, col1_len) = w.write_values_buffer(&col1, 8).unwrap();

        let metas = [
            ColumnMeta {
                column_id: 0,
                physical_type: V0_PHYSICAL_FIXED_WIDTH_I64,
                logical_type: 0,
                data_offset: col0_offset,
                data_length: col0_len,
                validity_offset: 0,
                validity_length: 0,
                offsets_offset: 0,
                offsets_length: 0,
                stats_offset: 0,
                stats_length: 0,
            },
            ColumnMeta {
                column_id: 1,
                physical_type: V0_PHYSICAL_FIXED_WIDTH_I64,
                logical_type: 0,
                data_offset: col1_offset,
                data_length: col1_len,
                validity_offset: 0,
                validity_length: 0,
                offsets_offset: 0,
                offsets_length: 0,
                stats_offset: 0,
                stats_length: 0,
            },
        ];
        w.patch_column_directory(&metas).unwrap();
        w.finalize_header().unwrap();

        let err = ColumnarReader::new(&w.into_inner()).unwrap_err();
        assert!(matches!(
            err,
            super::ColumnarReadError::ChunkRowCountMismatch {
                chunk_index: 0,
                expected_rows: 2,
                column_index: 1,
                got_rows: 3,
            }
        ));
    }

    #[test]
    fn writer_can_flag_64_byte_values_alignment() {
        let schema = b"ipc-schema-bytes";
        let mut w = ColumnarWriter::new().with_value_alignment(ValueAlignmentStrategy::Align64);
        w.write_header_placeholder().unwrap();
        w.write_schema_block(schema).unwrap();
        w.reserve_column_directory(1).unwrap();
        let values: Vec<u8> = (1i64..=4).flat_map(|v| v.to_le_bytes()).collect();
        let (data_off, data_len) = w.write_values_buffer(&values, 8).unwrap();
        let meta = ColumnMeta {
            column_id: 0,
            physical_type: V0_PHYSICAL_FIXED_WIDTH_I64,
            logical_type: 0,
            data_offset: data_off,
            data_length: data_len,
            validity_offset: 0,
            validity_length: 0,
            offsets_offset: 0,
            offsets_length: 0,
            stats_offset: 0,
            stats_length: 0,
        };
        w.patch_column_directory(std::slice::from_ref(&meta))
            .unwrap();
        w.finalize_header().unwrap();
        let file = w.into_inner();

        let reader = ColumnarReader::new(&file).unwrap();
        assert!(reader.header().values_are_64_aligned());
        assert_eq!(data_off % 64, 0);
        assert_eq!(reader.column_values(0).unwrap().as_ptr() as usize % 8, 0);
    }

    #[test]
    fn legacy_8_byte_values_alignment_remains_valid() {
        let schema = b"ipc-schema-bytes";
        let mut w = ColumnarWriter::new();
        w.write_header_placeholder().unwrap();
        w.write_schema_block(schema).unwrap();
        w.reserve_column_directory(1).unwrap();
        let values: Vec<u8> = (1i64..=2).flat_map(|v| v.to_le_bytes()).collect();
        let (data_off, data_len) = w.write_values_buffer(&values, 8).unwrap();
        let meta = ColumnMeta {
            column_id: 0,
            physical_type: V0_PHYSICAL_FIXED_WIDTH_I64,
            logical_type: 0,
            data_offset: data_off,
            data_length: data_len,
            validity_offset: 0,
            validity_length: 0,
            offsets_offset: 0,
            offsets_length: 0,
            stats_offset: 0,
            stats_length: 0,
        };
        w.patch_column_directory(std::slice::from_ref(&meta))
            .unwrap();
        w.finalize_header().unwrap();
        let file = w.into_inner();

        let reader = ColumnarReader::new(&file).unwrap();
        assert!(!reader.header().values_are_64_aligned());
        assert_eq!(data_off % 8, 0);
    }

    #[test]
    fn flagged_64_byte_values_alignment_rejects_misaligned_data() {
        let schema = b"ipc-schema-bytes";
        let mut w = ColumnarWriter::new().with_value_alignment(ValueAlignmentStrategy::Align64);
        w.write_header_placeholder().unwrap();
        w.write_schema_block(schema).unwrap();
        w.reserve_column_directory(1).unwrap();
        w.pad_to_alignment(MIN_BUFFER_ALIGN as usize).unwrap();
        let values: Vec<u8> = (1i64..=2).flat_map(|v| v.to_le_bytes()).collect();
        let (data_off, data_len) = w.write_fixed_width_values(&values, 8).unwrap();
        assert_ne!(data_off % 64, 0);
        let meta = ColumnMeta {
            column_id: 0,
            physical_type: V0_PHYSICAL_FIXED_WIDTH_I64,
            logical_type: 0,
            data_offset: data_off,
            data_length: data_len,
            validity_offset: 0,
            validity_length: 0,
            offsets_offset: 0,
            offsets_length: 0,
            stats_offset: 0,
            stats_length: 0,
        };
        let err = w
            .patch_column_directory(std::slice::from_ref(&meta))
            .unwrap_err();
        assert!(matches!(
            err,
            crate::ColumnarWriteError::MisalignedDataOffset {
                required_alignment: 64,
                ..
            }
        ));
    }

    #[test]
    fn truncated_file_rejected() {
        let mut file = vec![0u8; FILE_HEADER_LEN];
        let hdr = FileHeader {
            magic: *b"COLUMNAR",
            version: 1,
            flags: 0,
            header_size: 64,
            schema_offset: 64,
            schema_length: 100,
            column_dir_offset: 200,
            column_dir_length: 80,
            reserved: [0u8; 8],
        };
        file[0..FILE_HEADER_LEN].copy_from_slice(&hdr.serialize());
        let err = ColumnarReader::new(&file).unwrap_err();
        match err {
            super::ColumnarReadError::OutOfBounds { .. }
            | super::ColumnarReadError::TooSmall { .. } => {}
            e => panic!("unexpected {e:?}"),
        }
    }

    #[test]
    fn variable_length_column_offsets_and_values_roundtrip() {
        let schema = b"ipc-schema-utf8";
        let values = b"alphabetagamma";
        let offsets = [0i32, 5, 9, 14];
        let validity = [true, false, true];

        let mut w = ColumnarWriter::new();
        w.write_header_placeholder().unwrap();
        w.write_schema_block(schema).unwrap();
        w.reserve_column_directory(1).unwrap();

        w.pad_to_alignment(MIN_BUFFER_ALIGN as usize).unwrap();
        let mut bitmap = [0u8; 1];
        for (index, valid) in validity.iter().copied().enumerate() {
            if valid {
                bitmap[index / 8] |= 1u8 << (index % 8);
            }
        }
        let (validity_offset, validity_length) = w.write_fixed_width_values(&bitmap, 1).unwrap();

        w.pad_to_alignment(MIN_BUFFER_ALIGN as usize).unwrap();
        let offset_bytes: Vec<u8> = offsets
            .iter()
            .flat_map(|offset| offset.to_le_bytes())
            .collect();
        let (offsets_offset, offsets_length) =
            w.write_fixed_width_values(&offset_bytes, 4).unwrap();

        w.pad_to_alignment(VALUES_BUFFER_ALIGN).unwrap();
        let (data_offset, data_length) = w.write_fixed_width_values(values, 1).unwrap();

        let meta = ColumnMeta {
            column_id: 0,
            physical_type: V0_PHYSICAL_UTF8_I32,
            logical_type: 0,
            data_offset,
            data_length,
            validity_offset,
            validity_length,
            offsets_offset,
            offsets_length,
            stats_offset: 0,
            stats_length: 0,
        };
        w.patch_column_directory(std::slice::from_ref(&meta))
            .unwrap();
        w.finalize_header().unwrap();
        let file = w.into_inner();

        let reader = ColumnarReader::new(&file).unwrap();
        let variable = reader.variable_column_buffers(0).unwrap();
        assert_eq!(variable.values, values);
        assert_eq!(variable.offsets, offset_bytes.as_slice());
        assert_eq!(
            variable.values.as_ptr(),
            file[data_offset as usize..].as_ptr()
        );
        assert_eq!(
            variable.offsets.as_ptr(),
            file[offsets_offset as usize..].as_ptr()
        );
        assert_eq!(variable.validity.unwrap(), bitmap.as_slice());
    }

    #[test]
    fn variable_length_column_without_offsets_is_rejected() {
        let schema = b"ipc-schema-utf8";
        let values = b"hello";

        let mut w = ColumnarWriter::new();
        w.write_header_placeholder().unwrap();
        w.write_schema_block(schema).unwrap();
        w.reserve_column_directory(1).unwrap();
        w.pad_to_alignment(VALUES_BUFFER_ALIGN).unwrap();
        let (data_offset, data_length) = w.write_fixed_width_values(values, 1).unwrap();

        let meta = ColumnMeta {
            column_id: 0,
            physical_type: V0_PHYSICAL_UTF8_I32,
            logical_type: 0,
            data_offset,
            data_length,
            validity_offset: 0,
            validity_length: 0,
            offsets_offset: 0,
            offsets_length: 0,
            stats_offset: 0,
            stats_length: 0,
        };
        w.patch_column_directory(std::slice::from_ref(&meta))
            .unwrap();
        w.finalize_header().unwrap();
        let file = w.into_inner();

        let err = ColumnarReader::new(&file)
            .unwrap()
            .variable_column_buffers(0)
            .unwrap_err();
        assert!(matches!(
            err,
            super::ColumnarReadError::MissingOffsets { column_index: 0 }
        ));
    }
}
