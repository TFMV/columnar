//! Zero-copy read access to a Columnar file backed by a single byte slice (e.g. mmap).

use core::fmt;

use crate::directory::{
    BufferField, ColumnDirectoryError, ColumnDirectoryView, ColumnMeta, COLUMN_META_LEN,
    MIN_BUFFER_ALIGN,
};
use crate::header::{FileHeader, FileHeaderError, FILE_HEADER_LEN};

/// Parsed file with all payload views borrowing `bytes` (typically memory-mapped).
#[derive(Debug, Clone, Copy)]
pub struct ColumnarReader<'a> {
    bytes: &'a [u8],
    header: FileHeader,
    directory: ColumnDirectoryView<'a>,
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
    },
    BufferOverlapsStructure {
        column_index: usize,
        field: BufferField,
        start: u64,
        end: u64,
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
                write!(f, "file size does not fit in address space on this platform")
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
            ColumnarReadError::SchemaOffsetTooSmall { offset, min } => write!(
                f,
                "schema_offset {offset} is before end of header ({min})"
            ),
            ColumnarReadError::SchemaStartNotAligned { offset } => {
                write!(f, "schema_offset {offset} is not aligned to {MIN_BUFFER_ALIGN}")
            }
            ColumnarReadError::DirectoryStartNotAligned { offset } => {
                write!(f, "column_dir_offset {offset} is not aligned to {MIN_BUFFER_ALIGN}")
            }
            ColumnarReadError::SectionOrdering { msg } => write!(f, "{msg}"),
            ColumnarReadError::UnalignedBuffer {
                column_index,
                field,
                offset,
            } => write!(
                f,
                "column {column_index} {field} offset {offset} is not aligned to {MIN_BUFFER_ALIGN}"
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

        let file_len = u64::try_from(bytes.len()).map_err(|_| ColumnarReadError::FileTooLargeForPlatform)?;

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

        let (dir_start, dir_end) = u64_range_to_usize(
            header.column_dir_offset,
            header.column_dir_length,
            file_len,
        )?;

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

        let reader = Self {
            bytes,
            header,
            directory,
            schema_start,
            schema_end,
            dir_start,
            dir_end,
        };
        reader.validate_buffers_vs_structure(file_len)?;
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
        self.directory.len()
    }

    #[inline]
    pub fn column_meta(&self, index: usize) -> Result<ColumnMeta, ColumnarReadError> {
        Ok(self.directory.get(index)?)
    }

    /// Values buffer for column `index` (may be empty if `data_length == 0`).
    pub fn column_values(&self, index: usize) -> Result<&'a [u8], ColumnarReadError> {
        let m = self.column_meta(index)?;
        self.slice_buffer(index, BufferField::Data, m.data_offset, m.data_length)
    }

    pub fn column_validity(&self, index: usize) -> Result<Option<&'a [u8]>, ColumnarReadError> {
        let m = self.column_meta(index)?;
        self.optional_slice_buffer(index, BufferField::Validity, m.validity_offset, m.validity_length)
    }

    pub fn column_offsets(&self, index: usize) -> Result<Option<&'a [u8]>, ColumnarReadError> {
        let m = self.column_meta(index)?;
        self.optional_slice_buffer(index, BufferField::Offsets, m.offsets_offset, m.offsets_length)
    }

    pub fn column_stats(&self, index: usize) -> Result<Option<&'a [u8]>, ColumnarReadError> {
        let m = self.column_meta(index)?;
        self.optional_slice_buffer(index, BufferField::Stats, m.stats_offset, m.stats_length)
    }

    pub fn column_buffers(&self, index: usize) -> Result<ColumnBufferSlices<'a>, ColumnarReadError> {
        Ok(ColumnBufferSlices {
            values: self.column_values(index)?,
            validity: self.column_validity(index)?,
            offsets: self.column_offsets(index)?,
            stats: self.column_stats(index)?,
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

        let structural = [
            (0u64, hdr_end),
            (schema0, schema1),
            (dir0, dir1),
        ];

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
                let end = off.checked_add(len).ok_or(ColumnarReadError::RangeOverflow)?;
                if end > file_len {
                    return Err(ColumnarReadError::OutOfBounds {
                        offset: off,
                        length: len,
                        file_len,
                    });
                }
                if off % MIN_BUFFER_ALIGN != 0 {
                    return Err(ColumnarReadError::UnalignedBuffer {
                        column_index: i,
                        field,
                        offset: off,
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
        if offset % MIN_BUFFER_ALIGN != 0 {
            return Err(ColumnarReadError::UnalignedBuffer {
                column_index,
                field,
                offset,
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
        Ok(Some(self.slice_buffer(column_index, field, offset, length)?))
    }
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
    let file_len_u64 = u64::try_from(file_len).map_err(|_| ColumnarReadError::FileTooLargeForPlatform)?;
    let (start, end) = u64_range_to_usize(offset, length, file_len_u64)?;
    Ok((start, end))
}

#[cfg(test)]
mod tests {
    use super::ColumnarReader;
    use crate::{
        ColumnMeta, ColumnarWriter, FileHeader, FILE_HEADER_LEN, V0_PHYSICAL_FIXED_WIDTH_I64,
        VALUES_BUFFER_ALIGN,
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
        w.patch_column_directory(std::slice::from_ref(&meta)).unwrap();
        w.finalize_header().unwrap();
        let file = w.into_inner();

        let r = ColumnarReader::new(&file).expect("parse");
        assert_eq!(r.header().magic, *b"COLUMNAR");
        assert_eq!(r.schema_bytes(), schema);
        assert_eq!(r.schema_bytes().as_ptr(), file[r.header().schema_offset as usize..].as_ptr());
        assert_eq!(r.column_count(), 1);
        assert_eq!(r.column_values(0).unwrap(), values.as_slice());
        assert_eq!(
            r.column_values(0).unwrap().as_ptr(),
            file[data_off as usize..].as_ptr()
        );
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
}
