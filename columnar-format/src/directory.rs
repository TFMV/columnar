//! Column directory: contiguous [`ColumnMeta`] records (format §5–6).

use core::fmt;

/// Minimum buffer start alignment (format §1 / §6).
pub const MIN_BUFFER_ALIGN: u64 = 8;

/// Serialized length of one [`ColumnMeta`] on disk: 3×`u32` + 4-byte pad + 8×`u64` (little-endian).
pub const COLUMN_META_LEN: usize = 80;

/// One column entry in the directory (format §5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct ColumnMeta {
    pub column_id: u32,
    pub physical_type: u32,
    pub logical_type: u32,
    pub data_offset: u64,
    pub data_length: u64,
    pub validity_offset: u64,
    pub validity_length: u64,
    pub offsets_offset: u64,
    pub offsets_length: u64,
    pub stats_offset: u64,
    pub stats_length: u64,
}

const _: () = assert!(core::mem::size_of::<ColumnMeta>() == COLUMN_META_LEN);
const _: () = assert!(core::mem::align_of::<ColumnMeta>() == 8);

/// Identifies which buffer range failed validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferField {
    Data,
    Validity,
    Offsets,
    Stats,
}

impl fmt::Display for BufferField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            BufferField::Data => "data",
            BufferField::Validity => "validity",
            BufferField::Offsets => "offsets",
            BufferField::Stats => "stats",
        })
    }
}

/// Errors from parsing or validating directory / column metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ColumnDirectoryError {
    /// `ColumnMeta` slice shorter than [`COLUMN_META_LEN`].
    TruncatedEntry { need: usize, got: usize },
    /// Directory byte length is not a multiple of [`COLUMN_META_LEN`].
    WrongDirectoryLength { got: usize, multiple_of: usize },
    /// Bytes `[12..16]` in a wire record must be zero.
    NonZeroMetaPadding { got: u32 },
    /// Buffer start not aligned to [`MIN_BUFFER_ALIGN`] while length &gt; 0.
    UnalignedBufferStart {
        column_index: usize,
        field: BufferField,
        offset: u64,
    },
    /// `offset + length` overflowed `u64`.
    BufferLengthOverflow {
        column_index: usize,
        field: BufferField,
        offset: u64,
        length: u64,
    },
    /// Buffer extends past `file_size`.
    BufferOutOfBounds {
        column_index: usize,
        field: BufferField,
        offset: u64,
        length: u64,
        end: u64,
        file_size: u64,
    },
    /// Two present buffers overlap in file byte space.
    OverlappingBuffers {
        column_a: usize,
        field_a: BufferField,
        start_a: u64,
        end_a: u64,
        column_b: usize,
        field_b: BufferField,
        start_b: u64,
        end_b: u64,
    },
    /// `serialize_into` / `deserialize` expected a different buffer size.
    WrongSerializeBufferLen { need: usize, got: usize },
    /// Column index out of range for [`ColumnDirectoryView::get`].
    InvalidColumnIndex { index: usize, len: usize },
}

impl fmt::Display for ColumnDirectoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ColumnDirectoryError::TruncatedEntry { need, got } => {
                write!(f, "truncated column meta: need {need} bytes, got {got}")
            }
            ColumnDirectoryError::WrongDirectoryLength { got, multiple_of } => write!(
                f,
                "directory length {got} is not a multiple of {multiple_of}"
            ),
            ColumnDirectoryError::NonZeroMetaPadding { got } => {
                write!(f, "column meta wire padding must be zero, got {got}")
            }
            ColumnDirectoryError::UnalignedBufferStart {
                column_index,
                field,
                offset,
            } => write!(
                f,
                "column {column_index} {field} offset {offset} is not aligned to {MIN_BUFFER_ALIGN}"
            ),
            ColumnDirectoryError::BufferLengthOverflow {
                column_index,
                field,
                offset,
                length,
            } => write!(
                f,
                "column {column_index} {field} offset {offset} length {length} overflows u64"
            ),
            ColumnDirectoryError::BufferOutOfBounds {
                column_index,
                field,
                offset,
                length,
                end,
                file_size,
            } => write!(
                f,
                "column {column_index} {field} range [{offset}, {end}) (len {length}) exceeds file size {file_size}"
            ),
            ColumnDirectoryError::OverlappingBuffers {
                column_a,
                field_a,
                start_a,
                end_a,
                column_b,
                field_b,
                start_b,
                end_b,
            } => write!(
                f,
                "overlapping buffers: column {column_a} {field_a} [{start_a}, {end_a}) vs column {column_b} {field_b} [{start_b}, {end_b})"
            ),
            ColumnDirectoryError::WrongSerializeBufferLen { need, got } => {
                write!(f, "buffer length mismatch: need {need}, got {got}")
            }
            ColumnDirectoryError::InvalidColumnIndex { index, len } => {
                write!(f, "column index {index} out of range (column count {len})")
            }
        }
    }
}

impl std::error::Error for ColumnDirectoryError {}

impl ColumnMeta {
    /// Writes this record to `out` in little-endian wire form. `out.len()` must be [`COLUMN_META_LEN`].
    pub fn serialize_into(&self, out: &mut [u8]) -> Result<(), ColumnDirectoryError> {
        if out.len() != COLUMN_META_LEN {
            return Err(ColumnDirectoryError::WrongSerializeBufferLen {
                need: COLUMN_META_LEN,
                got: out.len(),
            });
        }
        out[0..4].copy_from_slice(&self.column_id.to_le_bytes());
        out[4..8].copy_from_slice(&self.physical_type.to_le_bytes());
        out[8..12].copy_from_slice(&self.logical_type.to_le_bytes());
        out[12..16].fill(0);
        out[16..24].copy_from_slice(&self.data_offset.to_le_bytes());
        out[24..32].copy_from_slice(&self.data_length.to_le_bytes());
        out[32..40].copy_from_slice(&self.validity_offset.to_le_bytes());
        out[40..48].copy_from_slice(&self.validity_length.to_le_bytes());
        out[48..56].copy_from_slice(&self.offsets_offset.to_le_bytes());
        out[56..64].copy_from_slice(&self.offsets_length.to_le_bytes());
        out[64..72].copy_from_slice(&self.stats_offset.to_le_bytes());
        out[72..80].copy_from_slice(&self.stats_length.to_le_bytes());
        Ok(())
    }

    /// Little-endian wire form of this record. No heap allocation.
    pub fn serialize(&self) -> [u8; COLUMN_META_LEN] {
        let mut out = [0u8; COLUMN_META_LEN];
        self.serialize_into(&mut out).expect("fixed buffer");
        out
    }

    /// Parses one record from the start of `bytes` (length ≥ [`COLUMN_META_LEN`]). No heap allocation.
    pub fn deserialize(bytes: &[u8]) -> Result<Self, ColumnDirectoryError> {
        if bytes.len() < COLUMN_META_LEN {
            return Err(ColumnDirectoryError::TruncatedEntry {
                need: COLUMN_META_LEN,
                got: bytes.len(),
            });
        }
        let pad = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        if pad != 0 {
            return Err(ColumnDirectoryError::NonZeroMetaPadding { got: pad });
        }
        Ok(ColumnMeta {
            column_id: u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            physical_type: u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            logical_type: u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            data_offset: u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            data_length: u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
            validity_offset: u64::from_le_bytes(bytes[32..40].try_into().unwrap()),
            validity_length: u64::from_le_bytes(bytes[40..48].try_into().unwrap()),
            offsets_offset: u64::from_le_bytes(bytes[48..56].try_into().unwrap()),
            offsets_length: u64::from_le_bytes(bytes[56..64].try_into().unwrap()),
            stats_offset: u64::from_le_bytes(bytes[64..72].try_into().unwrap()),
            stats_length: u64::from_le_bytes(bytes[72..80].try_into().unwrap()),
        })
    }
}

/// Owned, contiguous column directory (`Vec` of entries — single flat allocation).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnDirectory {
    entries: Vec<ColumnMeta>,
}

impl ColumnDirectory {
    pub fn new(entries: Vec<ColumnMeta>) -> Self {
        Self { entries }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[inline]
    pub fn as_slice(&self) -> &[ColumnMeta] {
        &self.entries
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [ColumnMeta] {
        &mut self.entries
    }

    /// Total wire size for [`ColumnDirectory::serialize_into`].
    pub fn serialized_byte_len(&self) -> usize {
        self.entries.len().saturating_mul(COLUMN_META_LEN)
    }

    pub fn serialize_into(&self, out: &mut [u8]) -> Result<(), ColumnDirectoryError> {
        let need = self.serialized_byte_len();
        if out.len() != need {
            return Err(ColumnDirectoryError::WrongSerializeBufferLen {
                need,
                got: out.len(),
            });
        }
        for (i, entry) in self.entries.iter().enumerate() {
            let start = i * COLUMN_META_LEN;
            let end = start + COLUMN_META_LEN;
            entry.serialize_into(&mut out[start..end])?;
        }
        Ok(())
    }

    /// Flat contiguous bytes (one allocation).
    pub fn serialize(&self) -> Vec<u8> {
        let mut v = vec![0u8; self.serialized_byte_len()];
        self.serialize_into(&mut v)
            .expect("buffer sized to serialized_byte_len");
        v
    }

    pub fn deserialize(bytes: &[u8]) -> Result<Self, ColumnDirectoryError> {
        if bytes.len() % COLUMN_META_LEN != 0 {
            return Err(ColumnDirectoryError::WrongDirectoryLength {
                got: bytes.len(),
                multiple_of: COLUMN_META_LEN,
            });
        }
        let n = bytes.len() / COLUMN_META_LEN;
        let mut entries = Vec::with_capacity(n);
        for i in 0..n {
            let start = i * COLUMN_META_LEN;
            let end = start + COLUMN_META_LEN;
            entries.push(ColumnMeta::deserialize(&bytes[start..end])?);
        }
        Ok(Self { entries })
    }

    pub fn validate(&self, file_size: u64) -> Result<(), ColumnDirectoryError> {
        validate_column_buffers(&self.entries, file_size)
    }
}

/// Borrowed directory backed by a contiguous byte slice (e.g. memory-mapped file region).
///
/// Layout matches [`ColumnMeta::serialize`] / [`ColumnMeta::deserialize`] so the slice can point
/// directly at mapped file bytes without nested vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColumnDirectoryView<'a> {
    bytes: &'a [u8],
}

impl<'a> ColumnDirectoryView<'a> {
    pub fn new(bytes: &'a [u8]) -> Result<Self, ColumnDirectoryError> {
        if bytes.len() % COLUMN_META_LEN != 0 {
            return Err(ColumnDirectoryError::WrongDirectoryLength {
                got: bytes.len(),
                multiple_of: COLUMN_META_LEN,
            });
        }
        Ok(Self { bytes })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.bytes.len() / COLUMN_META_LEN
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    #[inline]
    pub fn as_bytes(&self) -> &'a [u8] {
        self.bytes
    }

    pub fn get(&self, index: usize) -> Result<ColumnMeta, ColumnDirectoryError> {
        if index >= self.len() {
            return Err(ColumnDirectoryError::InvalidColumnIndex {
                index,
                len: self.len(),
            });
        }
        let start = index * COLUMN_META_LEN;
        let end = start + COLUMN_META_LEN;
        ColumnMeta::deserialize(&self.bytes[start..end])
    }

    pub fn validate(&self, file_size: u64) -> Result<(), ColumnDirectoryError> {
        let mut ranges: Vec<BufferRange> = Vec::with_capacity(self.len().saturating_mul(4));
        for i in 0..self.len() {
            let m = self.get(i)?;
            append_column_ranges(&mut ranges, i, &m)?;
        }
        validate_buffer_ranges(ranges, file_size)
    }
}

#[derive(Clone, Copy)]
struct BufferRange {
    column_index: usize,
    field: BufferField,
    start: u64,
    end: u64,
}

fn checked_end(
    column_index: usize,
    field: BufferField,
    offset: u64,
    length: u64,
) -> Result<u64, ColumnDirectoryError> {
    let end = offset
        .checked_add(length)
        .ok_or(ColumnDirectoryError::BufferLengthOverflow {
            column_index,
            field,
            offset,
            length,
        })?;
    Ok(end)
}

fn push_if_present(
    out: &mut Vec<BufferRange>,
    column_index: usize,
    field: BufferField,
    offset: u64,
    length: u64,
) -> Result<(), ColumnDirectoryError> {
    if length == 0 {
        return Ok(());
    }
    if offset % MIN_BUFFER_ALIGN != 0 {
        return Err(ColumnDirectoryError::UnalignedBufferStart {
            column_index,
            field,
            offset,
        });
    }
    let end = checked_end(column_index, field, offset, length)?;
    out.push(BufferRange {
        column_index,
        field,
        start: offset,
        end,
    });
    Ok(())
}

fn append_column_ranges(
    out: &mut Vec<BufferRange>,
    column_index: usize,
    m: &ColumnMeta,
) -> Result<(), ColumnDirectoryError> {
    push_if_present(
        out,
        column_index,
        BufferField::Data,
        m.data_offset,
        m.data_length,
    )?;
    push_if_present(
        out,
        column_index,
        BufferField::Validity,
        m.validity_offset,
        m.validity_length,
    )?;
    push_if_present(
        out,
        column_index,
        BufferField::Offsets,
        m.offsets_offset,
        m.offsets_length,
    )?;
    push_if_present(
        out,
        column_index,
        BufferField::Stats,
        m.stats_offset,
        m.stats_length,
    )?;
    Ok(())
}

fn validate_buffer_ranges(
    mut ranges: Vec<BufferRange>,
    file_size: u64,
) -> Result<(), ColumnDirectoryError> {
    for r in &ranges {
        if r.end > file_size {
            return Err(ColumnDirectoryError::BufferOutOfBounds {
                column_index: r.column_index,
                field: r.field,
                offset: r.start,
                length: r.end - r.start,
                end: r.end,
                file_size,
            });
        }
    }

    ranges.sort_by_key(|r| (r.start, r.end));

    let mut max_end = 0u64;
    let mut tail: Option<BufferRange> = None;
    for r in ranges {
        if r.start < max_end {
            let p = tail.expect("max_end > 0 implies an interval that set it");
            return Err(ColumnDirectoryError::OverlappingBuffers {
                column_a: p.column_index,
                field_a: p.field,
                start_a: p.start,
                end_a: p.end,
                column_b: r.column_index,
                field_b: r.field,
                start_b: r.start,
                end_b: r.end,
            });
        }
        if r.end > max_end {
            max_end = r.end;
            tail = Some(r);
        }
    }

    Ok(())
}

fn validate_column_buffers(
    columns: &[ColumnMeta],
    file_size: u64,
) -> Result<(), ColumnDirectoryError> {
    let mut ranges: Vec<BufferRange> = Vec::with_capacity(columns.len().saturating_mul(4));
    for (column_index, m) in columns.iter().enumerate() {
        append_column_ranges(&mut ranges, column_index, m)?;
    }
    validate_buffer_ranges(ranges, file_size)
}

#[cfg(test)]
mod tests {
    use super::{
        BufferField, ColumnDirectory, ColumnDirectoryError, ColumnDirectoryView, ColumnMeta,
        COLUMN_META_LEN,
    };

    fn meta(
        id: u32,
        data_off: u64,
        data_len: u64,
        val_off: u64,
        val_len: u64,
        off_off: u64,
        off_len: u64,
        st_off: u64,
        st_len: u64,
    ) -> ColumnMeta {
        ColumnMeta {
            column_id: id,
            physical_type: id,
            logical_type: 0,
            data_offset: data_off,
            data_length: data_len,
            validity_offset: val_off,
            validity_length: val_len,
            offsets_offset: off_off,
            offsets_length: off_len,
            stats_offset: st_off,
            stats_length: st_len,
        }
    }

    #[test]
    fn column_meta_round_trip_wire_matches_repr_size() {
        let m = meta(1, 100, 10, 0, 0, 0, 0, 0, 0);
        let b = m.serialize();
        assert_eq!(b.len(), COLUMN_META_LEN);
        assert_eq!(ColumnMeta::deserialize(&b).unwrap(), m);
    }

    #[test]
    fn directory_round_trip_three_columns() {
        let dir = ColumnDirectory::new(vec![
            meta(0, 64, 16, 0, 0, 0, 0, 0, 0),
            meta(1, 80, 8, 0, 0, 0, 0, 0, 0),
            meta(2, 88, 24, 0, 0, 0, 0, 0, 0),
        ]);
        let bytes = dir.serialize();
        assert_eq!(bytes.len(), 3 * COLUMN_META_LEN);
        let got = ColumnDirectory::deserialize(&bytes).unwrap();
        assert_eq!(got, dir);
        ColumnDirectoryView::new(&bytes)
            .unwrap()
            .validate(200)
            .unwrap();
    }

    #[test]
    fn mmap_view_get_matches_deserialize() {
        let dir = ColumnDirectory::new(vec![
            meta(10, 128, 32, 0, 0, 0, 0, 0, 0),
            meta(11, 160, 32, 0, 0, 0, 0, 0, 0),
        ]);
        let bytes = dir.serialize();
        let view = ColumnDirectoryView::new(&bytes).unwrap();
        assert_eq!(view.len(), 2);
        assert_eq!(view.get(0).unwrap(), dir.as_slice()[0]);
        assert_eq!(view.get(1).unwrap(), dir.as_slice()[1]);
    }

    #[test]
    fn wrong_directory_length() {
        let err = ColumnDirectory::deserialize(&[0u8; COLUMN_META_LEN + 1]).unwrap_err();
        assert_eq!(
            err,
            ColumnDirectoryError::WrongDirectoryLength {
                got: COLUMN_META_LEN + 1,
                multiple_of: COLUMN_META_LEN
            }
        );
    }

    #[test]
    fn overlapping_buffers_rejected() {
        let dir = ColumnDirectory::new(vec![
            meta(0, 64, 32, 0, 0, 0, 0, 0, 0),
            meta(1, 80, 32, 0, 0, 0, 0, 0, 0),
        ]);
        let err = dir.validate(10_000).unwrap_err();
        match err {
            ColumnDirectoryError::OverlappingBuffers { .. } => {}
            e => panic!("expected overlap, got {e:?}"),
        }
    }

    #[test]
    fn unaligned_data_offset_rejected() {
        let dir = ColumnDirectory::new(vec![meta(0, 65, 8, 0, 0, 0, 0, 0, 0)]);
        let err = dir.validate(200).unwrap_err();
        assert_eq!(
            err,
            ColumnDirectoryError::UnalignedBufferStart {
                column_index: 0,
                field: BufferField::Data,
                offset: 65
            }
        );
    }

    #[test]
    fn out_of_bounds_rejected() {
        let dir = ColumnDirectory::new(vec![meta(0, 64, 100, 0, 0, 0, 0, 0, 0)]);
        let err = dir.validate(120).unwrap_err();
        match err {
            ColumnDirectoryError::BufferOutOfBounds { .. } => {}
            e => panic!("expected oob, got {e:?}"),
        }
    }

    #[test]
    fn multi_column_non_overlapping_valid() {
        let dir = ColumnDirectory::new(vec![
            meta(0, 64, 32, 96, 8, 104, 16, 120, 8),
            meta(1, 128, 64, 192, 8, 200, 8, 208, 16),
            meta(2, 224, 32, 0, 0, 0, 0, 256, 8),
        ]);
        dir.validate(300).unwrap();
    }

    #[test]
    fn non_zero_meta_padding_rejected() {
        let mut b = meta(0, 0, 0, 0, 0, 0, 0, 0, 0).serialize();
        b[12..16].copy_from_slice(&1u32.to_le_bytes());
        let err = ColumnMeta::deserialize(&b).unwrap_err();
        assert_eq!(err, ColumnDirectoryError::NonZeroMetaPadding { got: 1 });
    }

    #[test]
    fn view_index_out_of_range() {
        let dir = ColumnDirectory::new(vec![meta(0, 64, 8, 0, 0, 0, 0, 0, 0)]);
        let bytes = dir.serialize();
        let view = ColumnDirectoryView::new(&bytes).unwrap();
        let err = view.get(1).unwrap_err();
        assert_eq!(
            err,
            ColumnDirectoryError::InvalidColumnIndex { index: 1, len: 1 }
        );
    }

    #[test]
    fn overlap_report_includes_columns() {
        let dir = ColumnDirectory::new(vec![
            meta(0, 64, 32, 0, 0, 0, 0, 0, 0),
            meta(1, 80, 8, 0, 0, 0, 0, 0, 0),
        ]);
        let err = dir.validate(500).unwrap_err();
        let ColumnDirectoryError::OverlappingBuffers {
            column_a,
            column_b,
            field_a,
            field_b,
            ..
        } = err
        else {
            panic!("expected overlap");
        };
        assert_ne!(column_a, column_b);
        assert_eq!(field_a, BufferField::Data);
        assert_eq!(field_b, BufferField::Data);
    }
}
