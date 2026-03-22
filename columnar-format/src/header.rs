//! Fixed 64-byte file header (format §3).

/// ASCII magic for Columnar files (`"COLUMNAR"`).
pub const FILE_HEADER_MAGIC: [u8; 8] = *b"COLUMNAR";

/// Supported format version for this implementation.
pub const FILE_HEADER_VERSION: u16 = 1;
/// Header flag indicating all values buffers are 64-byte aligned.
pub const FILE_FLAG_VALUES_ALIGNED_64: u16 = 1 << 0;

/// Declared header size for v1 on disk (format §3).
pub const FILE_HEADER_ON_DISK_SIZE: u32 = 64;

/// Serialized header length in bytes (fixed 64B header + 8-byte alignment constraint).
pub const FILE_HEADER_LEN: usize = 64;

/// Size of the [`FileHeader`] type in memory (`repr(C)` layout without trailing file padding).
pub const FILE_HEADER_STRUCT_LEN: usize = 56;

/// Fixed file header at offset 0. Integer fields are **little-endian** on disk (Arrow IPC convention).
///
/// The on-disk record is always [`FILE_HEADER_LEN`] bytes: this layout (56 bytes) followed by 8
/// padding bytes. [`FileHeader::serialize`] writes zeros in that tail; [`FileHeader::deserialize`]
/// requires exactly 64 bytes and ignores the content of the tail (it is not stored).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct FileHeader {
    pub magic: [u8; 8],
    pub version: u16,
    pub flags: u16,
    pub header_size: u32,
    pub schema_offset: u64,
    pub schema_length: u64,
    pub column_dir_offset: u64,
    pub column_dir_length: u64,
    pub reserved: [u8; 8],
}

const _: () = assert!(std::mem::size_of::<FileHeader>() == FILE_HEADER_STRUCT_LEN);
const _: () = assert!(std::mem::align_of::<FileHeader>() >= 8);

/// Errors from parsing or validating a [`FileHeader`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileHeaderError {
    /// Slice length was not [`FILE_HEADER_LEN`].
    WrongLength { got: usize },
    /// `magic` was not [`FILE_HEADER_MAGIC`].
    BadMagic,
    /// `version` is not supported (only [`FILE_HEADER_VERSION`] is accepted).
    UnsupportedVersion { got: u16 },
    /// `header_size` must be [`FILE_HEADER_ON_DISK_SIZE`] for v1.
    InvalidHeaderSize { got: u32 },
    /// Header uses bits not defined by this implementation.
    InvalidFlags { got: u16 },
}

impl std::fmt::Display for FileHeaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FileHeaderError::WrongLength { got } => {
                write!(f, "expected {FILE_HEADER_LEN} header bytes, got {got}")
            }
            FileHeaderError::BadMagic => write!(f, "invalid magic (expected COLUMNAR)"),
            FileHeaderError::UnsupportedVersion { got } => {
                write!(
                    f,
                    "unsupported header version {got} (expected {FILE_HEADER_VERSION})"
                )
            }
            FileHeaderError::InvalidHeaderSize { got } => write!(
                f,
                "invalid header_size {got} (expected {FILE_HEADER_ON_DISK_SIZE} for v1)"
            ),
            FileHeaderError::InvalidFlags { got } => {
                write!(f, "invalid header flags {got:#06x}")
            }
        }
    }
}

impl std::error::Error for FileHeaderError {}

impl FileHeader {
    /// Validates magic, version, and `header_size` for v1.
    pub fn validate(&self) -> Result<(), FileHeaderError> {
        if self.magic != FILE_HEADER_MAGIC {
            return Err(FileHeaderError::BadMagic);
        }
        if self.version != FILE_HEADER_VERSION {
            return Err(FileHeaderError::UnsupportedVersion { got: self.version });
        }
        if self.header_size != FILE_HEADER_ON_DISK_SIZE {
            return Err(FileHeaderError::InvalidHeaderSize {
                got: self.header_size,
            });
        }
        if self.flags & !FILE_FLAG_VALUES_ALIGNED_64 != 0 {
            return Err(FileHeaderError::InvalidFlags { got: self.flags });
        }
        Ok(())
    }

    #[inline]
    pub fn values_are_64_aligned(&self) -> bool {
        self.flags & FILE_FLAG_VALUES_ALIGNED_64 != 0
    }

    #[inline]
    pub fn logical_column_count(&self) -> u32 {
        u32::from_le_bytes(self.reserved[..4].try_into().expect("reserved width"))
    }

    #[inline]
    pub fn chunk_count(&self) -> u32 {
        u32::from_le_bytes(self.reserved[4..8].try_into().expect("reserved width"))
    }

    #[inline]
    pub fn with_chunk_layout(mut self, logical_column_count: u32, chunk_count: u32) -> Self {
        self.reserved[..4].copy_from_slice(&logical_column_count.to_le_bytes());
        self.reserved[4..8].copy_from_slice(&chunk_count.to_le_bytes());
        self
    }

    /// Serializes the header to its on-disk little-endian form (64 bytes). No heap allocation.
    /// The final 8 bytes are zero padding after `reserved`.
    pub fn serialize(&self) -> [u8; FILE_HEADER_LEN] {
        let mut out = [0u8; FILE_HEADER_LEN];
        out[0..8].copy_from_slice(&self.magic);
        out[8..10].copy_from_slice(&self.version.to_le_bytes());
        out[10..12].copy_from_slice(&self.flags.to_le_bytes());
        out[12..16].copy_from_slice(&self.header_size.to_le_bytes());
        out[16..24].copy_from_slice(&self.schema_offset.to_le_bytes());
        out[24..32].copy_from_slice(&self.schema_length.to_le_bytes());
        out[32..40].copy_from_slice(&self.column_dir_offset.to_le_bytes());
        out[40..48].copy_from_slice(&self.column_dir_length.to_le_bytes());
        out[48..56].copy_from_slice(&self.reserved);
        out
    }

    /// Parses a 64-byte header. No heap allocation; no `unsafe`.
    ///
    /// Bytes `[56..64]` are format padding and are not represented in [`FileHeader`].
    pub fn deserialize(bytes: &[u8]) -> Result<Self, FileHeaderError> {
        if bytes.len() != FILE_HEADER_LEN {
            return Err(FileHeaderError::WrongLength { got: bytes.len() });
        }
        let header = FileHeader {
            magic: read_fixed(&bytes[0..8]),
            version: u16::from_le_bytes(read_fixed(&bytes[8..10])),
            flags: u16::from_le_bytes(read_fixed(&bytes[10..12])),
            header_size: u32::from_le_bytes(read_fixed(&bytes[12..16])),
            schema_offset: u64::from_le_bytes(read_fixed(&bytes[16..24])),
            schema_length: u64::from_le_bytes(read_fixed(&bytes[24..32])),
            column_dir_offset: u64::from_le_bytes(read_fixed(&bytes[32..40])),
            column_dir_length: u64::from_le_bytes(read_fixed(&bytes[40..48])),
            reserved: read_fixed(&bytes[48..56]),
        };
        header.validate()?;
        Ok(header)
    }
}

#[inline(always)]
fn read_fixed<const N: usize>(chunk: &[u8]) -> [u8; N] {
    debug_assert_eq!(chunk.len(), N);
    let mut out = [0u8; N];
    out.copy_from_slice(chunk);
    out
}

#[cfg(test)]
mod tests {
    use super::{
        FileHeader, FileHeaderError, FILE_FLAG_VALUES_ALIGNED_64, FILE_HEADER_LEN,
        FILE_HEADER_MAGIC, FILE_HEADER_ON_DISK_SIZE, FILE_HEADER_VERSION,
    };

    fn sample_header() -> FileHeader {
        FileHeader {
            magic: FILE_HEADER_MAGIC,
            version: FILE_HEADER_VERSION,
            flags: 0,
            header_size: FILE_HEADER_ON_DISK_SIZE,
            schema_offset: 64,
            schema_length: 128,
            column_dir_offset: 200,
            column_dir_length: 48,
            reserved: [0xAB; 8],
        }
    }

    #[test]
    fn round_trip() {
        let h = sample_header();
        let bytes = h.serialize();
        assert_eq!(bytes.len(), FILE_HEADER_LEN);
        assert_eq!(&bytes[56..64], &[0u8; 8]);
        let got = FileHeader::deserialize(&bytes).expect("deserialize");
        assert_eq!(got, h);
    }

    #[test]
    fn invalid_magic() {
        let mut h = sample_header();
        h.magic = *b"NOTCOL!!";
        let err = FileHeader::deserialize(&h.serialize()).unwrap_err();
        assert_eq!(err, FileHeaderError::BadMagic);
    }

    #[test]
    fn invalid_size_too_short() {
        let bytes = [0u8; 32];
        let err = FileHeader::deserialize(&bytes).unwrap_err();
        assert_eq!(err, FileHeaderError::WrongLength { got: 32 });
    }

    #[test]
    fn invalid_size_too_long() {
        let mut v = sample_header().serialize().to_vec();
        v.push(0);
        let err = FileHeader::deserialize(v.as_slice()).unwrap_err();
        assert_eq!(
            err,
            FileHeaderError::WrongLength {
                got: FILE_HEADER_LEN + 1
            }
        );
    }

    #[test]
    fn invalid_version() {
        let mut h = sample_header();
        h.version = 999;
        let err = FileHeader::deserialize(&h.serialize()).unwrap_err();
        assert_eq!(err, FileHeaderError::UnsupportedVersion { got: 999 });
    }

    #[test]
    fn invalid_header_size_field() {
        let mut h = sample_header();
        h.header_size = 56;
        let err = FileHeader::deserialize(&h.serialize()).unwrap_err();
        assert_eq!(err, FileHeaderError::InvalidHeaderSize { got: 56 });
    }

    #[test]
    fn serialize_endian_smoke() {
        let mut h = sample_header();
        h.schema_offset = 0x0102_0304_0506_0708;
        let b = h.serialize();
        assert_eq!(&b[16..24], &(0x0102_0304_0506_0708u64).to_le_bytes());
    }

    #[test]
    fn deserialize_ignores_trailing_padding_bytes() {
        let mut bytes = sample_header().serialize();
        bytes[56..64].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let got = FileHeader::deserialize(&bytes).expect("ok");
        assert_eq!(got, sample_header());
    }

    #[test]
    fn values_alignment_flag_round_trip() {
        let mut h = sample_header();
        h.flags = FILE_FLAG_VALUES_ALIGNED_64;
        let got = FileHeader::deserialize(&h.serialize()).expect("deserialize");
        assert!(got.values_are_64_aligned());
    }
}
