use crate::{ColumnarError, ColumnarErrorType, ColumnarType};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

pub const COLUMN_META_LEN: usize = 72;

// These are aspirational. The directory is not currently compressed.
// pub const META_COMPRESSION_TYPE: &str = "zstd";
// pub const META_COMPRESSION_LEVEL: i32 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColumnMeta {
    pub column_id: u32,
    pub column_type: ColumnarType,
    pub data_offset: u64,
    pub data_length: u64,
    pub validity_offset: u64,
    pub validity_length: u64,
    pub offsets_offset: u64,
    pub offsets_length: u64,
    pub stats_offset: u64,
    pub stats_length: u64,
}

impl Default for ColumnMeta {
    fn default() -> Self {
        Self {
            column_id: 0,
            column_type: ColumnarType::Int64,
            data_offset: 0,
            data_length: 0,
            validity_offset: 0,
            validity_length: 0,
            offsets_offset: 0,
            offsets_length: 0,
            stats_offset: 0,
            stats_length: 0,
        }
    }
}

impl ColumnMeta {
    pub fn new(column_id: u32, column_type: ColumnarType) -> Self {
        Self {
            column_id,
            column_type,
            ..Default::default()
        }
    }

    pub fn row_count(&self) -> usize {
        if self.offsets_length > 0 {
            (self.offsets_length as usize / self.column_type.offset_width().unwrap_or(1)) - 1
        } else {
            self.data_length as usize / self.column_type.element_width().unwrap_or(1)
        }
    }

    pub fn serialize(&self, mut bytes: &mut [u8]) {
        bytes.write_u32::<LittleEndian>(self.column_id).unwrap();
        bytes
            .write_u32::<LittleEndian>(self.column_type as u32)
            .unwrap();
        bytes.write_u64::<LittleEndian>(self.data_offset).unwrap();
        bytes.write_u64::<LittleEndian>(self.data_length).unwrap();
        bytes
            .write_u64::<LittleEndian>(self.validity_offset)
            .unwrap();
        bytes
            .write_u64::<LittleEndian>(self.validity_length)
            .unwrap();
        bytes
            .write_u64::<LittleEndian>(self.offsets_offset)
            .unwrap();
        bytes
            .write_u64::<LittleEndian>(self.offsets_length)
            .unwrap();
        bytes.write_u64::<LittleEndian>(self.stats_offset).unwrap();
        bytes.write_u64::<LittleEndian>(self.stats_length).unwrap();
    }

    pub fn deserialize(mut bytes: &[u8]) -> Result<Self, ColumnarError> {
        if bytes.len() != COLUMN_META_LEN {
            return Err(ColumnarError::new(
                ColumnarErrorType::Corrupt,
                format!(
                    "Invalid column meta length: {}, expected: {}",
                    bytes.len(),
                    COLUMN_META_LEN
                ),
            ));
        }
        let column_id = bytes.read_u32::<LittleEndian>().unwrap();
        let column_type = ColumnarType::try_from(bytes.read_u32::<LittleEndian>().unwrap())?;
        let data_offset = bytes.read_u64::<LittleEndian>().unwrap();
        let data_length = bytes.read_u64::<LittleEndian>().unwrap();
        let validity_offset = bytes.read_u64::<LittleEndian>().unwrap();
        let validity_length = bytes.read_u64::<LittleEndian>().unwrap();
        let offsets_offset = bytes.read_u64::<LittleEndian>().unwrap();
        let offsets_length = bytes.read_u64::<LittleEndian>().unwrap();
        let stats_offset = bytes.read_u64::<LittleEndian>().unwrap();
        let stats_length = bytes.read_u64::<LittleEndian>().unwrap();
        Ok(Self {
            column_id,
            column_type,
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
}

pub fn compress_metas(metas: &[ColumnMeta]) -> Result<Vec<u8>, ColumnarError> {
    let mut serialized = vec![0; metas.len() * COLUMN_META_LEN];
    for (i, meta) in metas.iter().enumerate() {
        meta.serialize(&mut serialized[i * COLUMN_META_LEN..(i + 1) * COLUMN_META_LEN]);
    }
    // The directory is not currently compressed.
    // let mut encoder = zstd::stream::Encoder::new(Vec::new(), META_COMPRESSION_LEVEL)?;
    // encoder.write_all(&serialized)?;
    // encoder.finish().map_err(|e| e.into())
    Ok(serialized)
}

pub fn decompress_metas(bytes: &[u8]) -> Result<Vec<ColumnMeta>, ColumnarError> {
    // The directory is not currently compressed.
    // let mut decoder = zstd::stream::Decoder::new(bytes)?;
    // let mut decompressed = Vec::new();
    // decoder.read_to_end(&mut decompressed)?;
    let mut metas = Vec::with_capacity(bytes.len() / COLUMN_META_LEN);
    for chunk in bytes.chunks_exact(COLUMN_META_LEN) {
        metas.push(ColumnMeta::deserialize(chunk)?);
    }
    Ok(metas)
}
