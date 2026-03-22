//! Typed on-disk stats blocks for chunk-level pruning.

use crate::types::ColumnarType;
use core::fmt;

const INT64_STATS_FLAG_HAS_MIN_MAX: u64 = 1 << 0;
const INT64_STATS_FLAG_HAS_DISTINCT_COUNT: u64 = 1 << 1;

/// Typed stats for a fixed-width `i64` column chunk.
///
/// On disk this is serialized as five little-endian 8-byte words:
/// `(min, max, null_count, distinct_count_or_absent, flags)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Int64Stats {
    pub min: Option<i64>,
    pub max: Option<i64>,
    pub null_count: u64,
    pub distinct_count: Option<u64>,
}

/// Typed stats variants supported by the current format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnStats {
    Int64(Int64Stats),
}

/// Errors while decoding or validating a stats block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatsBlockError {
    UnsupportedPhysicalType {
        physical_type: ColumnarType,
    },
    WrongLength {
        physical_type: ColumnarType,
        got: usize,
        expected: usize,
    },
    InvalidFlags {
        physical_type: ColumnarType,
        flags: u64,
    },
}

impl fmt::Display for StatsBlockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StatsBlockError::UnsupportedPhysicalType { physical_type } => write!(
                f,
                "physical type {physical_type:?} does not have a typed stats block"
            ),
            StatsBlockError::WrongLength {
                physical_type,
                got,
                expected,
            } => write!(
                f,
                "physical type {physical_type:?} stats length {got} does not match expected {expected}"
            ),
            StatsBlockError::InvalidFlags {
                physical_type,
                flags,
            } => write!(
                f,
                "physical type {physical_type:?} stats flags {flags:#x} are invalid"
            ),
        }
    }
}

impl std::error::Error for StatsBlockError {}

impl Int64Stats {
    pub const SERIALIZED_LEN: usize = 40;

    #[inline]
    pub fn serialize(self) -> [u8; Self::SERIALIZED_LEN] {
        let mut out = [0u8; Self::SERIALIZED_LEN];
        let (min, max, min_max_flags) = match (self.min, self.max) {
            (Some(min), Some(max)) => (min, max, INT64_STATS_FLAG_HAS_MIN_MAX),
            (None, None) => (0, 0, 0),
            _ => panic!("Int64Stats min/max must be both present or both absent"),
        };
        let (distinct_count, distinct_flags) = match self.distinct_count {
            Some(count) => (count, INT64_STATS_FLAG_HAS_DISTINCT_COUNT),
            None => (0, 0),
        };
        let flags = min_max_flags | distinct_flags;

        out[0..8].copy_from_slice(&min.to_le_bytes());
        out[8..16].copy_from_slice(&max.to_le_bytes());
        out[16..24].copy_from_slice(&self.null_count.to_le_bytes());
        out[24..32].copy_from_slice(&distinct_count.to_le_bytes());
        out[32..40].copy_from_slice(&flags.to_le_bytes());
        out
    }

    #[inline]
    pub fn deserialize(bytes: &[u8]) -> Result<Self, StatsBlockError> {
        if bytes.len() != Self::SERIALIZED_LEN {
            return Err(StatsBlockError::WrongLength {
                physical_type: ColumnarType::Int64,
                got: bytes.len(),
                expected: Self::SERIALIZED_LEN,
            });
        }

        let min = i64::from_le_bytes(bytes[0..8].try_into().expect("validated stats length"));
        let max = i64::from_le_bytes(bytes[8..16].try_into().expect("validated stats length"));
        let null_count =
            u64::from_le_bytes(bytes[16..24].try_into().expect("validated stats length"));
        let distinct_count =
            u64::from_le_bytes(bytes[24..32].try_into().expect("validated stats length"));
        let flags = u64::from_le_bytes(bytes[32..40].try_into().expect("validated stats length"));

        if flags & !(INT64_STATS_FLAG_HAS_MIN_MAX | INT64_STATS_FLAG_HAS_DISTINCT_COUNT) != 0 {
            return Err(StatsBlockError::InvalidFlags {
                physical_type: ColumnarType::Int64,
                flags,
            });
        }

        let has_min_max = (flags & INT64_STATS_FLAG_HAS_MIN_MAX) != 0;
        let has_distinct_count = (flags & INT64_STATS_FLAG_HAS_DISTINCT_COUNT) != 0;
        let (min, max) = if has_min_max {
            (Some(min), Some(max))
        } else {
            (None, None)
        };

        Ok(Self {
            min,
            max,
            null_count,
            distinct_count: has_distinct_count.then_some(distinct_count),
        })
    }
}

impl ColumnStats {
    #[inline]
    pub fn deserialize(column_type: ColumnarType, bytes: &[u8]) -> Result<Self, StatsBlockError> {
        match column_type {
            ColumnarType::Int64 => Ok(Self::Int64(Int64Stats::deserialize(bytes)?)),
            ColumnarType::Utf8 | ColumnarType::LargeUtf8 => Err(StatsBlockError::UnsupportedPhysicalType {
                physical_type: column_type,
            }),
            _ => Err(StatsBlockError::UnsupportedPhysicalType {
                physical_type: column_type,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int64_stats_round_trip_with_distinct_count() {
        let stats = Int64Stats {
            min: Some(-7),
            max: Some(42),
            null_count: 3,
            distinct_count: Some(9),
        };

        let decoded = Int64Stats::deserialize(&stats.serialize()).expect("decode stats");
        assert_eq!(decoded, stats);
    }

    #[test]
    fn int64_stats_round_trip_without_min_max_or_distinct() {
        let stats = Int64Stats {
            min: None,
            max: None,
            null_count: 8,
            distinct_count: None,
        };

        let decoded = Int64Stats::deserialize(&stats.serialize()).expect("decode stats");
        assert_eq!(decoded, stats);
    }
}
