//! Byte offset alignment helpers for on-disk layout (minimum 8, preferred 64).

/// Returns the number of padding bytes so that `offset + pad_length(..)` is a multiple of `alignment`.
///
/// # Panics
///
/// Panics if `alignment` is zero or not a power of two.
///
/// # Correctness
///
/// For valid `alignment`, let `r = offset % alignment`. By definition of Euclidean remainder,
/// `0 <= r < alignment` and `offset = q * alignment + r` for some `q`.
/// If `r == 0`, the offset is already aligned and `0` padding is correct.
/// Otherwise `alignment - r` is in `1..alignment`, and
/// `offset + (alignment - r) = (q + 1) * alignment`, which is the smallest aligned value
/// strictly greater than `offset` when `r != 0`.
#[inline]
pub fn pad_length(offset: usize, alignment: usize) -> usize {
    assert_power_of_two_alignment(alignment);
    let rem = offset % alignment;
    if rem == 0 {
        0
    } else {
        alignment - rem
    }
}

/// Returns the smallest `aligned >= offset` such that `aligned` is a multiple of `alignment`.
///
/// # Panics
///
/// Panics if `alignment` is zero or not a power of two, or if the aligned offset would exceed
/// `usize::MAX`.
///
/// # Correctness
///
/// `pad_length` gives the unique padding in `[0, alignment)` such that `offset + pad` is
/// divisible by `alignment`. That sum is therefore the least aligned offset not below `offset`.
/// `checked_add` rejects the single case where that sum does not fit in `usize` (unaligned
/// `usize::MAX` with any alignment `> 1`).
#[inline]
pub fn align_offset(offset: usize, alignment: usize) -> usize {
    assert_power_of_two_alignment(alignment);
    let pad = pad_length(offset, alignment);
    offset
        .checked_add(pad)
        .expect("align_offset: aligned offset does not fit in usize")
}

#[inline]
fn assert_power_of_two_alignment(alignment: usize) {
    assert!(
        alignment > 0 && alignment.is_power_of_two(),
        "alignment must be a non-zero power of two, got {alignment}"
    );
}

#[cfg(test)]
mod tests {
    use super::{align_offset, pad_length};

    #[test]
    fn pad_length_zero_offset() {
        assert_eq!(pad_length(0, 8), 0);
        assert_eq!(pad_length(0, 64), 0);
    }

    #[test]
    fn pad_length_already_aligned() {
        assert_eq!(pad_length(8, 8), 0);
        assert_eq!(pad_length(64, 64), 0);
        assert_eq!(pad_length(1024, 8), 0);
    }

    #[test]
    fn pad_length_one_byte_past_alignment() {
        assert_eq!(pad_length(1, 8), 7);
        assert_eq!(pad_length(9, 8), 7);
        assert_eq!(pad_length(1, 64), 63);
        assert_eq!(pad_length(65, 64), 63);
    }

    #[test]
    fn pad_length_just_before_boundary() {
        assert_eq!(pad_length(7, 8), 1);
        assert_eq!(pad_length(63, 64), 1);
    }

    #[test]
    fn align_offset_identity_when_aligned() {
        assert_eq!(align_offset(0, 8), 0);
        assert_eq!(align_offset(8, 8), 8);
        assert_eq!(align_offset(0, 64), 0);
        assert_eq!(align_offset(64, 64), 64);
    }

    #[test]
    fn align_offset_rounds_up() {
        assert_eq!(align_offset(1, 8), 8);
        assert_eq!(align_offset(7, 8), 8);
        assert_eq!(align_offset(9, 8), 16);
        assert_eq!(align_offset(1, 64), 64);
        assert_eq!(align_offset(63, 64), 64);
        assert_eq!(align_offset(65, 64), 128);
    }

    #[test]
    fn offset_plus_pad_equals_align() {
        for alignment in [8usize, 64] {
            for offset in 0usize..500 {
                let pad = pad_length(offset, alignment);
                let aligned = align_offset(offset, alignment);
                assert_eq!(offset + pad, aligned);
                assert_eq!(aligned % alignment, 0);
                assert!(aligned >= offset);
            }
        }
    }

    #[test]
    fn smallest_aligned_at_or_above() {
        for alignment in [8usize, 64] {
            for offset in 0usize..200 {
                let got = align_offset(offset, alignment);
                assert!(got >= offset);
                assert_eq!(got % alignment, 0);
                if offset % alignment == 0 {
                    assert_eq!(got, offset);
                } else {
                    assert!(got > offset);
                    assert!(got < offset + alignment);
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "non-zero power of two")]
    fn alignment_zero_panics_pad() {
        pad_length(0, 0);
    }

    #[test]
    #[should_panic(expected = "non-zero power of two")]
    fn alignment_zero_panics_align() {
        align_offset(0, 0);
    }

    #[test]
    #[should_panic(expected = "non-zero power of two")]
    fn alignment_not_power_of_two_panics() {
        pad_length(0, 3);
    }

    #[test]
    #[should_panic(expected = "does not fit")]
    fn align_offset_overflow_max_unaligned() {
        align_offset(usize::MAX, 8);
    }

    #[test]
    fn align_offset_max_when_aligned() {
        // usize::MAX is odd on 64-bit; use a multiple that fits.
        let alignment = 8usize;
        let offset = usize::MAX - (usize::MAX % alignment);
        assert_eq!(offset % alignment, 0);
        assert_eq!(align_offset(offset, alignment), offset);
        assert_eq!(pad_length(offset, alignment), 0);
    }

    #[test]
    fn large_offsets_8_and_64() {
        let block8 = (usize::MAX / 8).saturating_mul(8);
        assert!(block8 >= 3);
        let off8 = block8 - 3;
        assert_eq!(pad_length(off8, 8), 3);
        assert_eq!(align_offset(off8, 8), block8);

        let block64 = (usize::MAX / 64).saturating_mul(64);
        assert!(block64 >= 3);
        let off64 = block64 - 3;
        assert_eq!(pad_length(off64, 64), 3);
        assert_eq!(align_offset(off64, 64), block64);
    }
}
