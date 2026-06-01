//! Tests for D4 symmetries on the Morris board.

use morris_tablebase::symmetry::{apply_transform, canonicalize, NUM_TRANSFORMS, PERMS};

#[test]
fn identity_is_noop() {
    for p in 0..24u8 {
        assert_eq!(PERMS[0][p as usize], p);
    }
}

#[test]
fn rotation_90_four_times_identity() {
    // Each bit position, rotated 4 times by 90° CW, must come back to itself.
    for p in 0..24usize {
        let after_one = PERMS[1][p];
        let after_two = PERMS[1][after_one as usize];
        let after_three = PERMS[1][after_two as usize];
        let after_four = PERMS[1][after_three as usize];
        assert_eq!(after_four, p as u8, "position {} not back to itself", p);
    }
}

#[test]
fn rotation_90_maps_top_to_right_mill() {
    // mill [0,1,2] (outer top) under 90° CW -> mill [2,3,4] (outer right).
    let top = 0b111u32; // bits 0, 1, 2
    let right = apply_transform(top, 1);
    let expected = (1u32 << 2) | (1u32 << 3) | (1u32 << 4);
    assert_eq!(right, expected);
}

#[test]
fn rotation_90_maps_top_spoke_to_right_spoke() {
    // [1,9,17] -> [3,11,19] under 90° CW.
    let top_spoke = (1u32 << 1) | (1u32 << 9) | (1u32 << 17);
    let right_spoke = apply_transform(top_spoke, 1);
    let expected = (1u32 << 3) | (1u32 << 11) | (1u32 << 19);
    assert_eq!(right_spoke, expected);
}

#[test]
fn horizontal_reflection_maps_top_to_bottom() {
    // mill [0,1,2] (outer top) under horizontal reflection -> mill [6,5,4]
    // = same bitmask as [4,5,6] (outer bottom).
    let top = 0b111u32;
    let bot = apply_transform(top, 4);
    let expected = (1u32 << 4) | (1u32 << 5) | (1u32 << 6);
    assert_eq!(bot, expected);
}

#[test]
fn canonicalize_idempotent() {
    // Apply canonicalize twice, must give the same result.
    let wbb = (1u32 << 3) | (1u32 << 11) | (1u32 << 19); // right spoke
    let bbb = (1u32 << 5) | (1u32 << 13) | (1u32 << 21); // bottom spoke
    let (cw1, cb1) = canonicalize(wbb, bbb);
    let (cw2, cb2) = canonicalize(cw1, cb1);
    assert_eq!((cw1, cb1), (cw2, cb2));
}

#[test]
fn canonicalize_consistent_across_orbit() {
    // All 8 D4 variants of a position must canonicalise to the same orbit rep.
    let wbb_base = (1u32 << 0) | (1u32 << 8) | (1u32 << 16);
    let bbb_base = (1u32 << 2) | (1u32 << 10) | (1u32 << 18);
    let canon_base = canonicalize(wbb_base, bbb_base);
    for t in 0..NUM_TRANSFORMS {
        let wbb_t = apply_transform(wbb_base, t);
        let bbb_t = apply_transform(bbb_base, t);
        let canon_t = canonicalize(wbb_t, bbb_t);
        assert_eq!(canon_base, canon_t, "transform {} canonical differs", t);
    }
}
