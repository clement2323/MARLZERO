//! Smoke tests for mill detection and capture legality.

use morris_tablebase::board::{MILLS, MILL_BITMASKS, MILLS_THROUGH, NUM_POSITIONS};
use morris_tablebase::rules::{all_in_mills, is_mill_through, legal_capture_targets, popcount};

#[test]
fn mill_constants_self_consistent() {
    for (i, mill) in MILLS.iter().enumerate() {
        let expected = (1u32 << mill[0]) | (1u32 << mill[1]) | (1u32 << mill[2]);
        assert_eq!(MILL_BITMASKS[i], expected, "mill {} bitmask mismatch", i);
    }
}

#[test]
fn every_position_in_exactly_two_mills() {
    for p in 0..NUM_POSITIONS {
        let mills = MILLS_THROUGH[p];
        assert_ne!(mills[0], 0, "position {} has no mills", p);
        assert_ne!(mills[1], 0, "position {} has only 1 mill", p);
    }
}

#[test]
fn outer_top_mill_detected() {
    let bb = 0b111u32; // positions 0, 1, 2
    assert!(is_mill_through(bb, 0));
    assert!(is_mill_through(bb, 1));
    assert!(is_mill_through(bb, 2));
}

#[test]
fn incomplete_mill_not_detected() {
    let bb = 0b011u32;
    assert!(!is_mill_through(bb, 0));
    assert!(!is_mill_through(bb, 1));
}

#[test]
fn unrelated_position_not_in_mill() {
    let bb = 0b111u32; // positions 0, 1, 2
    assert!(!is_mill_through(bb, 5));
    assert!(!is_mill_through(bb, 17));
}

#[test]
fn spoke_mill_detected() {
    // mill [1, 9, 17]
    let bb = (1u32 << 1) | (1u32 << 9) | (1u32 << 17);
    assert!(is_mill_through(bb, 1));
    assert!(is_mill_through(bb, 9));
    assert!(is_mill_through(bb, 17));
}

#[test]
fn all_in_mills_true_when_every_piece_locked() {
    // 3 pieces forming one mill (0, 1, 2)
    let bb = 0b111u32;
    assert!(all_in_mills(bb));
}

#[test]
fn all_in_mills_false_when_any_loose() {
    // Mill at (0,1,2) plus a loose piece at 5
    let bb = 0b111u32 | (1 << 5);
    assert!(!all_in_mills(bb));
}

#[test]
fn capture_targets_exclude_mill_pieces_when_alternative_exists() {
    // Opponent has mill (0,1,2) plus loose piece at 5.
    // Only position 5 should be capturable.
    let opp = 0b111u32 | (1 << 5);
    let targets = legal_capture_targets(opp);
    assert_eq!(targets, 1u32 << 5);
}

#[test]
fn capture_targets_include_all_when_all_locked() {
    // Two non-overlapping mills (0,1,2) and (16,17,18) = all 6 pieces in mills.
    let opp = 0b111u32 | (0b111u32 << 16);
    let targets = legal_capture_targets(opp);
    assert_eq!(targets, opp);
}

#[test]
fn popcount_basic() {
    assert_eq!(popcount(0), 0);
    assert_eq!(popcount(0b111), 3);
    assert_eq!(popcount(0xFFFFFF), 24);
}
