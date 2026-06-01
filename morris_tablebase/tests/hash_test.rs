//! Tests for combinatorial ranking and compaction.

use morris_tablebase::hash::{
    compact_against, expand_against, rank_subset, unrank_subset, BINOM,
};

#[test]
fn binom_known_values() {
    assert_eq!(BINOM[0][0], 1);
    assert_eq!(BINOM[10][0], 1);
    assert_eq!(BINOM[10][10], 1);
    assert_eq!(BINOM[24][3], 2024);
    assert_eq!(BINOM[21][3], 1330);
    assert_eq!(BINOM[24][12], 2_704_156);
}

#[test]
fn binom_zero_when_k_gt_n() {
    assert_eq!(BINOM[3][5], 0);
    assert_eq!(BINOM[0][1], 0);
}

#[test]
fn rank_smallest_subset_is_zero() {
    // {0, 1, 2} -> C(0,1) + C(1,2) + C(2,3) = 0 + 0 + 0 = 0
    let bb = 0b111u32;
    assert_eq!(rank_subset(bb), 0);
}

#[test]
fn rank_largest_subset_is_c_minus_1() {
    // {21, 22, 23} -> C(21,1) + C(22,2) + C(23,3) = 21 + 231 + 1771 = 2023
    let bb = (1u32 << 21) | (1u32 << 22) | (1u32 << 23);
    assert_eq!(rank_subset(bb), 2023);
    assert_eq!(rank_subset(bb), BINOM[24][3] - 1);
}

#[test]
fn rank_specific_subset() {
    // {0, 1, 3} -> 0 + 0 + 1 = 1
    let bb = (1u32 << 0) | (1u32 << 1) | (1u32 << 3);
    assert_eq!(rank_subset(bb), 1);
    // {0, 2, 3} -> 0 + 1 + 1 = 2
    let bb = (1u32 << 0) | (1u32 << 2) | (1u32 << 3);
    assert_eq!(rank_subset(bb), 2);
    // {1, 2, 3} -> 1 + 1 + 1 = 3
    let bb = (1u32 << 1) | (1u32 << 2) | (1u32 << 3);
    assert_eq!(rank_subset(bb), 3);
}

#[test]
fn rank_unrank_roundtrip_3_of_24() {
    for rank in 0..BINOM[24][3] {
        let bb = unrank_subset(rank, 24, 3);
        assert_eq!(bb.count_ones(), 3, "rank {} -> wrong popcount", rank);
        assert_eq!(rank_subset(bb), rank, "rank({:#x}) != {}", bb, rank);
    }
}

#[test]
fn rank_unrank_roundtrip_3_of_21() {
    for rank in 0..BINOM[21][3] {
        let bb = unrank_subset(rank, 21, 3);
        assert_eq!(bb.count_ones(), 3);
        assert_eq!(rank_subset(bb), rank);
    }
}

#[test]
fn compact_expand_roundtrip() {
    // Use whites = {3, 7, 10} (3 set bits), blacks must be disjoint.
    let whites = (1u32 << 3) | (1u32 << 7) | (1u32 << 10);
    // For every 3-subset of the remaining 21 positions, compact+expand
    // must return the original.
    for rank in 0..BINOM[21][3] {
        let compact = unrank_subset(rank, 21, 3);
        let blacks = expand_against(compact, whites);
        assert_eq!(blacks & whites, 0, "blacks overlap whites at rank {}", rank);
        assert_eq!(blacks.count_ones(), 3);
        let compact_back = compact_against(blacks, whites);
        assert_eq!(compact_back, compact, "roundtrip failed for rank {}", rank);
    }
}
