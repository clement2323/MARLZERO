//! Rules: mill detection, legal moves, captures.
//!
//! The variant flag (with-flying vs no-flying) lives at the move-generation
//! call sites, not here — `is_mill_through` and `all_in_mills` are variant
//! independent. See [crate::board] for geometry constants.

use crate::board::{MILLS_THROUGH, NUM_POSITIONS};

/// True if `player_bb` contains a complete mill that includes `pos`.
///
/// Branchless: each position is in at most 2 mills, both pre-encoded as
/// bitmasks in [`MILLS_THROUGH`]. The check is `(bb & mill) == mill`.
#[inline]
pub fn is_mill_through(player_bb: u32, pos: u8) -> bool {
    let mills = MILLS_THROUGH[pos as usize];
    let m0 = mills[0];
    let m1 = mills[1];
    ((player_bb & m0) == m0) || (m1 != 0 && (player_bb & m1) == m1)
}

/// True if every piece in `player_bb` is part of some mill in `player_bb`.
///
/// Used for the capture rule: an opponent piece in a mill can only be
/// captured if ALL opponent pieces are in mills.
#[inline]
pub fn all_in_mills(player_bb: u32) -> bool {
    let mut bb = player_bb;
    while bb != 0 {
        let pos = bb.trailing_zeros() as u8;
        if !is_mill_through(player_bb, pos) {
            return false;
        }
        bb &= bb - 1; // clear lowest set bit
    }
    true
}

/// Bitmask of opponent pieces that can be legally captured.
///
/// Without the all-in-mills exception: any non-mill opponent piece.
/// With the exception (all opponent pieces in mills): any opponent piece.
#[inline]
pub fn legal_capture_targets(opponent_bb: u32) -> u32 {
    if all_in_mills(opponent_bb) {
        return opponent_bb;
    }
    let mut bb = opponent_bb;
    let mut out: u32 = 0;
    while bb != 0 {
        let pos = bb.trailing_zeros();
        if !is_mill_through(opponent_bb, pos as u8) {
            out |= 1 << pos;
        }
        bb &= bb - 1;
    }
    out
}

/// Population count of a position bitmask, capped at 24.
#[inline]
pub fn popcount(bb: u32) -> u32 {
    (bb & ((1u32 << NUM_POSITIONS) - 1)).count_ones()
}
