//! Board geometry: positions, mills, adjacency.
//!
//! Positions are labelled 0..23 in the same convention as the Python env
//! (see [src/morris_rl/env/board.py](../../../src/morris_rl/env/board.py)):
//! three concentric rings, indices clockwise from each ring's top-left
//! corner, outer ring first. Bitmasks use bit `i` for position `i`.
//!
//! All tables in this module are `const` and computed at compile time so
//! every lookup is a direct array load with no runtime initialisation.

pub const NUM_POSITIONS: usize = 24;
pub const NUM_PIECES_PER_PLAYER: usize = 9;

/// The 16 mills (4 sides × 3 rings + 4 spokes connecting the rings).
pub const MILLS: [[u8; 3]; 16] = [
    [0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 0],
    [8, 9, 10], [10, 11, 12], [12, 13, 14], [14, 15, 8],
    [16, 17, 18], [18, 19, 20], [20, 21, 22], [22, 23, 16],
    [1, 9, 17], [3, 11, 19], [5, 13, 21], [7, 15, 23],
];

/// Bitmask per mill: `(1 << a) | (1 << b) | (1 << c)`. Enables branchless
/// mill detection: `(player_bb & mill_mask) == mill_mask`.
pub const MILL_BITMASKS: [u32; 16] = {
    let mut out = [0u32; 16];
    let mut i = 0;
    while i < 16 {
        let m = MILLS[i];
        out[i] = (1u32 << m[0]) | (1u32 << m[1]) | (1u32 << m[2]);
        i += 1;
    }
    out
};

/// For each position, the bitmasks of mills that include it. Every position
/// belongs to exactly 2 mills (one ring-side + one spoke for midpoints,
/// or two adjacent ring-sides for corners), so two slots suffice.
pub const MILLS_THROUGH: [[u32; 2]; NUM_POSITIONS] = compute_mills_through();

const fn compute_mills_through() -> [[u32; 2]; NUM_POSITIONS] {
    let mut out = [[0u32; 2]; NUM_POSITIONS];
    let mut p: usize = 0;
    while p < NUM_POSITIONS {
        let mut slot = 0;
        let mut m = 0;
        while m < 16 {
            let mill = MILLS[m];
            if mill[0] as usize == p || mill[1] as usize == p || mill[2] as usize == p {
                out[p][slot] = MILL_BITMASKS[m];
                slot += 1;
            }
            m += 1;
        }
        p += 1;
    }
    out
}

/// Adjacency list per position. Each inner array is null-terminated with
/// 0xFF so we can iterate without storing a separate length. Longest list
/// is 4 (middle-ring spoke midpoints).
pub const ADJACENCY: [[u8; 4]; NUM_POSITIONS] = [
    [1, 7, 0xFF, 0xFF],     // 0  outer TL
    [0, 2, 9, 0xFF],        // 1  outer TM
    [1, 3, 0xFF, 0xFF],     // 2  outer TR
    [2, 4, 11, 0xFF],       // 3  outer MR
    [3, 5, 0xFF, 0xFF],     // 4  outer BR
    [4, 6, 13, 0xFF],       // 5  outer BM
    [5, 7, 0xFF, 0xFF],     // 6  outer BL
    [6, 0, 15, 0xFF],       // 7  outer ML
    [9, 15, 0xFF, 0xFF],    // 8  middle TL
    [8, 10, 1, 17],         // 9  middle TM
    [9, 11, 0xFF, 0xFF],    // 10 middle TR
    [10, 12, 3, 19],        // 11 middle MR
    [11, 13, 0xFF, 0xFF],   // 12 middle BR
    [12, 14, 5, 21],        // 13 middle BM
    [13, 15, 0xFF, 0xFF],   // 14 middle BL
    [14, 8, 7, 23],         // 15 middle ML
    [17, 23, 0xFF, 0xFF],   // 16 inner TL
    [16, 18, 9, 0xFF],      // 17 inner TM
    [17, 19, 0xFF, 0xFF],   // 18 inner TR
    [18, 20, 11, 0xFF],     // 19 inner MR
    [19, 21, 0xFF, 0xFF],   // 20 inner BR
    [20, 22, 13, 0xFF],     // 21 inner BM
    [21, 23, 0xFF, 0xFF],   // 22 inner BL
    [22, 16, 15, 0xFF],     // 23 inner ML
];
