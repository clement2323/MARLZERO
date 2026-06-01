//! Combinatorial ranking for piece-set bitmasks.
//!
//! For a sorted k-subset `{a_0 < a_1 < ... < a_{k-1}}` of `{0..n}`, the
//! colex rank is `Σ_{i=0..k} C(a_i, i+1)`. The rank is a bijection onto
//! `[0, C(n, k))` and is what indexes positions inside a subspace.

pub const MAX_N: usize = 25; // need C(24, k), 0 ≤ k ≤ 24

/// Pascal triangle, `BINOM[n][k] = C(n, k)`, zero for `k > n`.
pub const BINOM: [[u32; MAX_N]; MAX_N] = compute_binom();

const fn compute_binom() -> [[u32; MAX_N]; MAX_N] {
    let mut out = [[0u32; MAX_N]; MAX_N];
    let mut n = 0;
    while n < MAX_N {
        out[n][0] = 1;
        let mut k = 1;
        while k <= n {
            let left = out[n - 1][k - 1];
            let right = if k <= n - 1 { out[n - 1][k] } else { 0 };
            out[n][k] = left + right;
            k += 1;
        }
        n += 1;
    }
    out
}

/// Colex rank of the k-subset encoded as set bits of `bb`. Returns 0..C(n,k).
/// Reads bits via `trailing_zeros` and runs `popcount(bb)` iterations.
#[inline]
pub fn rank_subset(bb: u32) -> u32 {
    let mut b = bb;
    let mut i: usize = 1;
    let mut sum: u32 = 0;
    while b != 0 {
        let a = b.trailing_zeros() as usize;
        sum += BINOM[a][i];
        b &= b - 1;
        i += 1;
    }
    sum
}

/// Inverse of `rank_subset`: recover the k-subset bitmask from its rank.
/// `n` is the universe size (positions are 0..n) and `k` is the cardinality.
#[inline]
pub fn unrank_subset(rank: u32, n: u32, k: u32) -> u32 {
    let mut out = 0u32;
    let mut remaining = rank;
    let mut i = k as usize;
    let mut high = (n - 1) as usize;
    while i > 0 {
        // Find the largest a such that BINOM[a][i] <= remaining.
        while BINOM[high][i] > remaining {
            high -= 1;
        }
        out |= 1u32 << high;
        remaining -= BINOM[high][i];
        if high == 0 {
            break;
        }
        high -= 1;
        i -= 1;
    }
    out
}

/// Remove the bits of `removed_bb` from the universe and re-pack `target_bb`
/// into the compacted index space. Used to project a "blacks" bitmask onto
/// `[0..24 - popcount(whites_bb))` before ranking.
#[inline]
pub fn compact_against(target_bb: u32, removed_bb: u32) -> u32 {
    let mut out = 0u32;
    let mut t = target_bb;
    while t != 0 {
        let p = t.trailing_zeros();
        let nshifted = (removed_bb & ((1u32 << p) - 1)).count_ones();
        out |= 1u32 << (p - nshifted);
        t &= t - 1;
    }
    out
}

/// Inverse of `compact_against`: expand `compact_bb` (in the compacted
/// universe of size `24 - popcount(removed_bb)`) back into positions
/// `0..24` by skipping over set bits of `removed_bb`.
#[inline]
pub fn expand_against(compact_bb: u32, removed_bb: u32) -> u32 {
    let mut out = 0u32;
    let mut c = compact_bb;
    while c != 0 {
        let cp = c.trailing_zeros();
        // Find the actual position: walk through 0..24, skipping bits set
        // in `removed_bb`, and pick the cp-th unset bit (0-indexed).
        let mut count = 0u32;
        let mut p = 0u32;
        loop {
            if (removed_bb >> p) & 1 == 0 {
                if count == cp {
                    break;
                }
                count += 1;
            }
            p += 1;
        }
        out |= 1u32 << p;
        c &= c - 1;
    }
    out
}
