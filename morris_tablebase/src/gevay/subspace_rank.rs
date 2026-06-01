//! Section IV-A — heuristic value `val_s` per subspace and ordinal ranking.
//!
//! For each work unit we compute a single scalar `val_s ∈ [0, 1]` from the
//! Phase 1 W/D/L statistics, capturing "how good is this subspace for the
//! side to move". Ranks are then assigned ordinally with the constraint
//! `rank(s) = -rank(-s)`, centred around 0.
//!
//! The paper formula (non-transient work unit):
//! ```text
//! val_s = (W_s + L_{-s} + D_s/2 + D_{-s}/2) / (T_s + T_{-s})
//! ```
//!
//! For ESC work units (s == -s), this reduces to 0.5 — all ESC therefore
//! share rank 0. For non-ESC pairs, `val_s + val_{-s} = 1` is a hard
//! algebraic identity, which guarantees pairs are antipodal in ranking.

use std::collections::HashMap;

use crate::subspace::{Subspace, SubspaceTable};
use crate::symmetry::orbit_size;
use crate::wave::{DRAW, LOSS, STM_BLACK, STM_WHITE, WIN};
use crate::work_unit::{negate, WorkUnit};

/// Orbit-weighted counts split by side-to-move.
#[derive(Default, Debug, Clone, Copy)]
pub struct StmCounts {
    pub wins: u64,
    pub losses: u64,
    pub draws: u64,
}

impl StmCounts {
    pub fn total(&self) -> u64 {
        self.wins + self.losses + self.draws
    }
}

/// Per-subspace counts split by STM. White-to-move counts feed `val_s`;
/// black-to-move counts are kept for sanity checks (Invariant A says they
/// must match the white-to-move counts of `negate(s)`).
#[derive(Default, Debug, Clone, Copy)]
pub struct SubspaceStats {
    pub white_to_move: StmCounts,
    pub black_to_move: StmCounts,
}

/// Walk a [SubspaceTable]'s canonical positions and sum orbit-weighted
/// counts per STM. O(n_canonical) — fast even on the largest subspaces.
pub fn count_stats(table: &SubspaceTable) -> SubspaceStats {
    let sub = table.subspace;
    let mut out = SubspaceStats::default();
    sub.enumerate_positions(|wbb, bbb| {
        let osize = orbit_size(wbb, bbb) as u64;
        for stm in [STM_WHITE, STM_BLACK] {
            let idx = sub.state_index_canonical(wbb, bbb, stm) as usize;
            let v = table.verdict[idx];
            let bucket = if stm == STM_WHITE {
                &mut out.white_to_move
            } else {
                &mut out.black_to_move
            };
            match v {
                WIN => bucket.wins += osize,
                LOSS => bucket.losses += osize,
                DRAW => bucket.draws += osize,
                _ => {}
            }
        }
    });
    out
}

/// Compute `val_s` for one work unit given its primary subspaces' stats.
///
/// Inputs map `Subspace -> StmCounts` for the white-to-move side. The
/// formula handles both ESC (single-subspace) and pair work units.
/// Returns the value attached to `wu.primary[0]`; the other member of
/// a pair has `1.0 - returned`.
pub fn compute_val(
    wu: &WorkUnit,
    wtm_counts: &HashMap<Subspace, StmCounts>,
) -> f64 {
    if wu.is_transient {
        // Transient WU formula: val_s = (W_s + D_s/2) / T_s.
        let s = wu.primary[0];
        let c = wtm_counts.get(&s).copied().unwrap_or_default();
        let t = c.total();
        if t == 0 {
            return 0.5;
        }
        return (c.wins as f64 + c.draws as f64 / 2.0) / t as f64;
    }

    // Non-transient: pair (s, -s) (or s == -s for ESC).
    let s = wu.primary[0];
    let neg = if wu.is_esc { s } else { wu.primary[1] };
    debug_assert_eq!(negate(s), neg);

    let cs = wtm_counts.get(&s).copied().unwrap_or_default();
    let cn = wtm_counts.get(&neg).copied().unwrap_or_default();
    let denom = cs.total() + cn.total();
    if denom == 0 {
        return 0.5;
    }
    let numer = cs.wins as f64
        + cn.losses as f64
        + cs.draws as f64 / 2.0
        + cn.draws as f64 / 2.0;
    numer / denom as f64
}

/// Final assigned rank for a subspace. Pairs are antipodal: `rank(s) = -rank(-s)`.
/// ESC and ties at 0.5 collapse to rank 0.
pub type Rank = i16;

/// Manual correction registry, e.g. lower the rank of `(8, 9, 0, 0)` per
/// paper Section IV-A ("the rank ... seemed too high … so we manually
/// lowered it"). Keys are subspaces; values are absolute rank overrides.
#[derive(Default, Debug, Clone)]
pub struct RankOverrides {
    pub overrides: HashMap<Subspace, Rank>,
}

impl RankOverrides {
    pub fn empty() -> Self {
        Self::default()
    }

    /// Default overrides per the paper. Currently just `(8, 9, 0, 0)`
    /// lowered to a small absolute rank — exact target TBD from paper Table.
    pub fn paper_defaults() -> Self {
        let mut out = Self::default();
        // Placeholder: the paper does not publish the exact correction
        // value; we mark it for future tuning during cross-check.
        let _ = out.overrides.insert(Subspace::movement(8, 9), -1);
        let _ = out.overrides.insert(Subspace::movement(9, 8), 1);
        out
    }
}

/// Assigned ranks for every primary subspace of a list of work units.
///
/// Algorithm:
/// 1. Compute `val_s` for each WU.
/// 2. Separate ESC (val == 0.5) from non-ESC.
/// 3. Sort non-ESC WUs by `val_s` ascending. Assign ranks `-N, …, -1` to
///    the canonical sides (val < 0.5) and `+1, …, +N` to their negations
///    in matched order, so pairs are antipodal.
/// 4. All ESC subspaces get rank 0.
/// 5. Apply any `RankOverrides` after ordinal assignment.
pub fn assign_ranks(
    units: &[WorkUnit],
    wtm_counts: &HashMap<Subspace, StmCounts>,
    overrides: &RankOverrides,
) -> HashMap<Subspace, Rank> {
    let mut ranks: HashMap<Subspace, Rank> = HashMap::new();

    // Split ESC (rank 0) from pairs (need ordering).
    let mut pair_vals: Vec<(Subspace, Subspace, f64)> = Vec::new();
    for wu in units {
        let v = compute_val(wu, wtm_counts);
        if wu.is_esc {
            ranks.insert(wu.primary[0], 0);
        } else {
            // Canonical orientation: store (s_low_val, s_high_val) so
            // the lower-valued side gets the negative rank.
            let s0 = wu.primary[0];
            let s1 = wu.primary[1];
            if v <= 0.5 {
                pair_vals.push((s0, s1, v));
            } else {
                pair_vals.push((s1, s0, 1.0 - v));
            }
        }
    }

    // Sort pairs by their "low side" val ascending. Most pessimistic first.
    pair_vals.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

    // Assign ranks: the most pessimistic pair gets ±N, then ±(N-1), …, ±1.
    let n = pair_vals.len() as i16;
    for (i, (low_side, high_side, _v)) in pair_vals.iter().enumerate() {
        let magnitude = n - i as i16; // N at position 0, 1 at position N-1
        ranks.insert(*low_side, -magnitude);
        ranks.insert(*high_side, magnitude);
    }

    // Apply manual overrides last (e.g., paper's 8,9,0,0 correction).
    for (sub, rank) in &overrides.overrides {
        ranks.insert(*sub, *rank);
    }

    ranks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::work_unit::list_movement_work_units;

    fn dummy_counts(w: u64, l: u64, d: u64) -> StmCounts {
        StmCounts { wins: w, losses: l, draws: d }
    }

    #[test]
    fn esc_val_is_half() {
        // ESC (3, 3) — any non-zero counts give val = 0.5.
        let s = Subspace::movement(3, 3);
        let mut map = HashMap::new();
        map.insert(s, dummy_counts(100, 50, 200)); // arbitrary
        let wu = WorkUnit::esc(s);
        let v = compute_val(&wu, &map);
        assert!((v - 0.5).abs() < 1e-9, "ESC val should be 0.5, got {v}");
    }

    #[test]
    fn pair_vals_sum_to_one() {
        // Non-ESC pair: val(s) + val(-s) must equal 1 exactly.
        // Take asymmetric (4, 3) and (3, 4) using mock counts that
        // reflect Invariant A: (3,4 wtm) verdicts mirror (4,3 btm).
        let s = Subspace::movement(4, 3);
        let neg = Subspace::movement(3, 4);
        let mut map = HashMap::new();
        // (4,3 wtm): 1200 wins, 50 losses, 10000 draws
        map.insert(s, dummy_counts(1200, 50, 10000));
        // (3,4 wtm) = (4,3 btm) under Invariant A: 1600 wins, 0 losses, 10500 draws
        map.insert(neg, dummy_counts(1600, 0, 10500));
        let wu = WorkUnit::pair(s, neg);
        let v_s = compute_val(&wu, &map);
        // val_{-s} = numer({-s, s}) / denom = (1600 + 50 + 10500/2 + 10000/2)/total
        // By construction val_s + val_{-s} == 1.
        // Compute by hand: total = 11250 + 12100 = 23350
        // val_s numer = 1200 + 0 + 10000/2 + 10500/2 = 1200 + 5000 + 5250 = 11450
        // val_s = 11450 / 23350 ≈ 0.4903
        // val_{-s} numer = 1600 + 50 + 10500/2 + 10000/2 = 1650 + 5250 + 5000 = 11900
        // val_{-s} = 11900 / 23350 ≈ 0.5097
        // sum = 0.4903 + 0.5097 = 1.0 ✓
        assert!((v_s - 11450.0 / 23350.0).abs() < 1e-9);
        let wu_neg = WorkUnit::pair(neg, s);
        let v_neg = compute_val(&wu_neg, &map);
        assert!((v_s + v_neg - 1.0).abs() < 1e-9);
    }

    #[test]
    fn assign_ranks_esc_all_zero() {
        // All-ESC fake list — every rank should be 0.
        let wus = vec![
            WorkUnit::esc(Subspace::movement(3, 3)),
            WorkUnit::esc(Subspace::movement(4, 4)),
        ];
        let mut counts = HashMap::new();
        counts.insert(Subspace::movement(3, 3), dummy_counts(1, 1, 1));
        counts.insert(Subspace::movement(4, 4), dummy_counts(1, 1, 1));
        let ranks = assign_ranks(&wus, &counts, &RankOverrides::empty());
        assert_eq!(ranks[&Subspace::movement(3, 3)], 0);
        assert_eq!(ranks[&Subspace::movement(4, 4)], 0);
    }

    #[test]
    fn assign_ranks_pairs_are_antipodal() {
        let wus = list_movement_work_units(9); // small set
        let mut counts: HashMap<Subspace, StmCounts> = HashMap::new();
        // Fake stats: every (w, b) gets counts proportional to w (so (4,3)
        // beats (3,4) deterministically).
        for wu in &wus {
            for p in &wu.primary {
                counts.insert(*p, dummy_counts(p.w_board as u64 * 100, 10, 1000));
            }
        }
        let ranks = assign_ranks(&wus, &counts, &RankOverrides::empty());
        // For every non-ESC pair, the two primary subspaces must be antipodal.
        for wu in &wus {
            if !wu.is_esc {
                let s = wu.primary[0];
                let neg = wu.primary[1];
                assert_eq!(ranks[&s], -ranks[&neg],
                    "WU {:?} not antipodal: {} vs {}", wu, ranks[&s], ranks[&neg]);
            }
        }
    }

    #[test]
    fn assign_ranks_uses_overrides() {
        let wus = list_movement_work_units(9);
        let mut counts: HashMap<Subspace, StmCounts> = HashMap::new();
        for wu in &wus {
            for p in &wu.primary {
                counts.insert(*p, dummy_counts(100, 10, 1000));
            }
        }
        let mut overrides = RankOverrides::empty();
        // Force (3, 4) to rank +99 — an arbitrary high value to be sure.
        overrides.overrides.insert(Subspace::movement(3, 4), 99);
        let ranks = assign_ranks(&wus, &counts, &overrides);
        assert_eq!(ranks[&Subspace::movement(3, 4)], 99);
    }
}
