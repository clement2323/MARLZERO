//! Work units for multi-valued retrograde analysis (Phase 2).
//!
//! Per Gévay-Danner 2014 Section II: when we drop black-to-move positions
//! and use color-swap symmetry, sliding moves cross between subspace `s`
//! and its negation `-s = swap(w_board, b_board) × swap(w_to_place, b_to_place)`.
//! This breaks the strict DAG of Phase 1.
//!
//! Solution: process pairs `(s, -s)` together as a single "work unit",
//! handling the internal cycles atomically. ESC (Equal Stone Count)
//! subspaces are self-negations (s == -s) and form single-subspace WUs.
//!
//! A WU is "transient" iff every move from every position leaves the WU
//! (only happens in placement phase, never in pure movement — sliding
//! moves always stay within the WU).

use crate::subspace::Subspace;

/// A work unit groups 1 or 2 primary subspaces that must be processed
/// together because sliding moves create cycles between them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkUnit {
    /// 1 element for ESC subspaces, 2 elements for non-ESC pairs (s, -s).
    pub primary: Vec<Subspace>,
    pub is_esc: bool,
    pub is_transient: bool,
}

impl WorkUnit {
    pub fn esc(sub: Subspace) -> Self {
        Self {
            primary: vec![sub],
            is_esc: true,
            is_transient: false,
        }
    }

    pub fn pair(s: Subspace, neg_s: Subspace) -> Self {
        debug_assert_eq!(negate(s), neg_s, "pair primary subspaces must be negations");
        Self {
            primary: vec![s, neg_s],
            is_esc: false,
            is_transient: false,
        }
    }

    /// Total piece count is constant across a WU (negation preserves it).
    pub fn total_pieces(&self) -> u8 {
        let p = &self.primary[0];
        p.w_board + p.b_board + p.w_to_place + p.b_to_place
    }
}

/// Negate a subspace by swapping the white/black piece counts.
/// Used to express the color-swap symmetry on subspace identifiers.
#[inline]
pub fn negate(sub: Subspace) -> Subspace {
    Subspace {
        w_board: sub.b_board,
        b_board: sub.w_board,
        w_to_place: sub.b_to_place,
        b_to_place: sub.w_to_place,
    }
}

/// All movement work units with `w + b ≤ max_total`, in topological order
/// (ascending total piece count → smaller WUs solved first, used as
/// secondary subspaces by larger WUs).
pub fn list_movement_work_units(max_total: u8) -> Vec<WorkUnit> {
    let mut out: Vec<WorkUnit> = Vec::new();
    let mut seen: std::collections::HashSet<Subspace> = std::collections::HashSet::new();

    let cap = max_total.min(18);
    for total in 6..=cap {
        for w in 3..=9u8 {
            let b_signed = total as i32 - w as i32;
            if !(3..=9).contains(&b_signed) {
                continue;
            }
            let b = b_signed as u8;
            let sub = Subspace::movement(w, b);
            if seen.contains(&sub) {
                continue;
            }
            if w == b {
                out.push(WorkUnit::esc(sub));
                seen.insert(sub);
            } else {
                let neg = negate(sub);
                if !seen.contains(&neg) {
                    out.push(WorkUnit::pair(sub, neg));
                    seen.insert(sub);
                    seen.insert(neg);
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negate_is_involution() {
        let s = Subspace::movement(4, 3);
        assert_eq!(negate(s), Subspace::movement(3, 4));
        assert_eq!(negate(negate(s)), s);
    }

    #[test]
    fn esc_self_negation() {
        let s = Subspace::movement(5, 5);
        assert_eq!(negate(s), s);
    }

    #[test]
    fn list_movement_work_units_count() {
        // Movement: 3 ≤ w,b ≤ 9.
        // ESC: (3,3), (4,4), …, (9,9) → 7 single-subspace WUs.
        // Non-ESC pairs: C(7, 2) = 21 pairs.
        // Total: 7 + 21 = 28 work units (covers all 49 (w,b) ordered pairs).
        let wus = list_movement_work_units(18);
        assert_eq!(wus.len(), 28);

        let esc_count = wus.iter().filter(|w| w.is_esc).count();
        assert_eq!(esc_count, 7);
        let pair_count = wus.iter().filter(|w| !w.is_esc).count();
        assert_eq!(pair_count, 21);
    }

    #[test]
    fn topological_order_by_total_pieces() {
        let wus = list_movement_work_units(18);
        let mut prev_total = 0u8;
        for wu in &wus {
            let t = wu.total_pieces();
            assert!(
                t >= prev_total,
                "WU {:?} (total={}) appears after a WU with total={}",
                wu, t, prev_total
            );
            prev_total = t;
        }
    }

    #[test]
    fn pair_primaries_are_negations_of_each_other() {
        let wus = list_movement_work_units(18);
        for wu in &wus {
            if !wu.is_esc {
                assert_eq!(wu.primary.len(), 2);
                assert_eq!(wu.primary[1], negate(wu.primary[0]));
            }
        }
    }

    #[test]
    fn max_total_caps_work_units() {
        let wus_8 = list_movement_work_units(8);
        // Totals 6, 7, 8: (3,3) ESC + (3,4)/(4,3) pair + (3,5)/(5,3) pair + (4,4) ESC
        assert_eq!(wus_8.len(), 4);
    }
}
