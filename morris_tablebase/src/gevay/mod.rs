//! Gévay-Danner 2014 ultra-strong solver: multi-valued retrograde analysis
//! that classifies draws by tension toward the "stable draw subspace"
//! eventually reached with optimal play.
//!
//! See paper Section IV. Implementation split:
//! - [`subspace_rank`] : Section IV-A heuristic — compute `val_s` from
//!   Phase 1 W/D/L stats, ordinally rank, manual correction for 8,9,0,0.
//! - `multi_value` (TODO) : Section IV-B multi-valued retrograde wave.
//! - `dtw_adjusted` (TODO) : Section IV-B-2 DTW direction by first-key sign.

pub mod canonical_indexer;
pub mod multi_value;
pub mod subspace_rank;
