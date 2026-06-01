//! Morris tablebase: retrograde analysis for Nine Men's Morris.
//!
//! Phase 1 reproduces Gasser 1996 (with-flying variant). See
//! [docs/decisions/002-phase1-gasser-tables.md](../../docs/decisions/002-phase1-gasser-tables.md)
//! for the design and validation strategy.

pub mod board;
pub mod hash;
pub mod rules;
pub mod storage;
pub mod subspace;
pub mod symmetry;
pub mod wave;
