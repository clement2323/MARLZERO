//! D4 dihedral symmetries of the Morris board.
//!
//! The 8 transforms decompose as `4 rotations × 2 reflections`. Each ring
//! (outer 0..8, middle 8..16, inner 16..24) is rotated within itself; the
//! 4 spokes (1,9,17 / 3,11,19 / 5,13,21 / 7,15,23) rotate together because
//! all three rings rotate by the same angle.
//!
//! Color swap is a separate (commuting) symmetry — kept in [crate::rules]
//! land since it interacts with the side-to-move convention rather than
//! board geometry.

pub const NUM_TRANSFORMS: usize = 8;

/// `PERMS[t][p]` = position that `p` maps to under transform `t`.
/// `t = 0..4` are rotations by `90° × t` CW. `t = 4..8` add a horizontal
/// reflection on top of the rotation.
pub const PERMS: [[u8; 24]; NUM_TRANSFORMS] = compute_perms();

const fn compute_perms() -> [[u8; 24]; NUM_TRANSFORMS] {
    let mut out = [[0u8; 24]; NUM_TRANSFORMS];
    let mut t = 0;
    while t < NUM_TRANSFORMS {
        let rot = t % 4;
        let refl = t / 4;
        let mut p = 0;
        while p < 24 {
            let ring = p / 8;
            let k = p % 8;
            let k_rot = (k + 2 * rot) % 8;
            // Horizontal reflection on outer ring (labels TL=0, TM=1, ...
            // ML=7, going clockwise): the axis is horizontal so TL<->BL,
            // TM<->BM, TR<->BR; MR=3 and ML=7 are on the axis and stay.
            // Closed form: k_refl = (6 - k) mod 8 for k in {0..6}; k=7 -> 7.
            // Verifies as (14 - k) % 8 across all 8 cases.
            let k_final = if refl == 0 { k_rot } else { (14 - k_rot) % 8 };
            out[t][p] = (ring * 8 + k_final) as u8;
            p += 1;
        }
        t += 1;
    }
    out
}

/// Apply transform `t` to a bitmask. Returns the bitmask of remapped bits.
#[inline]
pub fn apply_transform(bb: u32, t: usize) -> u32 {
    let perm = &PERMS[t];
    let mut out = 0u32;
    let mut b = bb;
    while b != 0 {
        let p = b.trailing_zeros() as usize;
        out |= 1u32 << perm[p];
        b &= b - 1;
    }
    out
}

/// Return the lex-min `(whites_bb, blacks_bb)` over the 8 D4 transforms.
/// This is the canonical orbit representative under D4 (color swap is
/// orthogonal and handled separately when collapsing tables across STMs).
#[inline]
pub fn canonicalize(wbb: u32, bbb: u32) -> (u32, u32) {
    let mut best = (wbb, bbb);
    let mut t = 1;
    while t < NUM_TRANSFORMS {
        let w = apply_transform(wbb, t);
        let b = apply_transform(bbb, t);
        if (w, b) < best {
            best = (w, b);
        }
        t += 1;
    }
    best
}

/// Orbit size of `(wbb, bbb)` under D4. By orbit-stabilizer theorem:
/// orbit_size = |G| / |Stab(p)|. Most positions have trivial stabilizer
/// (only identity fixes them) so orbit_size = 8; positions on symmetry
/// axes have smaller orbits {1, 2, 4}.
#[inline]
pub fn orbit_size(wbb: u32, bbb: u32) -> u32 {
    let mut stab = 1u32; // identity always fixes p
    let mut t = 1;
    while t < NUM_TRANSFORMS {
        let w = apply_transform(wbb, t);
        let b = apply_transform(bbb, t);
        if (w, b) == (wbb, bbb) {
            stab += 1;
        }
        t += 1;
    }
    NUM_TRANSFORMS as u32 / stab
}
