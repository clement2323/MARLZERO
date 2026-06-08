//! `cargo run --release --bin play_tb -- <PHASE1_DIR> [--side white|black]`
//!
//! Human-vs-tablebase Morris CLI. The computer plays **perfectly** by
//! querying the Phase 1 tablebase: from any position, pick the move that
//! achieves the best game-theoretic outcome (WIN, then DRAW, then LOSS),
//! breaking ties by DTW (fastest win / slowest loss).
//!
//! Game starts in the **movement phase** at (3,3) by default (3 white, 3
//! black, both can fly). To play a different starting position, edit the
//! `startup_white` / `startup_black` literals below.
//!
//! Notation matches `scripts/play_human.py`:
//!   a7 d7 g7 / a4 g4 / a1 d1 g1 (outer)
//!   b6 d6 f6 / b4 f4 / b2 d2 f2 (middle)
//!   c5 d5 e5 / c4 e4 / c3 d3 e3 (inner)

use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

use morris_tablebase::board::{ADJACENCY, NUM_POSITIONS};
use morris_tablebase::gevay::canonical_indexer::CanonicalIndexer;
use morris_tablebase::gevay::multi_value::WIN_ABS;
use morris_tablebase::rules::{is_mill_through, legal_capture_targets, popcount};
use morris_tablebase::storage::{default_filename, gevay_filename, load_gevay_canonical_mmap, MmapGevay};
use morris_tablebase::subspace::{MappedTable, Subspace, Tablebase};
use morris_tablebase::wave::{DRAW, LOSS, Variant, WIN};

/// Loaded V_Gévay tables for the JSONL `--gevay-dir` query mode. Keyed by
/// movement subspace; each value bundles a lazily-built CanonicalIndexer
/// (wrapped in OnceCell — first query for a subspace pays the build cost,
/// subsequent queries are O(1) hashmap lookups) with the mmap-backed
/// `first_key` / `dtw` slices.
///
/// Why lazy + mmap: loading all 49 Gévay files into `Vec<i16>` would
/// commit ~107 GB of RAM per process and OOM-kill any multi-worker
/// training run. The mmap approach defers page residency to the kernel
/// page cache (shared across processes) and the lazy indexer build
/// avoids the ~5-50s cost for subspaces that never get queried.
type GevayStore = std::collections::HashMap<Subspace, (std::sync::OnceLock<CanonicalIndexer>, MmapGevay)>;

const POSITION_LABELS: [&str; 24] = [
    "a7", "d7", "g7", "g4", "g1", "d1", "a1", "a4",
    "b6", "d6", "f6", "f4", "f2", "d2", "b2", "b4",
    "c5", "d5", "e5", "e4", "e3", "d3", "c3", "c4",
];

fn label_of(pos: u8) -> &'static str {
    POSITION_LABELS[pos as usize]
}

fn parse_label(s: &str) -> Option<u8> {
    POSITION_LABELS.iter().position(|&l| l.eq_ignore_ascii_case(s)).map(|p| p as u8)
}

// ANSI escape codes matching scripts/replay_game.py
const ANSI_RESET: &str = "\x1b[0m";
const ANSI_BOLD: &str = "\x1b[1m";
const ANSI_YELLOW: &str = "\x1b[33m";
const ANSI_BLUE: &str = "\x1b[34m";

// 2D grid coordinates (row, col) for each of the 24 board positions.
// Identical to scripts/replay_game.py _POS_COORDS — 13 rows × 31 cols.
const POS_COORDS: [(usize, usize); 24] = [
    (0, 0),  (0, 15), (0, 30),     // 0..2 outer top
    (6, 30), (12, 30), (12, 15),   // 3..5
    (12, 0), (6, 0),               // 6, 7
    (2, 5),  (2, 15), (2, 25),     // 8..10 middle top
    (6, 25), (10, 25), (10, 15),   // 11..13
    (10, 5), (6, 5),               // 14, 15
    (4, 10), (4, 15), (4, 20),     // 16..18 inner top
    (6, 20), (8, 20), (8, 15),     // 19..21
    (8, 10), (6, 10),              // 22, 23
];

/// Render board with X (yellow) for white, O (blue) for black, · for empty.
/// Mirrors the renderer in scripts/replay_game.py for visual consistency
/// with the existing Python tooling.
fn render(wbb: u32, bbb: u32) -> String {
    // Initialise a 13×31 grid of single-space strings.
    let mut grid: Vec<Vec<String>> = (0..13)
        .map(|_| (0..31).map(|_| " ".to_string()).collect())
        .collect();

    // Draw connecting line characters between adjacent positions.
    for src in 0..NUM_POSITIONS {
        for &dst_raw in &ADJACENCY[src] {
            if dst_raw == 0xFF { break; }
            let dst = dst_raw as usize;
            if src >= dst { continue; } // each undirected pair once
            let (r1, c1) = POS_COORDS[src];
            let (r2, c2) = POS_COORDS[dst];
            if r1 == r2 {
                let (lo, hi) = if c1 < c2 { (c1, c2) } else { (c2, c1) };
                for c in (lo + 1)..hi {
                    grid[r1][c] = "─".to_string();
                }
            } else if c1 == c2 {
                let (lo, hi) = if r1 < r2 { (r1, r2) } else { (r2, r1) };
                for r in (lo + 1)..hi {
                    grid[r][c1] = "│".to_string();
                }
            }
        }
    }

    // Place pieces.
    for p in 0..24 {
        let (r, c) = POS_COORDS[p];
        let glyph = if (wbb >> p) & 1 != 0 {
            format!("{}{}X{}", ANSI_BOLD, ANSI_YELLOW, ANSI_RESET)
        } else if (bbb >> p) & 1 != 0 {
            format!("{}{}O{}", ANSI_BOLD, ANSI_BLUE, ANSI_RESET)
        } else {
            "·".to_string()
        };
        grid[r][c] = glyph;
    }

    // Assemble with row labels (7,6,5,4,3,2,1 on alternating rows).
    let row_labels = ["7", "", "6", "", "5", "", "4", "", "3", "", "2", "", "1"];
    let mut out = String::from("    a    b    c    d    e    f    g\n");
    for (r, row) in grid.iter().enumerate() {
        let label = if !row_labels[r].is_empty() { row_labels[r] } else { " " };
        out += &format!("{}   ", label);
        for cell in row {
            out += cell;
        }
        out += "\n";
    }
    out
}

#[derive(Debug, Clone, Copy)]
struct State {
    wbb: u32,
    bbb: u32,
    stm: u8,
}

const STM_WHITE: u8 = 1;
const STM_BLACK: u8 = 2;

/// Enumerate legal (src, dst, capture) moves from this state.
/// For non-mill moves, capture = None. For mill moves, one entry per
/// legal capture target.
fn legal_moves(s: State) -> Vec<(u8, u8, Option<u8>)> {
    let (stm_bb, opp_bb) = if s.stm == STM_WHITE { (s.wbb, s.bbb) } else { (s.bbb, s.wbb) };
    let stm_count = popcount(stm_bb);
    let can_fly = stm_count == 3; // Flying variant: 3 pieces -> fly anywhere
    let occupied = s.wbb | s.bbb;
    let empties = !occupied & ((1u32 << NUM_POSITIONS) - 1);
    let mut out = Vec::new();

    let mut srcs = stm_bb;
    while srcs != 0 {
        let src = srcs.trailing_zeros() as u8;
        srcs &= srcs - 1;
        let after_lift = stm_bb & !(1u32 << src);
        let dests = if can_fly {
            empties
        } else {
            let mut m = 0u32;
            for &p in &ADJACENCY[src as usize] {
                if p == 0xFF { break; }
                if (empties >> p) & 1 != 0 { m |= 1u32 << p; }
            }
            m
        };
        let mut d = dests;
        while d != 0 {
            let dst = d.trailing_zeros() as u8;
            d &= d - 1;
            let new_stm = after_lift | (1u32 << dst);
            if is_mill_through(new_stm, dst) {
                let cap = legal_capture_targets(opp_bb);
                let mut c = cap;
                while c != 0 {
                    let cp = c.trailing_zeros() as u8;
                    c &= c - 1;
                    out.push((src, dst, Some(cp)));
                }
            } else {
                out.push((src, dst, None));
            }
        }
    }
    out
}

fn apply_move(s: State, mv: (u8, u8, Option<u8>)) -> State {
    let (src, dst, cap) = mv;
    let mut wbb = s.wbb;
    let mut bbb = s.bbb;
    if s.stm == STM_WHITE {
        wbb = (wbb & !(1u32 << src)) | (1u32 << dst);
        if let Some(c) = cap { bbb &= !(1u32 << c); }
    } else {
        bbb = (bbb & !(1u32 << src)) | (1u32 << dst);
        if let Some(c) = cap { wbb &= !(1u32 << c); }
    }
    State { wbb, bbb, stm: 3 - s.stm }
}

/// Return None if the resulting child is terminal (opp < 3 pieces).
/// Otherwise returns the subspace it lies in.
fn child_subspace(s: State) -> Option<Subspace> {
    let w = popcount(s.wbb) as u8;
    let b = popcount(s.bbb) as u8;
    if w < 3 || b < 3 { None } else { Some(Subspace::movement(w, b)) }
}

/// Map a (verdict, dtw) at a child to a "preference score" the parent
/// (which is about to move into this child) wants to maximise.
///
/// Lower score = better for the parent's STM. We pack ordering:
/// - LOSS for child STM (= win for us) ranks BEST: bucket 0, DTW asc (fastest win)
/// - DRAW for child STM ranks middle: bucket 1, DTW arbitrary
/// - WIN for child STM (= loss for us) ranks WORST: bucket 2, DTW desc (delay)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MoveScore {
    bucket: u8,
    dtw_signed: i32,
}

impl MoveScore {
    fn from_child(verdict: u8, dtw: u16) -> Self {
        match verdict {
            LOSS => MoveScore { bucket: 0, dtw_signed: dtw as i32 },         // win for us, smaller dtw better
            DRAW => MoveScore { bucket: 1, dtw_signed: 0 },
            WIN  => MoveScore { bucket: 2, dtw_signed: -(dtw as i32) },      // loss for us, bigger dtw better (so we negate)
            _ => MoveScore { bucket: 3, dtw_signed: 0 },
        }
    }
    fn terminal_win() -> Self {
        // Capturing to opp-below-3 is instant win at DTW=1
        MoveScore { bucket: 0, dtw_signed: 1 }
    }
}

impl Ord for MoveScore {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.bucket.cmp(&other.bucket).then(self.dtw_signed.cmp(&other.dtw_signed))
    }
}
impl PartialOrd for MoveScore {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
}

fn computer_pick_move(tb: &Tablebase, s: State) -> Option<((u8, u8, Option<u8>), MoveScore)> {
    let moves = legal_moves(s);
    if moves.is_empty() { return None; }
    let mut scored: Vec<((u8, u8, Option<u8>), MoveScore)> = Vec::new();
    for mv in moves {
        let child = apply_move(s, mv);
        let score = match child_subspace(child) {
            None => MoveScore::terminal_win(), // capture brought opp below 3 — we just won
            Some(target_sub) => {
                match tb.query(target_sub, child.wbb, child.bbb, child.stm) {
                    Some((v, d)) => MoveScore::from_child(v, d),
                    None => MoveScore { bucket: 3, dtw_signed: 0 }, // subspace not loaded → skip
                }
            }
        };
        scored.push((mv, score));
    }
    scored.sort_by(|a, b| a.1.cmp(&b.1));
    Some(scored.into_iter().next().unwrap())
}

fn parse_move_input(input: &str) -> Option<(u8, u8)> {
    let parts: Vec<&str> = input
        .split(|c: char| c == ' ' || c == '-' || c == '>')
        .filter(|s| !s.is_empty())
        .collect();
    if parts.len() != 2 { return None; }
    let src = parse_label(parts[0])?;
    let dst = parse_label(parts[1])?;
    Some((src, dst))
}

/// Generate a random valid (w, b) movement position. Uses a simple LCG
/// seeded from `seed` so runs are reproducible.
fn random_position(w: u8, b: u8, stm: u8, mut seed: u64) -> State {
    debug_assert!(w >= 3 && w <= 9 && b >= 3 && b <= 9);
    let mut next = || -> u64 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        seed
    };
    // Pick w distinct positions for whites.
    let mut whites: Vec<u8> = Vec::new();
    while (whites.len() as u8) < w {
        let p = (next() % 24) as u8;
        if !whites.contains(&p) { whites.push(p); }
    }
    let mut blacks: Vec<u8> = Vec::new();
    while (blacks.len() as u8) < b {
        let p = (next() % 24) as u8;
        if !whites.contains(&p) && !blacks.contains(&p) { blacks.push(p); }
    }
    let wbb: u32 = whites.iter().fold(0u32, |acc, &p| acc | (1u32 << p));
    let bbb: u32 = blacks.iter().fold(0u32, |acc, &p| acc | (1u32 << p));
    State { wbb, bbb, stm }
}

/// Parse a position spec like:
///   "3-3"                                                    → random (3,3)
///   "9-9"                                                    → random (9,9)
///   "a7,d7,g7/b6,d6,f6"                                      → explicit whites/blacks via algebraic labels
///   "a7 d7 g7 / b6 d6 f6"                                    → same, space-separated
fn parse_start_spec(spec: &str, stm: u8, seed: u64) -> Result<State, String> {
    // Form 1: "<w>-<b>" → random
    let bytes = spec.as_bytes();
    if let Some(dash) = spec.find('-') {
        let (lhs, rhs) = spec.split_at(dash);
        let rhs = &rhs[1..];
        if let (Ok(w), Ok(b)) = (lhs.parse::<u8>(), rhs.parse::<u8>()) {
            if w >= 3 && w <= 9 && b >= 3 && b <= 9 {
                return Ok(random_position(w, b, stm, seed));
            }
            return Err(format!("piece counts out of range [3..9]: {}-{}", w, b));
        }
    }
    // Form 2: "<white_labels>/<black_labels>"
    if let Some(slash) = spec.find('/') {
        let whites_part = &spec[..slash];
        let blacks_part = &spec[slash + 1..];
        let parse_list = |s: &str| -> Result<u32, String> {
            let mut bb = 0u32;
            for tok in s.split(|c: char| c == ',' || c.is_whitespace()).filter(|t| !t.is_empty()) {
                let Some(p) = parse_label(tok) else {
                    return Err(format!("unknown position label '{}'", tok));
                };
                if (bb >> p) & 1 != 0 {
                    return Err(format!("duplicate position '{}'", tok));
                }
                bb |= 1u32 << p;
            }
            Ok(bb)
        };
        let wbb = parse_list(whites_part)?;
        let bbb = parse_list(blacks_part)?;
        if wbb & bbb != 0 {
            return Err("white and black pieces share a square".to_string());
        }
        let wc = (wbb.count_ones()) as u8;
        let bc = (bbb.count_ones()) as u8;
        if !(3..=9).contains(&wc) || !(3..=9).contains(&bc) {
            return Err(format!("piece counts {}/{} not in [3..9]", wc, bc));
        }
        return Ok(State { wbb, bbb, stm });
    }
    let _ = bytes;
    Err(format!("unrecognised --start spec '{}'\n\
               Examples: '3-3', '9-9', 'a7,d7,g7,b6,d6,f6,c5,d5,e5/a1,d1,g1,b2,d2,f2,c3,d3,e3'", spec))
}

/// JSONL `--serve` mode for Python wrappers.
///
/// Reads one request per line on stdin and writes one response per line on
/// stdout. Designed to be spawned once by Python and kept alive — mmap'd
/// tablebases stay hot across queries (~1 ms per query end-to-end).
///
/// Request:  `{"wbb":<u32>,"bbb":<u32>,"stm":<1|2>}\n`
/// Response: `{"verdict":<u8>,"dtw":<u16>,"best_action":{"src":<u8>,"dst":<u8>,"cap":<u8>|null}|null,"top_moves":[{"src":<u8>,"dst":<u8>,"cap":<u8>|null,"verdict":<u8>,"dtw":<u16>}]}\n`
///
/// On parse error: `{"error":"<msg>"}\n`. On EOF: exit cleanly.
fn serve_loop(tb: &Tablebase, gevay: &GevayStore, indexer_cache_dir: Option<&Path>) {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut line = String::new();
    loop {
        line.clear();
        match stdin.lock().read_line(&mut line) {
            Ok(0) | Err(_) => return,
            Ok(_) => {}
        }
        let trimmed = line.trim();
        // Cheap discriminator: any request containing `"gevay":true` (with
        // or without internal whitespace) routes to the Gévay handler.
        // Full JSON parse stays in the per-handler functions.
        let resp = if trimmed.contains("\"gevay\"") && trimmed.contains("true") {
            match serve_handle_gevay(gevay, indexer_cache_dir, trimmed) {
                Ok(json) => json,
                Err(msg) => format!("{{\"error\":\"{}\"}}", msg.replace('"', "'")),
            }
        } else {
            match serve_handle(tb, trimmed) {
                Ok(json) => json,
                Err(msg) => format!("{{\"error\":\"{}\"}}", msg.replace('"', "'")),
            }
        };
        let _ = writeln!(stdout, "{}", resp);
        let _ = stdout.flush();
    }
}

/// JSONL Gévay query handler. Request: `{"gevay":true,"wbb":N,"bbb":N,"stm":<1|2>}`.
/// Response: `{"first_key":<i16>,"dtw":<i16>,"normalized":<f32>}` where
/// `normalized = first_key / WIN_ABS` (∈ [-1, +1] approximately). Errors
/// when the corresponding (w, b) gevay file wasn't loaded at startup.
fn serve_handle_gevay(
    gevay: &GevayStore,
    indexer_cache_dir: Option<&Path>,
    line: &str,
) -> Result<String, String> {
    if line.is_empty() { return Err("empty request".to_string()); }
    let state = parse_serve_request(line)?;
    let w = popcount(state.wbb) as u8;
    let b = popcount(state.bbb) as u8;
    if w < 3 || b < 3 || w > 9 || b > 9 {
        return Err(format!("piece counts ({},{}) out of [3,9]", w, b));
    }
    let sub = Subspace::movement(w, b);
    let (indexer_cell, mmap_gevay) = gevay.get(&sub)
        .ok_or_else(|| format!("gevay subspace ({},{}) not loaded", w, b))?;
    // Lazy build on first query for this subspace. OnceLock::get_or_init
    // is thread-safe and idempotent, so even if multiple worker threads
    // race on the first query they all see the same indexer.
    // Lazy build OR mmap-load from the indexer cache directory if set.
    // `open_or_build` falls back to in-memory build if the cache file is
    // missing or stale, so even a fresh setup works; subsequent processes
    // hit the cache file via mmap and share pages via the OS page cache.
    let indexer = indexer_cell.get_or_init(|| {
        CanonicalIndexer::open_or_build(sub, indexer_cache_dir)
    });
    let idx = indexer.index(state.wbb, state.bbb, state.stm) as usize;
    let fk = mmap_gevay.first_key()[idx];
    let d = mmap_gevay.dtw()[idx];
    let normalized = (fk as f32) / (WIN_ABS as f32);
    Ok(format!(
        "{{\"first_key\":{},\"dtw\":{},\"normalized\":{:.6}}}",
        fk, d, normalized
    ))
}

/// Parse a `{"wbb":N,"bbb":N,"stm":N}` line. Returns Err on any malformed
/// input — we don't need a full JSON parser here, just three named integers.
fn parse_serve_request(s: &str) -> Result<State, String> {
    let extract = |key: &str| -> Result<u64, String> {
        let needle = format!("\"{}\"", key);
        let i = s.find(&needle).ok_or_else(|| format!("missing key {}", key))?;
        let after = &s[i + needle.len()..];
        let colon = after.find(':').ok_or_else(|| format!("no ':' after {}", key))?;
        let rest = &after[colon + 1..];
        let bytes = rest.as_bytes();
        let mut start = 0;
        while start < bytes.len() && (bytes[start] == b' ' || bytes[start] == b'\t') { start += 1; }
        let mut end = start;
        while end < bytes.len() && (bytes[end].is_ascii_digit()) { end += 1; }
        if end == start { return Err(format!("no digits after {}", key)); }
        std::str::from_utf8(&bytes[start..end])
            .map_err(|_| "utf8 error".to_string())?
            .parse::<u64>()
            .map_err(|e| format!("parse {}: {}", key, e))
    };
    let wbb = extract("wbb")?;
    let bbb = extract("bbb")?;
    let stm = extract("stm")?;
    if wbb > u32::MAX as u64 || bbb > u32::MAX as u64 {
        return Err("wbb/bbb > u32::MAX".to_string());
    }
    if stm != 1 && stm != 2 {
        return Err(format!("stm must be 1 or 2, got {}", stm));
    }
    let wbb = wbb as u32;
    let bbb = bbb as u32;
    if wbb & bbb != 0 {
        return Err("wbb and bbb overlap".to_string());
    }
    Ok(State { wbb, bbb, stm: stm as u8 })
}

fn format_move(mv: (u8, u8, Option<u8>)) -> String {
    let (src, dst, cap) = mv;
    match cap {
        Some(c) => format!("{{\"src\":{},\"dst\":{},\"cap\":{}}}", src, dst, c),
        None => format!("{{\"src\":{},\"dst\":{},\"cap\":null}}", src, dst),
    }
}

fn serve_handle(tb: &Tablebase, line: &str) -> Result<String, String> {
    if line.is_empty() { return Err("empty request".to_string()); }
    let state = parse_serve_request(line)?;
    let w = popcount(state.wbb) as u8;
    let b = popcount(state.bbb) as u8;
    if w < 3 || b < 3 || w > 9 || b > 9 {
        return Err(format!("piece counts ({},{}) out of [3,9]", w, b));
    }
    let curr_sub = Subspace::movement(w, b);
    let (verdict, dtw) = match tb.query(curr_sub, state.wbb, state.bbb, state.stm) {
        Some(v) => v,
        None => return Err(format!("subspace ({},{}) not loaded", w, b)),
    };
    // Build top-3 ranked moves + best action via the same scoring as
    // computer_pick_move, then serialize.
    let moves = legal_moves(state);
    let mut scored: Vec<((u8, u8, Option<u8>), MoveScore, u8, u16)> = Vec::with_capacity(moves.len());
    for mv in moves {
        let child = apply_move(state, mv);
        let (cv, cd, score) = match child_subspace(child) {
            None => (LOSS, 1u16, MoveScore::terminal_win()),
            Some(target_sub) => match tb.query(target_sub, child.wbb, child.bbb, child.stm) {
                Some((v, d)) => (v, d, MoveScore::from_child(v, d)),
                None => (255u8, 0u16, MoveScore { bucket: 3, dtw_signed: 0 }),
            },
        };
        scored.push((mv, score, cv, cd));
    }
    scored.sort_by(|a, b| a.1.cmp(&b.1));
    let best_action = scored.first().map(|(mv, _, _, _)| format_move(*mv))
        .unwrap_or_else(|| "null".to_string());
    let top_n = scored.iter().take(3)
        .map(|(mv, _, cv, cd)| {
            let m = format_move(*mv);
            // Strip trailing '}' from move json to merge verdict+dtw in.
            let core = &m[..m.len() - 1];
            // verdict/dtw at the child state — caller can interpret WIN/LOSS
            // (these are child-STM verdicts, NOT this-STM).
            format!("{},\"verdict\":{},\"dtw\":{}}}", core, cv, cd)
        })
        .collect::<Vec<_>>()
        .join(",");
    Ok(format!(
        "{{\"verdict\":{},\"dtw\":{},\"best_action\":{},\"top_moves\":[{}]}}",
        verdict, dtw, best_action, top_n
    ))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: play_tb <PHASE1_DIR> [--side white|black] [--start <SPEC>] [--seed N] [--serve] [--gevay-dir <DIR>]");
        eprintln!();
        eprintln!("  --start SPEC : starting position. Two forms:");
        eprintln!("    'w-b'      : random (w whites, b blacks) movement position, e.g. '9-9'");
        eprintln!("    'W_list/B_list' : explicit pieces, e.g. 'a7,d7,g7/a1,d1,g1'");
        eprintln!("    default    : '3-3' with a fixed configuration");
        eprintln!("  --seed N     : RNG seed for random positions (default 42)");
        eprintln!("  --serve      : JSONL stdio mode for Python wrappers (see serve_loop docs)");
        eprintln!("  --gevay-dir DIR : load Phase 2 V_Gévay canonical tables for the");
        eprintln!("                    'gevay':true JSONL query path in --serve mode.");
        std::process::exit(1);
    }
    let phase1_dir = PathBuf::from(&args[1]);
    let mut human_side = STM_WHITE;
    let mut start_spec: Option<String> = None;
    let mut seed: u64 = 42;
    let mut serve = false;
    let mut gevay_dir: Option<PathBuf> = None;
    let mut indexer_cache_dir: Option<PathBuf> = None;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--side" if i + 1 < args.len() => {
                human_side = if args[i + 1].eq_ignore_ascii_case("black") { STM_BLACK } else { STM_WHITE };
                i += 2;
            }
            "--start" if i + 1 < args.len() => {
                start_spec = Some(args[i + 1].clone());
                i += 2;
            }
            "--seed" if i + 1 < args.len() => {
                seed = args[i + 1].parse().unwrap_or(42);
                i += 2;
            }
            "--serve" => {
                serve = true;
                i += 1;
            }
            "--gevay-dir" if i + 1 < args.len() => {
                gevay_dir = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--indexer-cache-dir" if i + 1 < args.len() => {
                indexer_cache_dir = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            _ => { i += 1; }
        }
    }
    // Default cache location: <gevay-dir>/.indexers/. Multiple play_tb
    // processes (e.g. self-play workers spawning their own subprocess)
    // mmap the same files there → resident RAM shared across processes
    // via the OS page cache. Without this, each subprocess rebuilds its
    // own CanonicalIndexer per subspace (~5 GB for (8,8)) and OOM-kills
    // the box once a few of them race on (7,7)/(8,8)/(8,9)/(9,9).
    if indexer_cache_dir.is_none() {
        if let Some(g) = &gevay_dir {
            indexer_cache_dir = Some(g.join(".indexers"));
        }
    }

    // Load Phase 1 tablebase: every movement .bin in the directory.
    // In --serve mode we log to stderr instead of stdout (stdout is the JSONL
    // response channel).
    if serve {
        eprintln!("Loading Phase 1 tablebase from {} ...", phase1_dir.display());
    } else {
        println!("Loading Phase 1 tablebase from {} ...", phase1_dir.display());
    }
    let mut tb = Tablebase::new();
    let mut loaded = 0;
    for w in 3..=9u8 {
        for b in 3..=9u8 {
            let sub = Subspace::movement(w, b);
            let path = phase1_dir.join(default_filename(sub, Variant::Flying));
            if path.exists() {
                let mt = MappedTable::open(&path).expect("mmap");
                tb.insert_mapped(mt);
                loaded += 1;
            }
        }
    }
    // If --gevay-dir was passed, load any V_Gévay canonical tables found
    // there. Missing files are silently skipped — the JSONL handler will
    // return a clean per-query error for those subspaces.
    let mut gevay_store: GevayStore = std::collections::HashMap::new();
    if let Some(dir) = &gevay_dir {
        if serve {
            eprintln!("Loading V_Gévay canonical tables from {} ...", dir.display());
        } else {
            println!("Loading V_Gévay canonical tables from {} ...", dir.display());
        }
        let mut g_loaded = 0;
        for w in 3..=9u8 {
            for b in 3..=9u8 {
                let sub = Subspace::movement(w, b);
                let path = dir.join(gevay_filename(sub, Variant::Flying));
                if !path.exists() { continue; }
                match load_gevay_canonical_mmap(&path) {
                    Ok(mmap_gevay) => {
                        // CanonicalIndexer is built lazily on first query
                        // for this subspace — avoids spending 5-50s × 49
                        // subspaces (~10 min) at startup when most queries
                        // only hit a handful of (w, b) tuples.
                        gevay_store.insert(
                            mmap_gevay.subspace,
                            (std::sync::OnceLock::new(), mmap_gevay),
                        );
                        g_loaded += 1;
                    }
                    Err(e) => {
                        eprintln!("  skip {} : {}", path.display(), e);
                    }
                }
            }
        }
        if serve {
            eprintln!("Loaded {} V_Gévay subspaces.", g_loaded);
        } else {
            println!("Loaded {} V_Gévay subspaces.", g_loaded);
        }
    }

    if serve {
        eprintln!("Loaded {} Phase 1 subspaces. Entering --serve loop.", loaded);
        serve_loop(&tb, &gevay_store, indexer_cache_dir.as_deref());
        return;
    }
    println!("Loaded {} subspaces.\n", loaded);

    // Starting position. Either parsed from --start, or hardcoded (3,3) example.
    let mut state = match start_spec.as_deref() {
        Some(spec) => match parse_start_spec(spec, STM_WHITE, seed) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("--start error: {}", e);
                std::process::exit(1);
            }
        },
        None => State {
            wbb: (1u32 << 0) | (1u32 << 7) | (1u32 << 15), // a7, a4, b4
            bbb: (1u32 << 4) | (1u32 << 3) | (1u32 << 11), // g1, g4, f4
            stm: STM_WHITE,
        },
    };

    println!("Starting position. You are {}.\n",
        if human_side == STM_WHITE { "WHITE (W)" } else { "BLACK (B)" });
    println!("{}", render(state.wbb, state.bbb));

    let stdin = io::stdin();
    let mut input_buf = String::new();
    loop {
        // Check game end: STM has < 3 pieces or no legal moves.
        let stm_bb = if state.stm == STM_WHITE { state.wbb } else { state.bbb };
        if popcount(stm_bb) < 3 {
            let loser = if state.stm == STM_WHITE { "WHITE" } else { "BLACK" };
            println!("Game over: {} has < 3 pieces. {} loses.", loser, loser);
            break;
        }
        let mvs = legal_moves(state);
        if mvs.is_empty() {
            let loser = if state.stm == STM_WHITE { "WHITE" } else { "BLACK" };
            println!("Game over: {} has no legal move (stalemate). {} loses.", loser, loser);
            break;
        }

        // Show whose turn + tablebase verdict on the current position.
        let stm_name = if state.stm == STM_WHITE { "WHITE" } else { "BLACK" };
        let curr_sub = Subspace::movement(popcount(state.wbb) as u8, popcount(state.bbb) as u8);
        let tb_query = tb.query(curr_sub, state.wbb, state.bbb, state.stm);
        match tb_query {
            Some((WIN, d)) => println!("[Tablebase] {} to move: WIN in {} plies", stm_name, d),
            Some((LOSS, d)) => println!("[Tablebase] {} to move: LOSS in {} plies", stm_name, d),
            Some((DRAW, _)) => println!("[Tablebase] {} to move: DRAW", stm_name),
            _ => println!("[Tablebase] subspace ({},{}) not loaded — no oracle for this position",
                popcount(state.wbb), popcount(state.bbb)),
        }

        if state.stm == human_side {
            println!("Legal moves: {} options",
                mvs.iter()
                    .map(|(s, d, c)| match c {
                        Some(cap) => format!("{}-{}x{}", label_of(*s), label_of(*d), label_of(*cap)),
                        None => format!("{}-{}", label_of(*s), label_of(*d)),
                    })
                    .collect::<Vec<_>>()
                    .join(", "));
            print!("Your move (e.g. 'a7-d7' or 'a7 d7', then capture if mill): ");
            io::stdout().flush().unwrap();
            input_buf.clear();
            if stdin.lock().read_line(&mut input_buf).is_err() { break; }
            let raw = input_buf.trim();
            if raw == "q" || raw == "quit" { break; }
            let Some((src, dst)) = parse_move_input(raw) else {
                println!("Could not parse '{}'. Use e.g. 'a7-d7'.", raw);
                continue;
            };
            // Find matching legal move(s)
            let candidates: Vec<_> = mvs.iter().filter(|(s, d, _)| *s == src && *d == dst).cloned().collect();
            if candidates.is_empty() {
                println!("'{} -> {}' is not legal here.", label_of(src), label_of(dst));
                continue;
            }
            let chosen = if candidates.len() == 1 {
                candidates[0]
            } else {
                // Mill formed: prompt for capture
                let opts: Vec<String> = candidates.iter()
                    .map(|(_, _, c)| label_of(c.unwrap()).to_string())
                    .collect();
                print!("Mill formed! Capture which opponent piece? Options: {} — ", opts.join(", "));
                io::stdout().flush().unwrap();
                input_buf.clear();
                if stdin.lock().read_line(&mut input_buf).is_err() { break; }
                let pick = input_buf.trim();
                let Some(cap_pos) = parse_label(pick) else {
                    println!("Bad input.");
                    continue;
                };
                match candidates.into_iter().find(|(_, _, c)| *c == Some(cap_pos)) {
                    Some(c) => c,
                    None => { println!("That capture isn't legal."); continue; }
                }
            };
            state = apply_move(state, chosen);
            println!();
            println!("{}", render(state.wbb, state.bbb));
        } else {
            // Computer's turn
            let pick = computer_pick_move(&tb, state);
            let Some((mv, score)) = pick else {
                println!("Computer has no legal moves.");
                break;
            };
            let (src, dst, cap) = mv;
            let label = match cap {
                Some(c) => format!("{}-{}x{} (capture)", label_of(src), label_of(dst), label_of(c)),
                None => format!("{}-{}", label_of(src), label_of(dst)),
            };
            let intent = match score.bucket {
                0 => "winning move",
                1 => "drawing move",
                2 => "losing move (longest defence)",
                _ => "unknown",
            };
            println!("Computer plays: {}  [{}]\n", label, intent);
            state = apply_move(state, mv);
            println!("{}", render(state.wbb, state.bbb));
        }
    }
}
