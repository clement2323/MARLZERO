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

use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use morris_tablebase::board::{ADJACENCY, MILLS, NUM_POSITIONS};
use morris_tablebase::rules::{is_mill_through, legal_capture_targets, popcount};
use morris_tablebase::storage::default_filename;
use morris_tablebase::subspace::{MappedTable, Subspace, Tablebase};
use morris_tablebase::wave::{DRAW, LOSS, Variant, WIN};

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

/// Render the board with W/B/. markers.
fn render(wbb: u32, bbb: u32) -> String {
    let cell = |p: u8| -> char {
        if (wbb >> p) & 1 != 0 { 'W' }
        else if (bbb >> p) & 1 != 0 { 'B' }
        else { '.' }
    };
    let mut s = String::new();
    s += "      a   b   c   d   e   f   g\n";
    s += &format!("  7   {} ----------- {} ----------- {}\n", cell(0), cell(1), cell(2));
    s += "      |                       |                       |\n";
    s += &format!("  6   |   {} --------- {} --------- {}   |\n", cell(8), cell(9), cell(10));
    s += "      |   |                   |                   |   |\n";
    s += &format!("  5   |   |   {} ----- {} ----- {}   |   |\n", cell(16), cell(17), cell(18));
    s += "      |   |   |               |               |   |   |\n";
    s += &format!("  4   {} - {} - {}             {} - {} - {}\n",
        cell(7), cell(15), cell(23), cell(19), cell(11), cell(3));
    s += "      |   |   |               |               |   |   |\n";
    s += &format!("  3   |   |   {} ----- {} ----- {}   |   |\n", cell(22), cell(21), cell(20));
    s += "      |   |                   |                   |   |\n";
    s += &format!("  2   |   {} --------- {} --------- {}   |\n", cell(14), cell(13), cell(12));
    s += "      |                       |                       |\n";
    s += &format!("  1   {} ----------- {} ----------- {}\n", cell(6), cell(5), cell(4));
    s
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: play_tb <PHASE1_DIR> [--side white|black]");
        std::process::exit(1);
    }
    let phase1_dir = PathBuf::from(&args[1]);
    let mut human_side = STM_WHITE;
    for win in args.windows(2) {
        if win[0] == "--side" {
            human_side = if win[1].eq_ignore_ascii_case("black") { STM_BLACK } else { STM_WHITE };
        }
    }

    // Load Phase 1 tablebase: every movement .bin in the directory.
    println!("Loading Phase 1 tablebase from {} ...", phase1_dir.display());
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
    println!("Loaded {} subspaces.\n", loaded);

    // Starting position: (3,3) with both sides at corners-ish (a configurable
    // example; flying makes any 3-3 position fully tactical).
    let mut state = State {
        wbb: (1u32 << 0) | (1u32 << 7) | (1u32 << 15), // a7, a4, b4
        bbb: (1u32 << 4) | (1u32 << 3) | (1u32 << 11), // g1, g4, f4
        stm: STM_WHITE,
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
