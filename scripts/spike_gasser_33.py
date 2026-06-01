#!/usr/bin/env python3
"""Spike: retrograde wave on Morris subspace (3,3,0,0) with flying.

Validates the algorithm in docs/decisions/002-phase1-gasser-tables.md on
the smallest non-trivial movement subspace. Both sides have exactly 3
pieces, so the flying rule is active for both.

At (3,3,0,0) the subspace is fully self-contained:
- Non-mill flying moves stay inside (3,3,0,0).
- Mill-forming flying moves capture an opponent piece, taking opponent to
  2 pieces -> instant LOSS for opponent.
So the wave needs zero cross-subspace lookups — a perfect proof-of-algo.

Self-contained: this file copies the MILLS constant rather than importing
morris_rl, because the production env is no-flying and we want a clean
flying-mode spike with no entanglement.

Runtime: a few minutes in pure Python (2.7M raw positions per STM,
no symmetry reduction).
"""
from __future__ import annotations

import time
from collections import Counter, deque
from itertools import combinations

import tqdm

NUM_POS = 24
NUM_W = 3
NUM_B = 3

MILLS: list[tuple[int, int, int]] = [
    (0, 1, 2), (2, 3, 4), (4, 5, 6), (6, 7, 0),
    (8, 9, 10), (10, 11, 12), (12, 13, 14), (14, 15, 8),
    (16, 17, 18), (18, 19, 20), (20, 21, 22), (22, 23, 16),
    (1, 9, 17), (3, 11, 19), (5, 13, 21), (7, 15, 23),
]
MILLS_THROUGH: list[list[tuple[int, int, int]]] = [
    [m for m in MILLS if p in m] for p in range(NUM_POS)
]

UNKNOWN, WIN, LOSS, DRAW = 0, 1, 2, 3
STM_WHITE, STM_BLACK = 1, 2
LABEL = {UNKNOWN: "UNK", WIN: "WIN", LOSS: "LOSS", DRAW: "DRAW"}


def is_mill_through(player_bb: int, pos: int) -> bool:
    """True if player_bb contains a complete mill that includes pos."""
    for m in MILLS_THROUGH[pos]:
        if all((player_bb >> p) & 1 for p in m):
            return True
    return False


def has_mill_move(stm_bb: int, opp_bb: int) -> bool:
    """True if STM has at least one flying move that forms a mill.

    At (3,3) any mill formation produces a legal capture (opponent has
    exactly 3 pieces, can never all be locked out by the mill rule),
    so this directly implies STM can win in 1 ply.
    """
    occupied = stm_bb | opp_bb
    for src in range(NUM_POS):
        if not ((stm_bb >> src) & 1):
            continue
        stm_after_lift = stm_bb & ~(1 << src)
        for dst in range(NUM_POS):
            if (occupied >> dst) & 1:
                continue
            new_stm = stm_after_lift | (1 << dst)
            if is_mill_through(new_stm, dst):
                return True
    return False


def enumerate_33():
    """Yield all raw (wbb, bbb) in (3,3): C(24,3) * C(21,3) = 2_692_872."""
    for whites in combinations(range(NUM_POS), NUM_W):
        wbb = 0
        for p in whites:
            wbb |= 1 << p
        avail = [i for i in range(NUM_POS) if not ((wbb >> i) & 1)]
        for blacks in combinations(avail, NUM_B):
            bbb = 0
            for p in blacks:
                bbb |= 1 << p
            yield wbb, bbb


def init_phase(verdict, dtw, count, queue):
    """Phase 0: enumerate, mark instant-WINs, set counts for the rest."""
    total = 0
    n_expected = 2_692_872  # C(24,3) * C(21,3)
    full_count = 3 * (NUM_POS - 2 * NUM_W)  # 3 own pieces × 18 empties = 54
    for wbb, bbb in tqdm.tqdm(enumerate_33(), desc="init ", unit="pos", total=n_expected):
        for stm in (STM_WHITE, STM_BLACK):
            stm_bb, opp_bb = (wbb, bbb) if stm == STM_WHITE else (bbb, wbb)
            key = (wbb, bbb, stm)
            if has_mill_move(stm_bb, opp_bb):
                verdict[key] = WIN
                dtw[key] = 1
                queue.append(key)
            else:
                verdict[key] = UNKNOWN
                count[key] = full_count
            total += 1
    return total


def enumerate_parents(p_key):
    """Yield (3,3) parents of p reachable by a non-mill flying move."""
    wbb, bbb, stm_p = p_key
    mover_stm = 3 - stm_p
    mover_bb, fixed_bb = (wbb, bbb) if mover_stm == STM_WHITE else (bbb, wbb)
    occupied = wbb | bbb
    for dst in range(NUM_POS):
        if not ((mover_bb >> dst) & 1):
            continue
        # Skip if the move src->dst would have formed a mill: that parent
        # would have captured, sending us to (4,3) or (3,4), not (3,3).
        if is_mill_through(mover_bb, dst):
            continue
        for src in range(NUM_POS):
            if (occupied >> src) & 1:
                continue
            new_mover = (mover_bb & ~(1 << dst)) | (1 << src)
            if mover_stm == STM_WHITE:
                yield (new_mover, fixed_bb, mover_stm)
            else:
                yield (fixed_bb, new_mover, mover_stm)


def enumerate_child_dtws(q_key, dtw):
    """Yield DTWs of non-mill (3,3) children of q (for LOSS DTW = max+1)."""
    wbb, bbb, stm = q_key
    stm_bb, opp_bb = (wbb, bbb) if stm == STM_WHITE else (bbb, wbb)
    occupied = wbb | bbb
    for src in range(NUM_POS):
        if not ((stm_bb >> src) & 1):
            continue
        stm_after_lift = stm_bb & ~(1 << src)
        for dst in range(NUM_POS):
            if (occupied >> dst) & 1:
                continue
            new_stm = stm_after_lift | (1 << dst)
            if is_mill_through(new_stm, dst):
                continue  # mill child went to terminal (not used in LOSS computation)
            if stm == STM_WHITE:
                ck = (new_stm, opp_bb, STM_BLACK)
            else:
                ck = (opp_bb, new_stm, STM_WHITE)
            d = dtw.get(ck)
            if d is not None:
                yield d


def wave_propagate(verdict, dtw, count, queue, total_states):
    """Phase 1: propagate verdicts through parent links."""
    resolved = sum(1 for v in verdict.values() if v != UNKNOWN)
    pbar = tqdm.tqdm(total=total_states, desc="wave ", unit="state")
    pbar.update(resolved)
    last_report = resolved

    while queue:
        p_key = queue.popleft()
        p_v = verdict[p_key]
        p_dtw = dtw[p_key]

        for q_key in enumerate_parents(p_key):
            v_q = verdict.get(q_key, None)
            if v_q is None or v_q != UNKNOWN:
                continue
            if p_v == LOSS:
                verdict[q_key] = WIN
                dtw[q_key] = p_dtw + 1
                queue.append(q_key)
                resolved += 1
            else:  # p_v == WIN
                count[q_key] -= 1
                if count[q_key] == 0:
                    verdict[q_key] = LOSS
                    child_dtws = list(enumerate_child_dtws(q_key, dtw))
                    dtw[q_key] = (max(child_dtws) if child_dtws else 0) + 1
                    queue.append(q_key)
                    resolved += 1
        if resolved - last_report >= 100_000:
            pbar.update(resolved - last_report)
            last_report = resolved
    pbar.update(resolved - last_report)
    pbar.close()


def finalize_draws(verdict):
    draws = 0
    for k, v in verdict.items():
        if v == UNKNOWN:
            verdict[k] = DRAW
            draws += 1
    return draws


def stats_and_invariants(verdict, dtw):
    print("\n=== Verdict counts ===")
    by_stm = {STM_WHITE: Counter(), STM_BLACK: Counter()}
    dtw_by_stm_loss = {STM_WHITE: Counter(), STM_BLACK: Counter()}
    dtw_by_stm_win = {STM_WHITE: Counter(), STM_BLACK: Counter()}
    for (wbb, bbb, stm), v in verdict.items():
        by_stm[stm][v] += 1
        if v == WIN:
            dtw_by_stm_win[stm][dtw[(wbb, bbb, stm)]] += 1
        elif v == LOSS:
            dtw_by_stm_loss[stm][dtw[(wbb, bbb, stm)]] += 1

    for stm in (STM_WHITE, STM_BLACK):
        name = "WHITE" if stm == STM_WHITE else "BLACK"
        total = sum(by_stm[stm].values())
        print(f"\nSTM={name}: total={total:,}")
        for v in (WIN, LOSS, DRAW):
            c = by_stm[stm][v]
            print(f"  {LABEL[v]:>4}: {c:>10,}  ({c/total*100:5.2f}%)")

    print("\n=== DTW histogram (WIN, top 10 by DTW) ===")
    for stm in (STM_WHITE, STM_BLACK):
        name = "WHITE" if stm == STM_WHITE else "BLACK"
        print(f"\nSTM={name} (WIN):")
        for d, c in sorted(dtw_by_stm_win[stm].items())[:10]:
            print(f"  DTW={d:>3}: {c:>10,}")

    print("\n=== DTW histogram (LOSS, top 10 by DTW) ===")
    for stm in (STM_WHITE, STM_BLACK):
        name = "WHITE" if stm == STM_WHITE else "BLACK"
        print(f"\nSTM={name} (LOSS):")
        for d, c in sorted(dtw_by_stm_loss[stm].items())[:10]:
            print(f"  DTW={d:>3}: {c:>10,}")

    # Invariant A — full rule symmetry: swap colors AND swap STM
    # encodes the SAME game from the SAME player's POV, so verdicts are EQUAL.
    print("\n=== Invariant A: swap(colors) + swap(stm) -> same verdict ===")
    bad_a = 0
    for (wbb, bbb, stm), v in verdict.items():
        sib = verdict.get((bbb, wbb, 3 - stm))
        if sib != v:
            bad_a += 1
            if bad_a <= 3:
                print(f"  mismatch w={wbb:#x} b={bbb:#x} stm={stm}: "
                      f"v={LABEL[v]} sib={LABEL[sib] if sib is not None else 'MISSING'}")
    print(f"  {'PASS' if bad_a == 0 else f'FAIL: {bad_a:,}'} (of {len(verdict):,})")

    # D4 geometric invariance — left for the Rust pass once symmetry transforms
    # are implemented. (The spike does not enforce canonicalisation.)

    print("\n=== Sample DRAW positions ===")
    draws = [(w, b, s) for (w, b, s), v in verdict.items() if v == DRAW]
    print(f"  total DRAW states: {len(draws):,} ({len(draws)/len(verdict)*100:.3f}%)")
    for wbb, bbb, stm in draws[:5]:
        w_pos = [i for i in range(NUM_POS) if (wbb >> i) & 1]
        b_pos = [i for i in range(NUM_POS) if (bbb >> i) & 1]
        name = "WHITE" if stm == STM_WHITE else "BLACK"
        print(f"  W={w_pos} B={b_pos} stm={name}")


def main():
    print("=== Gasser spike: subspace (3,3,0,0) with flying ===\n")
    verdict, dtw, count = {}, {}, {}
    queue: deque = deque()

    t0 = time.perf_counter()
    total = init_phase(verdict, dtw, count, queue)
    t1 = time.perf_counter()
    instant_wins = len(queue)
    print(f"\nPhase 0: {total:,} states, {instant_wins:,} instant-WIN, "
          f"{total-instant_wins:,} UNKNOWN | {t1-t0:.1f}s")

    wave_propagate(verdict, dtw, count, queue, total)
    t2 = time.perf_counter()
    print(f"Phase 1 (wave) | {t2-t1:.1f}s")

    draws = finalize_draws(verdict)
    t3 = time.perf_counter()
    print(f"Phase 2: {draws:,} relabeled DRAW | {t3-t2:.2f}s")
    print(f"\nTotal: {t3-t0:.1f}s")

    stats_and_invariants(verdict, dtw)


if __name__ == "__main__":
    main()
