"""Streamlit Reversi UI — play against the AlphaZero agent.

Launch:
    uv run streamlit run scripts/play_reversi_ui.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morris_rl.env.reversi.board import ACTION_SPACE_SIZE
from morris_rl.env.reversi.encoding import encode_state
from morris_rl.env.reversi.rules import (
    PASS_ACTION,
    PLAYER_1,
    PLAYER_2,
    GameState,
    Outcome,
    apply_action,
    get_legal_actions,
    initial_state,
    is_terminal,
    opponent,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Reversi — AlphaZero",
    page_icon="⚫",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
body { background: #050508; }
section.main { background: #050508; }
/* Score metrics */
div[data-testid="stMetric"] {
    background: #0d1117;
    border: 1px solid #1e2a3a;
    border-radius: 8px;
    padding: 10px 16px;
}
div[data-testid="stMetricValue"] { font-size: 2rem !important; }
/* Status bar */
div[data-testid="stAlert"] {
    background: #0d1117 !important;
    border: 1px solid #1e3a5f !important;
    color: #90caf9 !important;
}
/* Candidate move bars */
.cand-row { display:flex; align-items:center; gap:10px; margin-bottom:7px; }
.cand-bar-outer { flex:1; height:16px; background:#0d1117; border-radius:4px; overflow:hidden; border:1px solid #1e2a3a; }
.cand-bar-inner { height:100%; border-radius:4px; background: linear-gradient(90deg,#7c3aed,#e94560); }
.cand-label { font-size:13px; color:#aaa; min-width:32px; font-family:monospace; }
.cand-pct   { font-size:13px; color:#eee; min-width:42px; text-align:right; }
.cand-star  { color:#ffd700; font-size:14px; }
/* Eval bar */
.eval-wrap { display:flex; flex-direction:column; align-items:center; gap:6px; }
.eval-bar-outer { width:26px; border-radius:5px; overflow:hidden; border:2px solid #1e2a3a; }
.eval-black { width:100%; background: linear-gradient(180deg,#111,#333); }
.eval-white { width:100%; background: linear-gradient(180deg,#eee,#fff); }
.eval-lbl { font-size:11px; color:#aaa; text-align:center; font-weight:600; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COL_LABELS = "abcdefgh"


def _pos_label(pos: int) -> str:
    row, col = divmod(pos, 8)
    return f"{_COL_LABELS[col]}{row + 1}"


def _find_latest_checkpoint() -> Path | None:
    outputs = Path(__file__).parent.parent / "outputs"
    pts = sorted(outputs.glob("**/checkpoints/checkpoint_*.pt"))
    return pts[-1] if pts else None


def _load_agent(checkpoint_path: Path, num_sims: int, device: torch.device):
    from morris_rl.mcts.search import MorrisSearch
    from morris_rl.network.resnet import MorrisResNet

    payload = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    sd = payload["state_dict"]
    num_channels        = sd["input_conv.weight"].shape[0]
    num_planes          = sd["input_conv.weight"].shape[1]
    action_space_size   = sd["policy_head.fc2.weight"].shape[0]
    num_positions       = sd["value_head.fc1.weight"].shape[1]
    policy_head_hidden  = sd["policy_head.fc2.weight"].shape[1]
    value_head_hidden   = sd["value_head.fc2.weight"].shape[1]
    num_blocks = sum(1 for k in sd if k.startswith("trunk.") and k.endswith(".conv1.weight"))

    net = MorrisResNet(
        num_blocks=num_blocks, num_channels=num_channels, num_planes=num_planes,
        policy_head_hidden=policy_head_hidden, value_head_hidden=value_head_hidden,
        num_positions=num_positions, action_space_size=action_space_size,
    ).to(device)
    net.load_state_dict(sd)
    net.eval()

    reversi_fns = {
        "initial_state": initial_state, "get_legal_actions": get_legal_actions,
        "apply_action": apply_action, "is_terminal": is_terminal,
        "encode_state": encode_state, "action_space_size": action_space_size,
    }
    search = MorrisSearch(net, device, num_simulations=num_sims, game_fns=reversi_fns)
    return net, search


def _raw_value(net, state: GameState, device: torch.device) -> float:
    x = encode_state(state).to(device)
    legal = get_legal_actions(state)
    mask = torch.zeros(1, ACTION_SPACE_SIZE, dtype=torch.bool, device=device)
    for a in legal:
        mask[0, a] = True
    with torch.no_grad():
        _, value = net(x, mask)
    return float(value[0].item())


def _eval_to_black_pct(value: float, current_player: int) -> float:
    """Network value (current player POV, [-1,1]) → Black win probability [0,1]."""
    black_adv = value if current_player == PLAYER_1 else -value
    return (black_adv + 1.0) / 2.0


def _mcts_top_moves(search, state: GameState, top_n: int = 5):
    action, visit_probs = search.run(state, temperature=1e-6, add_noise=False)
    legal = get_legal_actions(state)
    candidates = sorted(
        [(a, float(visit_probs[a])) for a in legal if a != PASS_ACTION],
        key=lambda x: x[1], reverse=True,
    )[:top_n]
    return action, visit_probs, candidates


# ---------------------------------------------------------------------------
# Plotly board builder
# ---------------------------------------------------------------------------

_BOARD_BG    = "#050508"
_CELL_BG     = "#0d1117"
_CELL_LAST   = "#12150a"
_GRID_COLOR  = "#1e2a3a"
_LAST_BORDER = "#ffd700"
_LEGAL_COLOR = "#00ff88"
_HOVER_COLOR = "#1a2a3a"


def _build_board_figure(
    state: GameState,
    legal_set: set[int],
    is_human_turn: bool,
    show_legal: bool,
    last_move: int | None,
    visit_probs: np.ndarray | None = None,
) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=_BOARD_BG,
        plot_bgcolor=_CELL_BG,
        margin=dict(l=32, r=12, t=12, b=32),
        width=520, height=520,
        xaxis=dict(
            range=[-0.5, 7.5],
            tickvals=list(range(8)),
            ticktext=list(_COL_LABELS),
            showgrid=False, zeroline=False,
            tickfont=dict(color="#555", size=12),
            side="top",
        ),
        yaxis=dict(
            range=[7.5, -0.5],
            tickvals=list(range(8)),
            ticktext=[str(i + 1) for i in range(8)],
            showgrid=False, zeroline=False,
            tickfont=dict(color="#555", size=12),
            autorange=False,
        ),
        clickmode="event+select",
        dragmode=False,
        showlegend=False,
    )

    # --- Grid squares as shapes ---
    for pos in range(64):
        row, col = divmod(pos, 8)
        is_last = pos == last_move
        fig.add_shape(
            type="rect",
            x0=col - 0.48, y0=row - 0.48,
            x1=col + 0.48, y1=row + 0.48,
            fillcolor=_CELL_LAST if is_last else _CELL_BG,
            line=dict(
                color=_LAST_BORDER if is_last else _GRID_COLOR,
                width=2 if is_last else 1,
            ),
        )

    # --- Black pieces ---
    b_x, b_y = [], []
    for pos in range(64):
        if state.board[pos] == PLAYER_1:
            r, c = divmod(pos, 8)
            b_x.append(c); b_y.append(r)
    if b_x:
        fig.add_trace(go.Scatter(
            x=b_x, y=b_y, mode="markers",
            marker=dict(
                size=38, color="#111111",
                line=dict(color="#444", width=2),
                symbol="circle",
            ),
            hoverinfo="skip", showlegend=False,
        ))

    # --- White pieces ---
    w_x, w_y = [], []
    for pos in range(64):
        if state.board[pos] == PLAYER_2:
            r, c = divmod(pos, 8)
            w_x.append(c); w_y.append(r)
    if w_x:
        fig.add_trace(go.Scatter(
            x=w_x, y=w_y, mode="markers",
            marker=dict(
                size=38, color="#f0f0f0",
                line=dict(color="#aaa", width=2),
                symbol="circle",
            ),
            hoverinfo="skip", showlegend=False,
        ))

    # --- Visit probability heatmap on legal moves (if available) ---
    if visit_probs is not None and is_human_turn:
        legal_list = sorted(legal_set - {PASS_ACTION})
        if legal_list:
            max_v = max(visit_probs[a] for a in legal_list) or 1.0
            for pos in legal_list:
                r, c = divmod(pos, 8)
                alpha = 0.15 + 0.35 * (visit_probs[pos] / max_v)
                fig.add_shape(
                    type="rect",
                    x0=c - 0.48, y0=r - 0.48,
                    x1=c + 0.48, y1=r + 0.48,
                    fillcolor=f"rgba(124,58,237,{alpha:.2f})",
                    line=dict(color="rgba(0,0,0,0)", width=0),
                    layer="below",
                )

    # --- Legal move dots ---
    if show_legal and is_human_turn:
        l_x = [divmod(p, 8)[1] for p in legal_set if p != PASS_ACTION]
        l_y = [divmod(p, 8)[0] for p in legal_set if p != PASS_ACTION]
        if l_x:
            fig.add_trace(go.Scatter(
                x=l_x, y=l_y, mode="markers",
                marker=dict(
                    size=14, color=_LEGAL_COLOR,
                    symbol="circle",
                    line=dict(color="rgba(0,0,0,0)", width=0),
                ),
                hoverinfo="skip", showlegend=False,
            ))

    # --- Invisible clickable scatter for legal moves (human turn only) ---
    if is_human_turn:
        click_x, click_y, click_data, hover_text = [], [], [], []
        for pos in sorted(legal_set - {PASS_ACTION}):
            r, c = divmod(pos, 8)
            click_x.append(c)
            click_y.append(r)
            click_data.append(pos)
            hover_text.append(_pos_label(pos))
        if click_x:
            fig.add_trace(go.Scatter(
                x=click_x, y=click_y,
                mode="markers",
                marker=dict(
                    size=52, color="rgba(0,0,0,0.01)",
                    symbol="square",
                    line=dict(color="rgba(0,0,0,0)", width=0),
                ),
                customdata=click_data,
                hovertext=hover_text,
                hovertemplate="%{hovertext}<extra></extra>",
                hoverlabel=dict(bgcolor="#1e2a3a", font=dict(color="#fff", size=13)),
                showlegend=False,
                selectedpoints=[],
                unselected=dict(marker=dict(opacity=0)),
                selected=dict(marker=dict(color="rgba(233,69,96,0.3)", size=52)),
            ))

    return fig


# ---------------------------------------------------------------------------
# Eval bar HTML
# ---------------------------------------------------------------------------

def _eval_bar_html(black_pct: float, bar_height: int = 420) -> str:
    bh = int(black_pct * bar_height)
    wh = bar_height - bh
    return f"""
<div class="eval-wrap">
  <span class="eval-lbl">⚫<br>{black_pct*100:.0f}%</span>
  <div class="eval-bar-outer" style="height:{bar_height}px;">
    <div class="eval-black" style="height:{bh}px;"></div>
    <div class="eval-white" style="height:{wh}px;"></div>
  </div>
  <span class="eval-lbl">⚪<br>{(1-black_pct)*100:.0f}%</span>
</div>
"""


def _candidates_html(candidates: list[tuple[int, float]], best: int | None) -> str:
    rows = []
    for action, pct in candidates:
        label = _pos_label(action)
        bw = int(pct * 100)
        star = '<span class="cand-star">★</span>' if action == best else ""
        rows.append(f"""
<div class="cand-row">
  <span class="cand-label">{label}</span>{star}
  <div class="cand-bar-outer"><div class="cand-bar-inner" style="width:{bw}%;"></div></div>
  <span class="cand-pct">{pct*100:.1f}%</span>
</div>""")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

def _init() -> None:
    defaults: dict = {
        "game_state": initial_state(),
        "human_player": PLAYER_1,
        "last_move": None,
        "agent_candidates": [],
        "agent_best": None,
        "status": "",
        "net": None,
        "search": None,
        "ckpt_loaded": "",
        "eval_value": 0.0,
        "last_visit_probs": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _sidebar():
    st.sidebar.title("⚙️ Settings")

    latest = _find_latest_checkpoint()
    ckpt_str = st.sidebar.text_input(
        "Checkpoint", value=str(latest) if latest else "",
        help="Auto-detected latest checkpoint."
    )
    ckpt_path = Path(ckpt_str) if ckpt_str else None

    num_sims = st.sidebar.slider("MCTS simulations", 50, 1600, 400, step=50)
    show_legal = st.sidebar.toggle("Show legal moves", value=False)
    show_heatmap = st.sidebar.toggle("Show visit heatmap (human turn)", value=False,
                                     help="Requires running MCTS before your move — adds latency.")

    color = st.sidebar.radio("Play as", ["Black ⚫ (moves first)", "White ⚪"])
    human_color = PLAYER_1 if "Black" in color else PLAYER_2

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    st.sidebar.caption(f"Device: `{device_str}`")
    if latest:
        st.sidebar.caption(f"Latest ckpt: `{latest.name}`")

    if st.sidebar.button("🔄 New game", use_container_width=True):
        for k in ("game_state", "last_move", "agent_candidates", "agent_best",
                  "status", "eval_value", "last_visit_probs"):
            st.session_state[k] = {
                "game_state": initial_state(), "last_move": None,
                "agent_candidates": [], "agent_best": None,
                "status": "", "eval_value": 0.0, "last_visit_probs": None,
            }[k]
        st.session_state.human_player = human_color

    st.session_state.human_player = human_color
    return ckpt_path, num_sims, show_legal, show_heatmap, device_str, torch.device(device_str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _init()
    ckpt_path, num_sims, show_legal, show_heatmap, device_str, device = _sidebar()

    # Load agent on checkpoint / sims change
    ckpt_key = str(ckpt_path) + str(num_sims)
    if ckpt_path and ckpt_path.exists() and st.session_state.ckpt_loaded != ckpt_key:
        with st.spinner("Loading checkpoint…"):
            net, search = _load_agent(ckpt_path, num_sims, device)
        st.session_state.net = net
        st.session_state.search = search
        st.session_state.ckpt_loaded = ckpt_key
        st.toast(f"Loaded {ckpt_path.name}", icon="✅")
    elif not ckpt_path:
        st.session_state.net = None
        st.session_state.search = None

    state: GameState = st.session_state.game_state
    human_player: int = st.session_state.human_player
    net  = st.session_state.net
    search = st.session_state.search

    done, outcome = is_terminal(state)
    legal = get_legal_actions(state) if not done else []
    legal_set = set(legal)
    is_human_turn = (not done) and (state.current_player == human_player)

    # Eval bar value
    if net is not None and not done:
        st.session_state.eval_value = _raw_value(net, state, device)
    elif done:
        if outcome == Outcome.PLAYER_1_WINS:
            st.session_state.eval_value = 1.0 if human_player == PLAYER_1 else -1.0
        elif outcome == Outcome.PLAYER_2_WINS:
            st.session_state.eval_value = -1.0 if human_player == PLAYER_1 else 1.0
        else:
            st.session_state.eval_value = 0.0

    # Optional heatmap: run MCTS quietly on human turn
    visit_probs = None
    if show_heatmap and is_human_turn and search is not None:
        with st.spinner("Computing visit heatmap…"):
            _, visit_probs, _ = _mcts_top_moves(search, state, top_n=len(legal))
        st.session_state.last_visit_probs = visit_probs

    # Score header
    p1 = int(np.sum(state.board == PLAYER_1))
    p2 = int(np.sum(state.board == PLAYER_2))
    col_t1, col_t2, col_t3 = st.columns([2, 3, 2])
    with col_t1:
        st.metric("⚫ Black", p1)
    with col_t2:
        if done:
            msgs = {
                Outcome.PLAYER_1_WINS: "### 🏆 Black wins!",
                Outcome.PLAYER_2_WINS: "### 🏆 White wins!",
                Outcome.DRAW:          "### 🤝 Draw!",
            }
            st.markdown(msgs.get(outcome, "### Game over"))
        elif is_human_turn:
            st.markdown("### Your turn")
        else:
            st.markdown("### Agent thinking…")
    with col_t3:
        st.metric("⚪ White", p2)

    # Board + eval bar
    board_col, eval_col = st.columns([10, 1])
    with board_col:
        fig = _build_board_figure(
            state, legal_set, is_human_turn, show_legal,
            st.session_state.last_move, visit_probs,
        )
        event = st.plotly_chart(
            fig, use_container_width=False, key="board",
            on_select="rerun", config={"displayModeBar": False},
        )

    with eval_col:
        black_pct = _eval_to_black_pct(st.session_state.eval_value, state.current_player)
        st.markdown(_eval_bar_html(black_pct, bar_height=460), unsafe_allow_html=True)

    # Handle board click
    clicked_pos = None
    if event and hasattr(event, "selection") and event.selection:
        pts = event.selection.get("points", [])
        if pts:
            raw = pts[0].get("customdata") or pts[0].get("custom_data")
            if raw is not None:
                clicked_pos = int(raw)

    if clicked_pos is not None and clicked_pos in legal_set and is_human_turn:
        label = _pos_label(clicked_pos)
        st.session_state.game_state = apply_action(state, clicked_pos)
        st.session_state.last_move = clicked_pos
        st.session_state.agent_candidates = []
        st.session_state.agent_best = None
        st.session_state.last_visit_probs = None
        st.session_state.status = f"You played **{label}**."
        st.rerun()

    # Forced pass (human)
    if is_human_turn and not done and legal == [PASS_ACTION]:
        st.warning("No legal moves — you must pass.")
        if st.button("Pass →", use_container_width=True):
            st.session_state.game_state = apply_action(state, PASS_ACTION)
            st.session_state.last_move = None
            st.session_state.status = "You passed."
            st.rerun()

    # Status message
    if st.session_state.status:
        st.markdown(
            f'<div style="color:#90caf9;font-size:14px;margin:6px 0;">'
            f'ℹ️ {st.session_state.status}</div>',
            unsafe_allow_html=True,
        )

    # Agent turn
    if not is_human_turn and not done:
        time.sleep(0.05)
        if search is not None:
            with st.spinner("Agent thinking…"):
                best_action, visit_probs_agent, candidates = _mcts_top_moves(search, state, top_n=5)
        else:
            best_action = int(np.random.choice(legal))
            candidates = []

        label = _pos_label(best_action) if best_action != PASS_ACTION else "pass"
        mover = "Agent ⚫" if state.current_player == PLAYER_1 else "Agent ⚪"
        st.session_state.game_state = apply_action(state, best_action)
        st.session_state.last_move = best_action
        st.session_state.agent_candidates = candidates
        st.session_state.agent_best = best_action
        st.session_state.status = f"{mover} played **{label}**."
        st.rerun()

    # Agent analysis panel
    if st.session_state.agent_candidates:
        st.markdown("---")
        st.markdown(
            '<span style="color:#aaa;font-size:13px;letter-spacing:1px;">'
            'AGENT ANALYSIS — LAST MOVE</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            _candidates_html(st.session_state.agent_candidates, st.session_state.agent_best),
            unsafe_allow_html=True,
        )

    if done and st.button("🔄 Play again", use_container_width=True):
        for k in ("game_state", "last_move", "agent_candidates", "agent_best",
                  "status", "eval_value", "last_visit_probs"):
            st.session_state[k] = {
                "game_state": initial_state(), "last_move": None,
                "agent_candidates": [], "agent_best": None,
                "status": "", "eval_value": 0.0, "last_visit_probs": None,
            }[k]
        st.rerun()


if __name__ == "__main__":
    main()
