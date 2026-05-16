"""Morris RL — Training Analysis Dashboard.

Launch with:
    uv run streamlit run scripts/dashboard.py
or:
    make dashboard
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import plotly.graph_objects as go
import streamlit as st
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ---------------------------------------------------------------------------
# Metric glossary
# ---------------------------------------------------------------------------

METRIC_GLOSSARY: dict[str, str] = {
    "train/total_loss":
        "Total loss = L_policy + L_value. Target: < 2.0 after convergence.",
    "train/policy_loss":
        "Cross-entropy between MCTS visit distribution π(a) and network output p_θ(a). "
        "Theoretical max ≈ 3.76 (uniform). Target: < 1.5. Stuck > 2.0 = MCTS nearly uniform.",
    "train/value_loss":
        "MSE between v_θ(s) and outcome z ∈ {-1,0,+1}. "
        "Naive baseline (always predict 0) ≈ 0.40 at 50% draw rate. "
        "Target: < 0.15. Above 0.30 = draw attractor active.",
    "game/draw_rate":
        "Cumulative fraction of all games ending in a draw. "
        "Polluted by false-positive resigns — prefer normal/draw_rate for diagnosis.",
    "game/term_resign_rate":
        "Fraction of games ended by resignation. "
        "Should be 0 when resign disabled. Above 50% with FPR > 20% = disable resign.",
    "game/term_pieces_below_3_rate":
        "Fraction ended by reducing opponent to < 3 pieces = genuine win. "
        "Mature network: this should dominate (> 50%).",
    "game/term_halfmove_cap_rate":
        "Fraction ended by the 300-halfmove no-progress cap = draw by timeout. "
        "High = draw attractor active.",
    "game/term_no_legal_moves_rate":
        "Fraction ended by full blockade = strategic win. "
        "Non-zero = network understands positional blocking.",
    "game/term_threefold_rate":
        "Fraction ended by threefold repetition. Near-zero is normal with MCTS.",
    "game/length_mean_window":
        "Mean game length (half-moves) over last 200 games, all populations. "
        "Above 400 = draw attractor. Below 80 with resign active = resign dominates.",
    "game/mills_per_game_mean":
        "Mills formed per game (both players). Rising = network learns mill tactics.",
    "game/captures_per_game_mean":
        "Captures per game. Low = passive / draw attractor. High = decisive play.",
    "game/p1_win_rate":
        "Player 1 win rate across all games. Should balance around 0.5.",
    "game/p2_win_rate":
        "Player 2 win rate across all games.",
    "resign/eligible_rate":
        "Fraction of games where the resign threshold was crossed ≥ 1 time. "
        "100% = threshold too high (not negative enough) for current value head calibration.",
    "resign/triggered_rate":
        "Fraction of games actually ended by resignation. "
        "= eligible_rate × (1 − verify_fraction). Gap = verify games played to completion.",
    "resign/verified_false_positive_rate":
        "Among verify-games, fraction where the would-be-resigner actually won or drew. "
        "AlphaZero target: < 5%. Above 20% = recalibrate. Above 50% = disable resign.",
    "resign/verified_correct_rate":
        "Fraction of resign decisions that were correct (player would have lost). Target: > 95%.",
    "resign/length_mean":
        "Mean length of resigned games. "
        "Close to min_move_for_resign = resign fires too early.",
    "resign/captures_per_game":
        "Captures in resigned games. Low = resignation before any tactical engagement.",
    "resign/verify_total":
        "Cumulative count of verify-games played to completion. Should grow steadily.",
    "curriculum/start_rate":
        "Fraction of games starting from a random mid-game position. "
        "Should hover around random_start_fraction (default 0.5).",
    "curriculum/draw_rate":
        "Draw rate — curriculum games NOT ended by resign only. "
        "Should be below normal/draw_rate if curriculum helps.",
    "curriculum/win_rate":
        "Decisive rate — curriculum games (excluding resign). Complement of curriculum/draw_rate.",
    "curriculum/length_mean":
        "Mean length — curriculum games excluding resign. "
        "Should be below normal/length_mean (mid-game positions resolve faster).",
    "curriculum/captures_per_game":
        "Captures — curriculum games. "
        "Should be above normal/captures_per_game (pieces already on board).",
    "normal/draw_rate":
        "Draw rate — games from initial_state, excluding resign. "
        "THE true draw-attractor indicator. Target after convergence: < 0.30.",
    "normal/win_rate":
        "Decisive rate — normal games excluding resign. Target: > 0.70.",
    "normal/length_mean":
        "Mean length — normal games excluding resign. Above 200 = network cannot close games.",
    "normal/captures_per_game":
        "Captures — normal games excluding resign.",
    "playout_cap/full_ratio":
        "Fraction of moves played with full simulations. "
        "Target ≈ full_sim_fraction (0.25). Only these moves feed the replay buffer.",
    "playout_cap/full_moves_per_game":
        "Full-sim moves per game. Only these create training samples.",
    "playout_cap/fast_moves_per_game":
        "Fast-sim moves per game (60 sims). Not stored in the buffer.",
    "system/rss_gb":
        "Trainer process RSS memory (GB). Should stabilise after buffer fills.",
    "system/results_qsize":
        "Worker → trainer result queue size. "
        "Near 0 = balanced. Above 5 = workers outpace trainer.",
    "system/weights_qsize_max":
        "Max weight broadcast queue size. Should stay at 0.",
    "train/buffer_size":
        "Number of samples in the replay buffer. Fills to max capacity then stays constant.",
    "train/games_collected":
        "Total self-play games collected since run start.",
    "train/learning_rate":
        "Current learning rate (Adam optimizer).",
    "train/grad_norm":
        "Pre-clip L2 gradient norm across all parameters. "
        "Spikes = unstable update. Near 0 = vanishing gradients. "
        "Clipped at max_grad_norm (default 1.0).",
    "train/value_mean":
        "Mean value head output over the training batch in [-1, +1]. "
        "Near 0 = draw attractor (network predicts ~draw for everything). "
        "Healthy: |mean| > 0.1 with std > 0.2.",
    "train/value_std":
        "Std of value head output over the training batch. "
        "Below 0.1 = predictions clustered near 0 = draw attractor. "
        "Target: > 0.3 after convergence.",
    "curriculum/halfmove_cap_rate":
        "Fraction of curriculum games ending by 300-halfmove timeout. "
        "Should fall below normal/halfmove_cap_rate if curriculum breaks the draw attractor.",
    "normal/halfmove_cap_rate":
        "Fraction of normal (initial-state) games ending by timeout. "
        "Main draw-attractor indicator alongside normal/draw_rate. "
        "Target: < 0.10 after convergence.",
    "game/timeout_discard_rate":
        "Cumulative fraction of games discarded from the replay buffer (halfmove_cap, discard_timeout_games=true). "
        "Shows how much draw-attractor fuel is removed per game.",
    "game/term_double_pass_rate":
        "[Reversi] Fraction of games ending by double-pass (neither player can flip before board fills). "
        "Target: < 15%. Above 50% = both sides playing poorly and passing.",
    "game/term_board_full_rate":
        "[Reversi] Fraction of games ending when all 64 squares are occupied — natural full-game completion. "
        "Complement of double_pass_rate for decisive Reversi games.",
    "game/final_pieces_diff_mean":
        "Mean signed piece difference at game end (P1 − P2). "
        "Positive = Black (P1) advantage. Negative = White (P2) advantage. Near 0 = balanced self-play.",
    "game/term_piece_count_tiebreak_rate":
        "[Morris] Fraction of games decided by piece-count tiebreak (100-halfmove total cap reached). "
        "Should fall as the network learns to close games early. "
        "Winner = player with more board pieces → more active mills → P1 fallback.",
}

_COLORS = {
    "resign":     "#e05252",
    "curriculum": "#f0a030",
    "normal":     "#52c07a",
    "value":      "#4f8ef7",
    "policy":     "#c97ef2",
    "total":      "#f2c94c",
    "full":       "#52c07a",
    "fast":       "#888888",
    "pieces":     "#52c07a",
    "no_legal":   "#4f8ef7",
    "halfmove":   "#f0a030",
    "threefold":  "#f2c94c",
    "system":     "#888888",
}

_DARK = "plotly_dark"


def _hex_rgba(hex_color: str, alpha: float = 0.55) -> str:
    """Convert #rrggbb to rgba(r,g,b,alpha) for Plotly fillcolor."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@st.cache_data(ttl=30)
def load_run(tb_path: str) -> dict[str, list[tuple[int, float]]]:
    ea = EventAccumulator(tb_path)
    ea.Reload()
    return {
        tag: [(e.step, e.value) for e in ea.Scalars(tag)]
        for tag in ea.Tags().get("scalars", [])
    }


def _steps(series: list[tuple[int, float]]) -> list[int]:
    return [s for s, _ in series]


def _vals(series: list[tuple[int, float]]) -> list[float]:
    return [v for _, v in series]


def _last(data: dict, tag: str, default: float = float("nan")) -> float:
    s = data.get(tag, [])
    return s[-1][1] if s else default


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------


def _line(
    fig: go.Figure,
    data: dict,
    tag: str,
    name: str,
    color: str,
    x_axis: str = "game",
    dash: str = "solid",
) -> None:
    series = data.get(tag, [])
    if not series:
        return
    short_def = METRIC_GLOSSARY.get(tag, tag).split(".")[0]  # first sentence only
    fig.add_trace(
        go.Scatter(
            x=_steps(series),
            y=_vals(series),
            name=name,
            line=dict(color=color, dash=dash),
            hovertemplate=(
                f"<b>{name}</b><br>"
                f"{x_axis}=%{{x}}<br>value=%{{y:.4f}}<br>"
                f"<span style='color:#aaa;font-size:0.85em'>{short_def}</span>"
                "<extra></extra>"
            ),
        )
    )


def _hline(fig: go.Figure, y: float, color: str, label: str) -> None:
    fig.add_hline(
        y=y,
        line=dict(color=color, dash="dot", width=1),
        annotation_text=label,
        annotation_position="top right",
        annotation_font_color=color,
    )


def _layout(
    fig: go.Figure,
    title: str,
    xlabel: str = "game",
    ylabel: str = "",
    height: int = 300,
) -> None:
    fig.update_layout(
        template=_DARK,
        title=dict(text=title, font=dict(size=13)),
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        height=height,
        margin=dict(l=50, r=20, t=36, b=36),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
    )


def _note(text: str) -> None:
    st.markdown(
        f"<div style='background:#1a1f2e;border-left:3px solid #4f8ef7;"
        f"padding:6px 10px;border-radius:4px;font-size:0.80em;"
        f"color:#a8b4c8;margin-bottom:6px'>{text}</div>",
        unsafe_allow_html=True,
    )


def _alert(text: str, level: str = "error") -> None:
    colors = {"error": "#e05252", "warning": "#f0a030", "ok": "#52c07a"}
    c = colors.get(level, "#4f8ef7")
    st.markdown(
        f"<div style='background:{c}22;border-left:3px solid {c};"
        f"padding:6px 10px;border-radius:4px;font-size:0.86em;"
        f"color:{c};margin-bottom:6px'>{text}</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    layout="wide",
    page_icon="♟",
    page_title="Morris RL Dashboard",
    initial_sidebar_state="expanded",
)

st.markdown(
    "<style>"
    "div[data-testid='stMetric']{background:#1a1f2e;border-radius:6px;padding:8px 12px}"
    "section[data-testid='stSidebar']{background:#10141c}"
    "h1,h2,h3{color:#e0e0e0}"
    ".stExpander{background:#1a1f2e}"
    "</style>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ♟ Morris RL")
    st.markdown("---")

    outputs_root = Path("outputs")
    run_dirs = sorted(
        [p for p in outputs_root.glob("*/*/") if (p / "tensorboard").exists()],
        reverse=True,
    )

    if not run_dirs:
        st.error("No runs found in outputs/")
        st.stop()

    run_labels = [str(p.relative_to(outputs_root)) for p in run_dirs]
    selected_label = st.selectbox("Primary run", run_labels, index=0)
    selected_path = outputs_root / selected_label

    compare_enabled = st.checkbox("Compare with second run", value=False)
    compare_label: str | None = None
    if compare_enabled:
        other_labels = [l for l in run_labels if l != selected_label]
        if other_labels:
            compare_label = st.selectbox("Reference run", other_labels)

    st.markdown("---")
    auto_refresh = st.checkbox("Auto-refresh (30s)", value=True)
    if auto_refresh:
        st.caption(f"Next refresh ~30s · {time.strftime('%H:%M:%S')}")

    st.markdown("---")
    with st.expander("📖 Full metric glossary"):
        for metric, definition in METRIC_GLOSSARY.items():
            st.markdown(f"**`{metric}`**")
            st.caption(definition)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

data = load_run(str(selected_path / "tensorboard"))
compare_data: dict | None = (
    load_run(str(outputs_root / compare_label / "tensorboard"))
    if compare_label else None
)

# ---------------------------------------------------------------------------
# Header KPIs
# ---------------------------------------------------------------------------

st.markdown(f"## Run `{selected_label}`")

games_n = int(_last(data, "train/games_collected", 0))
buf_size = int(_last(data, "train/buffer_size", 0))
value_loss = _last(data, "train/value_loss")
policy_loss = _last(data, "train/policy_loss")
resign_fpr = _last(data, "resign/verified_false_positive_rate", -1.0)
resign_rate = _last(data, "game/term_resign_rate", 0.0)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Games collected", f"{games_n:,}", help=METRIC_GLOSSARY["train/games_collected"])
c2.metric("Buffer size", f"{buf_size:,}", help=METRIC_GLOSSARY["train/buffer_size"])
c3.metric("Value loss", f"{value_loss:.3f}", help=METRIC_GLOSSARY["train/value_loss"])
c4.metric("Policy loss", f"{policy_loss:.3f}", help=METRIC_GLOSSARY["train/policy_loss"])
c5.metric(
    "Resign FPR",
    f"{resign_fpr:.1%}" if resign_fpr >= 0 else "N/A",
    help=METRIC_GLOSSARY["resign/verified_false_positive_rate"],
)

if resign_fpr > 0.20 and resign_rate > 0.50:
    _alert(
        f"🚨 RESIGN CATASTROPHIC — FPR={resign_fpr:.0%} > 20% and triggered={resign_rate:.0%} > 50%. "
        "Disable resign (self_play.resign.enabled=false) and restart.",
        "error",
    )
elif value_loss > 0.30:
    _alert(
        f"⚠️ Draw attractor likely — value_loss={value_loss:.3f} > 0.30. "
        "Network predicts ~0 (draw) for almost all states.",
        "warning",
    )
elif value_loss < 0.15:
    _alert(f"✅ value_loss={value_loss:.3f} — good calibration.", "ok")

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 2 — Training Losses
# ---------------------------------------------------------------------------

st.markdown("### 📉 Training Losses")
_note(
    "value_loss > 0.30 = draw attractor (network predicts ~0 for everything). "
    "policy_loss > 2.0 = MCTS visit distribution near-uniform. "
    "Both must fall together for real learning."
)

fig_loss = go.Figure()
_line(fig_loss, data, "train/total_loss", "Total loss", _COLORS["total"], x_axis="step")
_line(fig_loss, data, "train/policy_loss", "Policy loss", _COLORS["policy"], x_axis="step")
_line(fig_loss, data, "train/value_loss", "Value loss", _COLORS["value"], x_axis="step")
if compare_data:
    _line(fig_loss, compare_data, "train/value_loss", "Value (ref)", _COLORS["value"], x_axis="step", dash="dash")
    _line(fig_loss, compare_data, "train/policy_loss", "Policy (ref)", _COLORS["policy"], x_axis="step", dash="dash")

_hline(fig_loss, 0.15, "#52c07a", "value target 0.15")
_hline(fig_loss, 0.30, "#f0a030", "value warning 0.30")
_hline(fig_loss, 2.0, "#e05252", "policy warning 2.0")
_layout(fig_loss, "Losses vs gradient steps", xlabel="gradient step")
st.plotly_chart(fig_loss, use_container_width=True)

col_g1, col_g2 = st.columns(2)

with col_g1:
    fig_grad = go.Figure()
    _line(fig_grad, data, "train/grad_norm", "Grad norm (pre-clip)", _COLORS["policy"], x_axis="step")
    if compare_data:
        _line(fig_grad, compare_data, "train/grad_norm", "Grad norm (ref)", _COLORS["policy"], x_axis="step", dash="dash")
    _hline(fig_grad, 1.0, "#f0a030", "max_grad_norm=1.0")
    _layout(fig_grad, "Gradient norm (pre-clip)", xlabel="gradient step")
    st.plotly_chart(fig_grad, use_container_width=True)

with col_g2:
    fig_vdist = go.Figure()
    _line(fig_vdist, data, "train/value_mean", "Value mean", _COLORS["value"], x_axis="step")
    _line(fig_vdist, data, "train/value_std",  "Value std",  _COLORS["curriculum"], x_axis="step")
    if compare_data:
        _line(fig_vdist, compare_data, "train/value_std", "Value std (ref)", _COLORS["curriculum"], x_axis="step", dash="dash")
    _hline(fig_vdist, 0.0,  "#888888", "zero")
    _hline(fig_vdist, 0.3,  "#52c07a", "std target 0.3")
    _layout(fig_vdist, "Value head distribution (batch)", xlabel="gradient step", ylabel="[-1,+1]")
    _note(
        "value_mean ≈ 0 + value_std < 0.1 = draw attractor: network outputs near-zero for everything. "
        "Healthy: std > 0.3 and |mean| rising as the network distinguishes winning/losing positions."
    )
    st.plotly_chart(fig_vdist, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 3 — Population breakdown
# ---------------------------------------------------------------------------

st.markdown("### 🎲 Game Population Split (resign / curriculum / normal)")
_note(
    "Three mutually exclusive populations. "
    "resign = ended by resignation — may contain false positives. "
    "curriculum = random mid-game start + genuine engine termination. "
    "normal = initial state + genuine engine termination. "
    "resign > 50% with FPR > 20% = training signal corrupted."
)

col_a, col_b = st.columns(2)

with col_a:
    fig_pop = go.Figure()
    _line(fig_pop, data, "game/term_resign_rate", "Resign rate", _COLORS["resign"])
    _line(fig_pop, data, "curriculum/start_rate", "Curriculum start rate", _COLORS["curriculum"])
    _line(fig_pop, data, "game/draw_rate", "Draw rate (global)", "#888888", dash="dash")
    _layout(fig_pop, "Resign & curriculum rates (rolling window)")
    st.plotly_chart(fig_pop, use_container_width=True)

with col_b:
    resign_v = _last(data, "game/term_resign_rate", 0.0)
    curriculum_v = max(0.0, _last(data, "curriculum/start_rate", 0.0) * (1 - resign_v))
    normal_v = max(0.0, 1.0 - resign_v - curriculum_v)
    fig_pie = go.Figure(
        go.Pie(
            labels=["Resign", "Curriculum", "Normal"],
            values=[resign_v, curriculum_v, normal_v],
            marker_colors=[_COLORS["resign"], _COLORS["curriculum"], _COLORS["normal"]],
            hole=0.4,
            textinfo="label+percent",
            hovertemplate="%{label}: %{value:.1%}<extra></extra>",
        )
    )
    fig_pie.update_layout(
        template=_DARK, height=300, paper_bgcolor="#0e1117",
        showlegend=False, margin=dict(l=20, r=20, t=30, b=20),
        title="Current distribution (last 200 games)",
    )
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 4 — Resign diagnostics
# ---------------------------------------------------------------------------

st.markdown("### 🚩 Resign Quality")
_note(
    "False Positive Rate (FPR) = fraction of resign decisions that were wrong "
    "(the position was actually a draw or win for the resigning player). "
    "AlphaZero standard (Silver 2018): FPR < 5%. "
    "Above 20% → recalibrate threshold. Above 50% → disable resign."
)

if not math.isnan(value_loss):
    sigma = math.sqrt(max(0.0, value_loss))
    rec_threshold = round(-2.0 * sigma, 2)
    _note(
        f"💡 Calibration: value_loss={value_loss:.3f} → σ_value≈{sigma:.3f}. "
        f"Recommended threshold for FPR < 5%: ~{rec_threshold:.2f} (= −2σ)."
    )

col_r1, col_r2 = st.columns(2)

with col_r1:
    fig_resign = go.Figure()
    _line(fig_resign, data, "resign/eligible_rate", "Eligible rate", "#f0a030")
    _line(fig_resign, data, "resign/triggered_rate", "Triggered rate", _COLORS["resign"])
    _hline(fig_resign, 0.0, "#52c07a", "0% (resign off)")
    _layout(fig_resign, "Eligible vs Triggered rate (rolling 200)")
    st.plotly_chart(fig_resign, use_container_width=True)

with col_r2:
    fig_fpr = go.Figure()
    _line(fig_fpr, data, "resign/verified_false_positive_rate", "False positive rate", _COLORS["resign"])
    _line(fig_fpr, data, "resign/verified_correct_rate", "Correct rate", _COLORS["normal"])
    _hline(fig_fpr, 0.05, "#52c07a", "target < 5%")
    _hline(fig_fpr, 0.20, "#f0a030", "warning 20%")
    _hline(fig_fpr, 0.50, "#e05252", "danger 50%")
    _layout(fig_fpr, "False Positive Rate (verify games)")
    st.plotly_chart(fig_fpr, use_container_width=True)

fpr_current = _last(data, "resign/verified_false_positive_rate", -1.0)
verify_total = int(_last(data, "resign/verify_total", 0))
if fpr_current >= 0:
    fpr_color = "#52c07a" if fpr_current < 0.05 else ("#f0a030" if fpr_current < 0.20 else "#e05252")
    st.markdown(
        f"<div style='text-align:center;padding:10px;background:#1a1f2e;border-radius:8px'>"
        f"<span style='font-size:2.4em;color:{fpr_color};font-weight:bold'>{fpr_current:.1%}</span><br>"
        f"<span style='color:#888;font-size:0.85em'>Current False Positive Rate · {verify_total} verify-games</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

col_r3, col_r4 = st.columns(2)
with col_r3:
    fig_rlen = go.Figure()
    _line(fig_rlen, data, "resign/length_mean", "Resigned games", _COLORS["resign"])
    _line(fig_rlen, data, "game/length_mean_window", "All games", "#888888", dash="dash")
    _hline(fig_rlen, 30.0, "#f0a030", "min_move_for_resign=30")
    _layout(fig_rlen, "Game length: resigned vs all", ylabel="half-moves")
    st.plotly_chart(fig_rlen, use_container_width=True)

with col_r4:
    fig_rcap = go.Figure()
    _line(fig_rcap, data, "resign/captures_per_game", "Resign", _COLORS["resign"])
    _line(fig_rcap, data, "normal/captures_per_game", "Normal", _COLORS["normal"])
    _line(fig_rcap, data, "curriculum/captures_per_game", "Curriculum", _COLORS["curriculum"])
    _layout(fig_rcap, "Captures per game by population", ylabel="captures")
    st.plotly_chart(fig_rcap, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 5 — Curriculum vs Normal
# ---------------------------------------------------------------------------

st.markdown("### 🎓 Curriculum vs Normal (resign excluded)")
_note(
    "Only games terminated by the game engine (no resign). "
    "curriculum/draw_rate < normal/draw_rate = curriculum generates more decisive positions ✓. "
    "curriculum/length_mean < normal/length_mean = mid-game positions resolve faster ✓. "
    "If both populations look identical: curriculum not helping → adjust pieces_per_player."
)

col_c1, col_c2, col_c3 = st.columns(3)

with col_c1:
    fig_drate = go.Figure()
    _line(fig_drate, data, "curriculum/draw_rate", "Curriculum", _COLORS["curriculum"])
    _line(fig_drate, data, "normal/draw_rate", "Normal", _COLORS["normal"])
    if compare_data:
        _line(fig_drate, compare_data, "normal/draw_rate", "Normal (ref)", _COLORS["normal"], dash="dash")
    _hline(fig_drate, 0.30, "#f0a030", "target < 0.30")
    _layout(fig_drate, "Draw rate (excl. resign)", ylabel="fraction")
    st.plotly_chart(fig_drate, use_container_width=True)

with col_c2:
    fig_clen = go.Figure()
    _line(fig_clen, data, "curriculum/length_mean", "Curriculum", _COLORS["curriculum"])
    _line(fig_clen, data, "normal/length_mean", "Normal", _COLORS["normal"])
    _layout(fig_clen, "Mean game length (excl. resign)", ylabel="half-moves")
    st.plotly_chart(fig_clen, use_container_width=True)

with col_c3:
    fig_ccap = go.Figure()
    _line(fig_ccap, data, "curriculum/captures_per_game", "Curriculum", _COLORS["curriculum"])
    _line(fig_ccap, data, "normal/captures_per_game", "Normal", _COLORS["normal"])
    _layout(fig_ccap, "Captures per game (excl. resign)", ylabel="captures")
    st.plotly_chart(fig_ccap, use_container_width=True)

col_c4, col_c5, col_c6 = st.columns(3)
with col_c4:
    fig_wrate = go.Figure()
    _line(fig_wrate, data, "curriculum/win_rate", "Curriculum", _COLORS["curriculum"])
    _line(fig_wrate, data, "normal/win_rate", "Normal", _COLORS["normal"])
    _hline(fig_wrate, 0.70, "#52c07a", "target > 0.70")
    _layout(fig_wrate, "Decisive rate (excl. resign)", ylabel="fraction decisive")
    st.plotly_chart(fig_wrate, use_container_width=True)

with col_c6:
    fig_hmc = go.Figure()
    _line(fig_hmc, data, "curriculum/halfmove_cap_rate", "Curriculum timeout", _COLORS["curriculum"])
    _line(fig_hmc, data, "normal/halfmove_cap_rate",    "Normal timeout",     _COLORS["halfmove"])
    if compare_data:
        _line(fig_hmc, compare_data, "normal/halfmove_cap_rate", "Normal timeout (ref)", _COLORS["halfmove"], dash="dash")
    _hline(fig_hmc, 0.10, "#52c07a", "target < 0.10")
    _layout(fig_hmc, "Timeout rate by population", ylabel="fraction timed-out")
    _note(
        "curriculum/halfmove_cap_rate < normal/halfmove_cap_rate = curriculum generates more decisive endings. "
        "Both should fall toward 0 as the network learns to close games."
    )
    st.plotly_chart(fig_hmc, use_container_width=True)

with col_c5:
    c_draw = _last(data, "curriculum/draw_rate", float("nan"))
    n_draw = _last(data, "normal/draw_rate", float("nan"))
    c_len  = _last(data, "curriculum/length_mean", float("nan"))
    n_len  = _last(data, "normal/length_mean", float("nan"))
    c_cap  = _last(data, "curriculum/captures_per_game", float("nan"))
    n_cap  = _last(data, "normal/captures_per_game", float("nan"))

    def _diff_cell(c: float, n: float, lower_is_better: bool = True) -> str:
        if math.isnan(c) or math.isnan(n):
            return "N/A"
        diff = c - n
        ok = diff < 0 if lower_is_better else diff > 0
        color = "#52c07a" if ok else "#e05252"
        arrow = "↓" if diff < 0 else "↑"
        return f"<span style='color:{color}'>{arrow} {abs(diff):.3f}</span>"

    def _fmt(v: float, decimals: int = 3) -> str:
        return f"{v:.{decimals}f}" if not math.isnan(v) else "—"

    st.markdown(
        f"<div style='background:#1a1f2e;padding:14px;border-radius:8px;margin-top:8px'>"
        f"<table style='width:100%;color:#e0e0e0;font-size:0.88em;border-collapse:collapse'>"
        f"<tr style='border-bottom:1px solid #333'>"
        f"<th style='text-align:left;padding:4px'>Metric</th>"
        f"<th>Curriculum</th><th>Normal</th>"
        f"<th>Δ (cur−nor)</th></tr>"
        f"<tr><td>draw_rate</td><td style='text-align:center'>{_fmt(c_draw)}</td>"
        f"<td style='text-align:center'>{_fmt(n_draw)}</td>"
        f"<td style='text-align:center'>{_diff_cell(c_draw, n_draw, True)}</td></tr>"
        f"<tr><td>length_mean</td><td style='text-align:center'>{_fmt(c_len,1)}</td>"
        f"<td style='text-align:center'>{_fmt(n_len,1)}</td>"
        f"<td style='text-align:center'>{_diff_cell(c_len, n_len, True)}</td></tr>"
        f"<tr><td>captures/game</td><td style='text-align:center'>{_fmt(c_cap,2)}</td>"
        f"<td style='text-align:center'>{_fmt(n_cap,2)}</td>"
        f"<td style='text-align:center'>{_diff_cell(c_cap, n_cap, False)}</td></tr>"
        f"</table>"
        f"<p style='font-size:0.72em;color:#555;margin-top:6px'>"
        f"Green ↓ = curriculum better (lower) · Green ↑ = curriculum better (higher)</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 6 — Termination breakdown
# ---------------------------------------------------------------------------

st.markdown("### 🏁 Termination Breakdown")
_note(
    "Maturity sequence: "
    "① Early → halfmove_cap dominant (draw attractor, games time out). "
    "② Mid → resign dominant if feature active. "
    "③ Mature → pieces_below_3 > 50% + no_legal_moves non-zero (network understands endgames). "
    "threefold near-zero is normal."
)

fig_term = go.Figure()
tags_term = [
    ("game/term_resign_rate",         "Resign",               _COLORS["resign"]),
    ("game/term_pieces_below_3_rate", "Pieces < 3",           _COLORS["pieces"]),
    ("game/term_no_legal_moves_rate", "Blockade",             _COLORS["no_legal"]),
    ("game/term_halfmove_cap_rate",   "Timeout (300hm)",      _COLORS["halfmove"]),
    ("game/term_threefold_rate",      "Threefold",            _COLORS["threefold"]),
    ("game/term_double_pass_rate",         "Double-pass (Reversi)", "#4fc3f7"),
    ("game/term_board_full_rate",          "Board full (Reversi)",  "#80cbc4"),
    ("game/term_piece_count_tiebreak_rate","Piece-count tiebreak",  "#9c27b0"),
]
for tag, name, color in tags_term:
    series = data.get(tag, [])
    if not series:
        continue
    short = METRIC_GLOSSARY.get(tag, tag).split(".")[0]
    fig_term.add_trace(
        go.Scatter(
            x=_steps(series),
            y=_vals(series),
            name=name,
            stackgroup="one",
            line=dict(color=color),
            fillcolor=_hex_rgba(color, 0.6),
            hovertemplate=(
                f"<b>{name}</b><br>game=%{{x}}<br>%{{y:.2%}}<br>"
                f"<span style='color:#aaa;font-size:0.85em'>{short}</span>"
                "<extra></extra>"
            ),
        )
    )

_layout(fig_term, "Termination reason breakdown (stacked)", height=340)
fig_term.update_layout(yaxis=dict(range=[0, 1]))
st.plotly_chart(fig_term, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 6b — Reversi / Othello breakdown (only rendered for Reversi runs)
# ---------------------------------------------------------------------------

_has_reversi = bool(data.get("game/term_double_pass_rate") or data.get("game/term_board_full_rate"))

if _has_reversi:
    st.markdown("### ♟ Reversi — Game Endings")
    _note(
        "Reversi games end either when the board is completely full (all 64 squares occupied) "
        "or via a double-pass (neither player has a legal flip available before the board fills). "
        "Double-pass rate ≈ 5–15% is typical. Above 50% signals poor play (both sides passing too early)."
    )

    col_rv1, col_rv2, col_rv3 = st.columns(3)

    with col_rv1:
        fig_dp = go.Figure()
        _line(fig_dp, data, "game/term_double_pass_rate", "Double-pass (early end)", "#4fc3f7")
        _line(fig_dp, data, "game/term_board_full_rate",  "Board full (natural end)", "#80cbc4")
        _hline(fig_dp, 0.15, "#f0a030", "double-pass warn > 15%")
        _layout(fig_dp, "Game ending reason", ylabel="fraction")
        st.plotly_chart(fig_dp, use_container_width=True)

    with col_rv2:
        fig_pdiff = go.Figure()
        _line(fig_pdiff, data, "game/final_pieces_diff_mean", "P1 − P2 pieces", "#f2c94c")
        _hline(fig_pdiff, 0.0, "#888888", "balanced")
        _layout(fig_pdiff, "Mean final piece difference (P1 − P2)", ylabel="pieces")
        _note("Positive = Black wins by more pieces. Negative = White advantage. Near 0 = balanced.")
        st.plotly_chart(fig_pdiff, use_container_width=True)

    with col_rv3:
        # KPI bar: current values
        dp_now = _last(data, "game/term_double_pass_rate", float("nan"))
        bf_now = _last(data, "game/term_board_full_rate",  float("nan"))
        pd_now = _last(data, "game/final_pieces_diff_mean", float("nan"))
        len_now = _last(data, "game/length_mean_window", float("nan"))

        dp_color = "#52c07a" if dp_now < 0.15 else "#f0a030"
        pd_color = "#4f8ef7" if abs(pd_now) < 3 else "#f0a030"

        st.markdown(
            f"<div style='background:#1a1f2e;padding:14px;border-radius:8px;margin-top:8px'>"
            f"<p style='color:#888;font-size:0.8em;margin-bottom:8px'>CURRENT VALUES</p>"
            f"<p style='color:{dp_color};font-size:1.3em;font-weight:bold'>{dp_now:.1%}</p>"
            f"<p style='color:#aaa;font-size:0.8em;margin-top:-8px'>double-pass rate (target &lt; 15%)</p>"
            f"<p style='color:#80cbc4;font-size:1.3em;font-weight:bold'>{bf_now:.1%}</p>"
            f"<p style='color:#aaa;font-size:0.8em;margin-top:-8px'>board-full rate</p>"
            f"<p style='color:{pd_color};font-size:1.3em;font-weight:bold'>{pd_now:+.2f}</p>"
            f"<p style='color:#aaa;font-size:0.8em;margin-top:-8px'>mean final piece diff (P1 − P2)</p>"
            f"<p style='color:#e0e0e0;font-size:1.3em;font-weight:bold'>{len_now:.1f}</p>"
            f"<p style='color:#aaa;font-size:0.8em;margin-top:-8px'>mean game length (moves)</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

# ---------------------------------------------------------------------------
# Section 7 — Playout cap
# ---------------------------------------------------------------------------

st.markdown("### ⚡ Playout Cap Randomization")
_note(
    "KataGo-style: 25% of moves use full sims (250) → stored in buffer. "
    "75% use fast sims (60) → not stored, just advance the game. "
    "full_ratio ≈ 0.25 confirms feature is active. "
    "Effective sims/move ≈ 0.25×250 + 0.75×60 = 107.5 → 2.3× more games/hour."
)

col_p1, col_p2 = st.columns(2)

with col_p1:
    fig_ratio = go.Figure()
    _line(fig_ratio, data, "playout_cap/full_ratio", "Full ratio", _COLORS["full"], x_axis="step")
    _hline(fig_ratio, 0.25, "#52c07a", "target 0.25")
    _layout(fig_ratio, "Full sim ratio (target ≈ 0.25)", xlabel="step")
    st.plotly_chart(fig_ratio, use_container_width=True)

with col_p2:
    fig_moves = go.Figure()
    _line(fig_moves, data, "playout_cap/full_moves_per_game", "Full moves/game", _COLORS["full"], x_axis="step")
    _line(fig_moves, data, "playout_cap/fast_moves_per_game", "Fast moves/game", _COLORS["fast"], x_axis="step")
    _layout(fig_moves, "Full vs fast moves per game", xlabel="step", ylabel="moves")
    st.plotly_chart(fig_moves, use_container_width=True)

full_ratio_v = _last(data, "playout_cap/full_ratio", float("nan"))
full_m = _last(data, "playout_cap/full_moves_per_game", 0.0)
fast_m = _last(data, "playout_cap/fast_moves_per_game", 0.0)
total_m = full_m + fast_m
if total_m > 0:
    eff = (full_m * 250 + fast_m * 60) / total_m
    cp1, cp2, cp3 = st.columns(3)
    cp1.metric("Full ratio", f"{full_ratio_v:.3f}", help=METRIC_GLOSSARY["playout_cap/full_ratio"])
    cp2.metric("Full moves/game", f"{full_m:.1f}", help=METRIC_GLOSSARY["playout_cap/full_moves_per_game"])
    cp3.metric("Effective sims/move", f"{eff:.1f} (vs 250 full)", help="= full×250 + fast×60")

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 8 — System health
# ---------------------------------------------------------------------------

st.markdown("### 🖥️ System Health")
_note(
    "RSS should stabilise after the buffer fills. "
    "results_qsize near 0 = pipeline balanced. "
    "Above 5 = workers outpace trainer → increase updates_per_game or reduce num_workers."
)

col_s1, col_s2 = st.columns(2)

with col_s1:
    fig_rss = go.Figure()
    _line(fig_rss, data, "system/rss_gb", "RSS (GB)", _COLORS["system"], x_axis="step")
    _hline(fig_rss, 8.0, "#e05252", "8 GB alert")
    _layout(fig_rss, "Trainer RSS memory", xlabel="step", ylabel="GB")
    st.plotly_chart(fig_rss, use_container_width=True)

with col_s2:
    fig_q = go.Figure()
    _line(fig_q, data, "system/results_qsize", "Results queue", _COLORS["system"], x_axis="step")
    _hline(fig_q, 5.0, "#f0a030", "warning 5")
    _layout(fig_q, "Worker → trainer queue size", xlabel="step", ylabel="items")
    st.plotly_chart(fig_q, use_container_width=True)

rss_v = _last(data, "system/rss_gb", float("nan"))
q_v   = _last(data, "system/results_qsize", float("nan"))
cs1, cs2 = st.columns(2)
cs1.metric("Current RSS", f"{rss_v:.2f} GB", help=METRIC_GLOSSARY["system/rss_gb"])
cs2.metric("Queue size",  f"{q_v:.0f}",      help=METRIC_GLOSSARY["system/results_qsize"])

col_b1, col_b2 = st.columns(2)
with col_b1:
    fig_buf = go.Figure()
    _line(fig_buf, data, "train/buffer_size", "Buffer samples", _COLORS["value"], x_axis="step")
    _layout(fig_buf, "Replay buffer occupancy", xlabel="gradient step", ylabel="samples")
    st.plotly_chart(fig_buf, use_container_width=True)

with col_b2:
    discard_rate = _last(data, "game/timeout_discard_rate", -1.0)
    if discard_rate >= 0:
        fig_dis = go.Figure()
        _line(fig_dis, data, "game/timeout_discard_rate", "Discard rate", _COLORS["halfmove"])
        _layout(fig_dis, "Timeout game discard rate (cumulative)", ylabel="fraction")
        _note(
            "Fraction of all games discarded from buffer (halfmove_cap + discard_timeout_games=true). "
            "Non-zero only when the flag is enabled."
        )
        st.plotly_chart(fig_dis, use_container_width=True)
    else:
        st.info("game/timeout_discard_rate not logged — discard_timeout_games=false or run pre-dates this metric.")

# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------

if auto_refresh:
    time.sleep(30)
    st.cache_data.clear()
    st.rerun()
