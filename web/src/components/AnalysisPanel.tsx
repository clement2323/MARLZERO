import { useEffect, useRef } from "react";
import type { MoveInfo } from "../types/game";
import type { MoveEntry } from "../hooks/useGame";

interface Props {
  topMoves: MoveInfo[];
  valueEstimate: number;
  usingNetwork: boolean;
  agentName: string;
  playerPieces: [number, number];
  piecesInHand: [number, number];
  moveHistory: MoveEntry[];
  humanPlayer: 1 | 2;
}

const ACCENT_VIOLET = "#a78bfa";
const ACCENT_CYAN = "#22d3ee";
const ACCENT_MAGENTA = "#f472b6";
const INK = "#e8ecf2";
const INK_DIM = "#8b94a3";
const INK_MUTE = "#4a5260";
const RULE = "#1c1f29";

const MONO =
  'ui-monospace, "JetBrains Mono", "SF Mono", Menlo, Consolas, monospace';

function ValueBar({ value }: { value: number }) {
  const pct = ((value + 1) / 2) * 100;
  const agentWinning = value >= 0;
  const fillColor = agentWinning ? ACCENT_CYAN : ACCENT_MAGENTA;
  return (
    <div style={{ margin: "10px 0 4px" }}>
      <div
        style={{
          height: 6,
          background: RULE,
          borderRadius: 999,
          overflow: "hidden",
          position: "relative",
        }}
      >
        <div
          style={{
            position: "absolute",
            left: `${pct < 50 ? pct : 50}%`,
            width: `${Math.abs(pct - 50)}%`,
            height: "100%",
            background: fillColor,
            boxShadow: `0 0 10px ${fillColor}aa`,
            transition: "all 0.3s",
          }}
        />
        <div
          style={{
            position: "absolute",
            left: "50%",
            top: -2,
            width: 1,
            height: 10,
            background: ACCENT_VIOLET,
            opacity: 0.7,
          }}
        />
      </div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: 10,
          color: INK_MUTE,
          letterSpacing: "0.12em",
          textTransform: "uppercase",
          marginTop: 8,
          fontFamily: MONO,
        }}
      >
        <span>Agent losing</span>
        <span style={{ color: fillColor, fontWeight: 600 }}>
          {agentWinning ? `+${value.toFixed(2)}` : value.toFixed(2)}
        </span>
        <span>Agent winning</span>
      </div>
    </div>
  );
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <div
      style={{
        fontSize: 10,
        color: INK_MUTE,
        letterSpacing: "0.18em",
        textTransform: "uppercase",
        marginBottom: 8,
        fontFamily: MONO,
      }}
    >
      {children}
    </div>
  );
}

export default function AnalysisPanel({
  topMoves,
  valueEstimate,
  usingNetwork,
  agentName,
  playerPieces,
  piecesInHand,
  moveHistory,
  humanPlayer,
}: Props) {
  const humanIdx = humanPlayer - 1;
  const agentIdx = 1 - humanIdx;
  const youGlyph = humanPlayer === 1 ? "○" : "●";
  const agentGlyph = humanPlayer === 1 ? "●" : "○";
  const historyRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (historyRef.current) {
      historyRef.current.scrollTop = historyRef.current.scrollHeight;
    }
  }, [moveHistory.length]);
  return (
    <div style={{ color: INK, fontSize: 13 }}>
      <h3
        style={{
          margin: "0 0 4px",
          color: INK,
          fontSize: 12,
          letterSpacing: "0.2em",
          textTransform: "uppercase",
          fontWeight: 600,
        }}
      >
        <span
          style={{
            display: "inline-block",
            width: 6,
            height: 6,
            borderRadius: "50%",
            marginRight: 8,
            verticalAlign: "middle",
            background: ACCENT_VIOLET,
            boxShadow: `0 0 10px ${ACCENT_VIOLET}`,
          }}
        />
        Analysis
      </h3>
      <div
        style={{
          fontSize: 11,
          color: INK_DIM,
          marginBottom: 18,
          fontFamily: MONO,
          letterSpacing: "0.04em",
        }}
      >
        {agentName || (usingNetwork ? "AlphaZero network" : "Minimax fallback")}
      </div>

      <div style={{ marginBottom: 20 }}>
        <SectionTitle>Engine eval</SectionTitle>
        <ValueBar value={valueEstimate} />
      </div>

      <div style={{ marginBottom: 20 }}>
        <SectionTitle>Top moves</SectionTitle>
        {topMoves.length === 0 && (
          <div style={{ color: INK_MUTE, fontSize: 12, fontFamily: MONO }}>—</div>
        )}
        {topMoves.map((m, i) => {
          const isTop = i === 0;
          return (
            <div
              key={m.action}
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                padding: "6px 10px",
                marginBottom: 3,
                background: isTop
                  ? `linear-gradient(90deg, ${ACCENT_VIOLET}1a, transparent)`
                  : "transparent",
                borderRadius: 6,
                borderLeft: `2px solid ${isTop ? ACCENT_VIOLET : "transparent"}`,
                fontFamily: MONO,
                fontSize: 12,
              }}
            >
              <span style={{ color: isTop ? INK : INK_DIM }}>{m.description}</span>
              <span style={{ color: isTop ? ACCENT_VIOLET : INK_MUTE, fontVariantNumeric: "tabular-nums" }}>
                {(m.visit_prob * 100).toFixed(1)}%
              </span>
            </div>
          );
        })}
      </div>

      <div style={{ borderTop: `1px solid ${RULE}`, paddingTop: 14, marginBottom: 14 }}>
        <SectionTitle>Material</SectionTitle>
        <div style={{ display: "flex", gap: 24, fontFamily: MONO, fontSize: 12 }}>
          <div>
            <div style={{ color: INK_MUTE, fontSize: 10, letterSpacing: "0.1em", textTransform: "uppercase" }}>
              You {youGlyph}
            </div>
            <div style={{ color: INK }}>{playerPieces[humanIdx]} on board</div>
            <div style={{ color: INK_DIM }}>{piecesInHand[humanIdx]} in hand</div>
          </div>
          <div>
            <div style={{ color: INK_MUTE, fontSize: 10, letterSpacing: "0.1em", textTransform: "uppercase" }}>
              Agent {agentGlyph}
            </div>
            <div style={{ color: INK }}>{playerPieces[agentIdx]} on board</div>
            <div style={{ color: INK_DIM }}>{piecesInHand[agentIdx]} in hand</div>
          </div>
        </div>
      </div>

      <div style={{ borderTop: `1px solid ${RULE}`, paddingTop: 14 }}>
        <SectionTitle>Move history · {moveHistory.length}</SectionTitle>
        <div
          ref={historyRef}
          style={{
            maxHeight: 180,
            overflowY: "auto",
            display: "flex",
            flexDirection: "column",
            gap: 2,
          }}
        >
          {moveHistory.length === 0 && (
            <div style={{ color: INK_MUTE, fontSize: 12, fontFamily: MONO }}>No moves yet.</div>
          )}
          {moveHistory.map((entry, i) => (
            <div
              key={i}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                padding: "3px 8px",
                borderRadius: 4,
                background: i % 2 === 0 ? "rgba(255,255,255,0.015)" : "transparent",
                fontFamily: MONO,
              }}
            >
              <span style={{ color: INK_MUTE, fontSize: 10, minWidth: 24, fontVariantNumeric: "tabular-nums" }}>
                {(i + 1).toString().padStart(2, "0")}
              </span>
              <span
                style={{
                  fontSize: 11,
                  color: entry.player === humanPlayer ? ACCENT_CYAN : INK_DIM,
                }}
              >
                {entry.player === 1 ? "○" : "●"}
              </span>
              <span style={{ fontSize: 11, color: INK_DIM }}>{entry.desc}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
