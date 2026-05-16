import { useEffect, useRef } from "react";
import type { MoveInfo } from "../types/game";
import type { MoveEntry } from "../hooks/useGame";
import EvalBar from "./EvalBar";

interface Props {
  topMoves: MoveInfo[];
  valueEstimate: number;   // Black's POV [-1, 1]
  usingNetwork: boolean;
  agentName: string;
  board: number[];         // 64 ints for piece counts
  moveHistory: MoveEntry[];
  humanPlayer: 1 | 2;
}

const ACCENT_VIOLET = "#a78bfa";
const ACCENT_CYAN = "#22d3ee";
const INK = "#e8ecf2";
const INK_DIM = "#8b94a3";
const INK_MUTE = "#4a5260";
const RULE = "#1c1f29";

const MONO =
  'ui-monospace, "JetBrains Mono", "SF Mono", Menlo, Consolas, monospace';

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
  board,
  moveHistory,
  humanPlayer,
}: Props) {
  const blackCount = board.filter((v) => v === 1).length;
  const whiteCount = board.filter((v) => v === 2).length;

  const historyRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (historyRef.current) {
      historyRef.current.scrollTop = historyRef.current.scrollHeight;
    }
  }, [moveHistory.length]);

  return (
    <div style={{ color: INK, fontSize: 13 }}>
      {/* Header */}
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
        {agentName || (usingNetwork ? "AlphaZero network" : "Random / Greedy")}
      </div>

      {/* Score */}
      <div style={{ marginBottom: 20 }}>
        <SectionTitle>Score</SectionTitle>
        <div style={{ display: "flex", gap: 20, fontFamily: MONO, fontSize: 12 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span
              style={{
                display: "inline-block",
                width: 14,
                height: 14,
                borderRadius: "50%",
                background: "#0a0c12",
                border: "1.5px solid #a4adb9",
                flexShrink: 0,
              }}
            />
            <span style={{ color: INK }}>{blackCount}</span>
            {humanPlayer === 1 && (
              <span style={{ color: INK_MUTE, fontSize: 10 }}>(you)</span>
            )}
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span
              style={{
                display: "inline-block",
                width: 14,
                height: 14,
                borderRadius: "50%",
                background: "#f5f6f8",
                border: "1px solid #2a2f3d",
                flexShrink: 0,
              }}
            />
            <span style={{ color: INK }}>{whiteCount}</span>
            {humanPlayer === 2 && (
              <span style={{ color: INK_MUTE, fontSize: 10 }}>(you)</span>
            )}
          </div>
        </div>
      </div>

      {/* Eval bar */}
      <div style={{ marginBottom: 20, display: "flex", justifyContent: "center" }}>
        <EvalBar valueEstimate={valueEstimate} />
      </div>

      {/* Top moves */}
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
              <span
                style={{
                  color: isTop ? ACCENT_VIOLET : INK_MUTE,
                  fontVariantNumeric: "tabular-nums",
                }}
              >
                {(m.visit_prob * 100).toFixed(1)}%
              </span>
            </div>
          );
        })}
      </div>

      {/* Move history */}
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
            <div style={{ color: INK_MUTE, fontSize: 12, fontFamily: MONO }}>
              No moves yet.
            </div>
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
                background:
                  i % 2 === 0 ? "rgba(255,255,255,0.015)" : "transparent",
                fontFamily: MONO,
              }}
            >
              <span
                style={{
                  color: INK_MUTE,
                  fontSize: 10,
                  minWidth: 24,
                  fontVariantNumeric: "tabular-nums",
                }}
              >
                {(i + 1).toString().padStart(2, "0")}
              </span>
              <span
                style={{
                  fontSize: 11,
                  color:
                    entry.player === humanPlayer ? ACCENT_CYAN : INK_DIM,
                }}
              >
                {entry.player === 1 ? "⚫" : "⚪"}
              </span>
              <span style={{ fontSize: 11, color: INK_DIM }}>{entry.desc}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
