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

function ValueBar({ value }: { value: number }) {
  const pct = ((value + 1) / 2) * 100;
  return (
    <div style={{ margin: "8px 0" }}>
      <div
        style={{
          height: 12,
          background: "#333",
          borderRadius: 6,
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
            background: value >= 0 ? "#e8e8e8" : "#1a1a2e",
            transition: "all 0.3s",
          }}
        />
        <div
          style={{
            position: "absolute",
            left: "50%",
            top: 0,
            width: 2,
            height: "100%",
            background: "#f0a500",
          }}
        />
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#aaa", marginTop: 2 }}>
        <span>Agent losing</span>
        <span style={{ fontWeight: "bold", color: value >= 0 ? "#e8e8e8" : "#888" }}>
          {value >= 0 ? `+${value.toFixed(2)}` : value.toFixed(2)}
        </span>
        <span>Agent winning</span>
      </div>
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
    <div style={{ color: "#e0e0e0", fontFamily: "monospace", fontSize: 14 }}>
      <h3 style={{ margin: "0 0 4px", color: "#f0a500", fontSize: 16 }}>Analysis</h3>
      <div style={{ fontSize: 13, color: "#ccc", marginBottom: 12, fontWeight: "bold" }}>
        {agentName || (usingNetwork ? "AlphaZero network" : "Minimax fallback")}
      </div>

      <div style={{ marginBottom: 12 }}>
        <div style={{ fontSize: 12, color: "#aaa", marginBottom: 4 }}>
          Engine: {usingNetwork ? "AlphaZero network" : "Minimax fallback"}
        </div>
        <ValueBar value={valueEstimate} />
      </div>

      <div style={{ marginBottom: 16 }}>
        <div style={{ fontSize: 12, color: "#aaa", marginBottom: 6 }}>Top moves</div>
        {topMoves.map((m, i) => (
          <div
            key={m.action}
            style={{
              display: "flex",
              justifyContent: "space-between",
              padding: "4px 8px",
              marginBottom: 3,
              background: i === 0 ? "rgba(240,165,0,0.15)" : "rgba(255,255,255,0.05)",
              borderRadius: 4,
              borderLeft: i === 0 ? "3px solid #f0a500" : "3px solid transparent",
            }}
          >
            <span style={{ color: "#ddd" }}>{m.description}</span>
            <span style={{ color: "#aaa" }}>{(m.visit_prob * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>

      <div style={{ borderTop: "1px solid #333", paddingTop: 12 }}>
        <div style={{ fontSize: 12, color: "#aaa", marginBottom: 6 }}>Material</div>
        <div style={{ display: "flex", gap: 24 }}>
          <div>
            <div style={{ color: "#888" }}>You ({youGlyph})</div>
            <div>{playerPieces[humanIdx]} on board</div>
            <div style={{ color: "#aaa" }}>{piecesInHand[humanIdx]} in hand</div>
          </div>
          <div>
            <div style={{ color: "#888" }}>Agent ({agentGlyph})</div>
            <div>{playerPieces[agentIdx]} on board</div>
            <div style={{ color: "#aaa" }}>{piecesInHand[agentIdx]} in hand</div>
          </div>
        </div>
      </div>

      <div style={{ borderTop: "1px solid #333", paddingTop: 12, marginTop: 12 }}>
        <div style={{ fontSize: 12, color: "#aaa", marginBottom: 6 }}>
          Move history ({moveHistory.length})
        </div>
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
            <div style={{ color: "#555", fontSize: 12 }}>No moves yet.</div>
          )}
          {moveHistory.map((entry, i) => (
            <div
              key={i}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 6,
                padding: "2px 6px",
                borderRadius: 3,
                background: "rgba(255,255,255,0.03)",
              }}
            >
              <span style={{ color: "#555", fontSize: 11, minWidth: 24 }}>
                {i + 1}.
              </span>
              <span
                style={{
                  fontSize: 11,
                  color: entry.player === humanPlayer ? "#aad4ff" : "#ddd",
                }}
              >
                {entry.player === 1 ? "○" : "●"}
              </span>
              <span style={{ fontSize: 12, color: "#ccc" }}>{entry.desc}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
