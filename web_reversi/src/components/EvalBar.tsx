interface Props {
  valueEstimate: number;  // Black's POV: [-1, 1] where +1 = full Black advantage
}

const MONO = 'ui-monospace, "JetBrains Mono", "SF Mono", Menlo, Consolas, monospace';

export default function EvalBar({ valueEstimate }: Props) {
  const blackPct = Math.max(0, Math.min(1, (valueEstimate + 1) / 2));
  const blackW = Math.round(blackPct * 100);
  const whiteW = 100 - blackW;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6, width: "100%" }}>
      {/* Bar — always full width, split black | white */}
      <div
        style={{
          width: "100%",
          height: 14,
          borderRadius: 7,
          overflow: "hidden",
          display: "flex",
          boxShadow: "0 0 0 1px rgba(255,255,255,0.06), 0 2px 8px rgba(0,0,0,0.4)",
        }}
      >
        <div
          style={{
            width: `${blackW}%`,
            background: "#0a0c12",
            transition: "width 0.5s cubic-bezier(0.4, 0, 0.2, 1)",
            flexShrink: 0,
            position: "relative",
          }}
        />
        <div
          style={{
            flex: 1,
            background: "#f5f6f8",
            transition: "flex 0.5s cubic-bezier(0.4, 0, 0.2, 1)",
          }}
        />
      </div>

      {/* Labels below */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontFamily: MONO,
          fontSize: 11,
          color: "#8b94a3",
        }}
      >
        <span>⚫ {blackW}%</span>
        <span>⚪ {whiteW}%</span>
      </div>
    </div>
  );
}
