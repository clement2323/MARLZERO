interface Props {
  valueEstimate: number;  // Black's POV: [-1, 1] where +1 = full Black advantage
}

const MONO = 'ui-monospace, "JetBrains Mono", "SF Mono", Menlo, Consolas, monospace';

export default function EvalBar({ valueEstimate }: Props) {
  // Map [-1, 1] → [0, 1] where 1.0 = full Black advantage
  const blackPct = Math.max(0, Math.min(1, (valueEstimate + 1) / 2));
  const whitePct = 1 - blackPct;

  const blackH = Math.round(blackPct * 100);
  const whiteH = Math.round(whitePct * 100);

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 6,
      }}
    >
      <div
        style={{
          fontSize: 10,
          color: "#8b94a3",
          letterSpacing: "0.12em",
          textTransform: "uppercase",
          fontFamily: MONO,
        }}
      >
        Eval (Black POV)
      </div>

      {/* Vertical bar */}
      <div
        style={{
          width: 28,
          height: 180,
          borderRadius: 6,
          overflow: "hidden",
          background: "#1c1f29",
          position: "relative",
          boxShadow: "inset 0 0 0 1px rgba(255,255,255,0.04)",
        }}
      >
        {/* Black section (top) */}
        <div
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            height: `${blackH}%`,
            background: "linear-gradient(180deg, #1a1d28 0%, #0a0c12 100%)",
            transition: "height 0.4s cubic-bezier(0.4, 0, 0.2, 1)",
          }}
        />
        {/* White section (bottom) */}
        <div
          style={{
            position: "absolute",
            bottom: 0,
            left: 0,
            right: 0,
            height: `${whiteH}%`,
            background: "linear-gradient(180deg, #d8dce8 0%, #f5f6f8 100%)",
            transition: "height 0.4s cubic-bezier(0.4, 0, 0.2, 1)",
          }}
        />
        {/* Centre line */}
        <div
          style={{
            position: "absolute",
            top: "50%",
            left: 0,
            right: 0,
            height: 1,
            background: "rgba(167,139,250,0.6)",
            boxShadow: "0 0 4px rgba(167,139,250,0.6)",
          }}
        />
      </div>

      {/* Labels */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 2,
          fontFamily: MONO,
          fontSize: 10,
        }}
      >
        <span style={{ color: "#8b94a3" }}>
          ⚫ {blackH}%
        </span>
        <span style={{ color: "#8b94a3" }}>
          ⚪ {whiteH}%
        </span>
      </div>
    </div>
  );
}
