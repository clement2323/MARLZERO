import Board from "./components/Board";
import AnalysisPanel from "./components/AnalysisPanel";
import { useGame } from "./hooks/useGame";
import "./App.css";

function statusMessage(
  status: string,
  mustCapture: boolean,
  piecesInHand: [number, number],
  winner: number | null,
  selectedPos: number | null,
  humanPlayer: 1 | 2
): string {
  if (status === "error") return "";
  if (status === "game_over") {
    if (winner === humanPlayer) return "You win!";
    if (winner !== null && winner !== humanPlayer) return "Agent wins.";
    return "Draw.";
  }
  if (status === "thinking") return "Agent is thinking…";
  if (status === "idle") return "Connecting…";
  if (mustCapture) return "Mill formed! Select an opponent piece to capture.";
  if (piecesInHand[humanPlayer - 1] > 0) {
    return `Place a piece (${piecesInHand[humanPlayer - 1]} remaining).`;
  }
  if (selectedPos !== null) return "Click an empty adjacent square to move.";
  return "Select one of your pieces to move.";
}

export default function App() {
  const { gs, legalActions, handlePositionClick, resetGame } = useGame();

  const playerPieces: [number, number] = [
    gs.board.filter((x) => x === 1).length,
    gs.board.filter((x) => x === 2).length,
  ];

  const msg = statusMessage(
    gs.status,
    gs.mustCapture,
    gs.piecesInHand,
    gs.winner,
    gs.selectedPos,
    gs.humanPlayer
  );

  const humanIsWhite = gs.humanPlayer === 1;
  const youGlyph = humanIsWhite ? "○" : "●";
  const agentGlyph = humanIsWhite ? "●" : "○";

  return (
    <div className="app">
      <header className="app-header">
        <h1>Nine Men&apos;s Morris</h1>
        <div className="legend">
          <span className="piece p1" /> You ({youGlyph} {humanIsWhite ? "white" : "black"})
          <span className="piece p2" /> Agent ({agentGlyph} {humanIsWhite ? "black" : "white"})
        </div>
      </header>

      <main className="app-main">
        <div className="board-section">
          <div
            className="status-bar"
            style={{ color: gs.status === "game_over" ? "#f0a500" : "#ddd" }}
          >
            {msg}
          </div>
          <Board
            board={gs.board}
            legalActions={legalActions}
            selectedPos={gs.selectedPos}
            onPositionClick={handlePositionClick}
            disabled={gs.status !== "waiting_human"}
          />
          <div className="controls" style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <button onClick={() => resetGame(gs.humanPlayer)} className="btn">
              New Game
            </button>
            <span style={{ color: "#888", fontSize: 12, marginLeft: 12 }}>Play as:</span>
            <button
              onClick={() => resetGame(1)}
              className="btn"
              style={{
                background: gs.humanPlayer === 1 ? "#e8e8e8" : "#333",
                color: gs.humanPlayer === 1 ? "#1a1a2e" : "#ccc",
                fontWeight: gs.humanPlayer === 1 ? "bold" : "normal",
              }}
            >
              ○ White (first)
            </button>
            <button
              onClick={() => resetGame(2)}
              className="btn"
              style={{
                background: gs.humanPlayer === 2 ? "#1a1a2e" : "#333",
                color: gs.humanPlayer === 2 ? "#e8e8e8" : "#ccc",
                fontWeight: gs.humanPlayer === 2 ? "bold" : "normal",
              }}
            >
              ● Black (second)
            </button>
          </div>
        </div>

        <aside className="panel-section">
          {gs.status === "error" ? (
            <div className="error-box">
              <strong>Connection error</strong>
              <p>{gs.errorMsg}</p>
            </div>
          ) : (
            <AnalysisPanel
              topMoves={gs.topMoves}
              valueEstimate={gs.valueEstimate}
              usingNetwork={gs.usingNetwork}
              agentName={gs.agentName}
              playerPieces={playerPieces}
              piecesInHand={gs.piecesInHand}
              moveHistory={gs.moveHistory}
              humanPlayer={gs.humanPlayer}
            />
          )}
        </aside>
      </main>
    </div>
  );
}
