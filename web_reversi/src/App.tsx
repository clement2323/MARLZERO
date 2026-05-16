import Board from "./components/Board";
import AnalysisPanel from "./components/AnalysisPanel";
import { useGame } from "./hooks/useGame";
import "./App.css";

const PASS_ACTION = 64;

function statusMessage(
  status: string,
  winner: number | null,
  humanPlayer: 1 | 2,
  currentPlayer: number,
  legalActions: number[],
): string {
  if (status === "error") return "";
  if (status === "game_over") {
    if (winner === 0) return "Draw.";
    if (winner === humanPlayer) return "You win!";
    if (winner !== null) return "Agent wins.";
    return "Game over.";
  }
  if (status === "thinking") return "Agent is thinking…";
  if (status === "idle") return "Connecting…";
  if (status === "waiting_human") {
    if (legalActions.length === 1 && legalActions[0] === PASS_ACTION) {
      return "No legal moves — passing…";
    }
    const color = currentPlayer === 1 ? "Black" : "White";
    return `${color} to move. Click a highlighted cell.`;
  }
  return "";
}

export default function App() {
  const { gs, handleCellClick, resetGame, setSelectedAgent } = useGame();

  const humanIsBlack = gs.humanPlayer === 1;
  const youLabel = humanIsBlack ? "⚫ Black" : "⚪ White";
  const agentLabel = humanIsBlack ? "⚪ White" : "⚫ Black";

  const msg = statusMessage(
    gs.status,
    gs.winner,
    gs.humanPlayer,
    gs.currentPlayer,
    gs.legalActions,
  );

  // Only show legal actions when it's the human's turn
  const legalForBoard = gs.status === "waiting_human" ? gs.legalActions.filter(a => a !== PASS_ACTION) : [];

  return (
    <div className="app">
      <header className="app-header">
        <h1>Reversi</h1>
        <div className="legend">
          <span>
            <span className={`piece-dot ${humanIsBlack ? "black" : "white"}`} />
            You ({youLabel})
          </span>
          <span>
            <span className={`piece-dot ${humanIsBlack ? "white" : "black"}`} />
            Agent ({agentLabel})
          </span>
        </div>
      </header>

      <main className="app-main">
        <div className="board-section">
          <div
            className={`status-bar${gs.status === "game_over" ? " is-game-over" : ""}`}
          >
            {msg}
          </div>

          <Board
            board={gs.board}
            legalActions={legalForBoard}
            onCellClick={handleCellClick}
            disabled={gs.status !== "waiting_human"}
            lastMove={gs.lastMove}
            lastMoveKey={gs.moveHistory.length}
          />

          <div className="controls">
            <button onClick={() => resetGame(gs.humanPlayer)} className="btn">
              New Game
            </button>
            <span className="controls-label">Play as</span>
            <button
              onClick={() => resetGame(1)}
              className={`btn${gs.humanPlayer === 1 ? " is-active-black" : ""}`}
            >
              ⚫ Black
            </button>
            <button
              onClick={() => resetGame(2)}
              className={`btn${gs.humanPlayer === 2 ? " is-active-white" : ""}`}
            >
              ⚪ White
            </button>
          </div>

          {gs.availableAgents.length > 0 && (
            <div className="agent-row">
              <span className="controls-label">Vs</span>
              <select
                className="agent-select"
                value={gs.selectedAgent ?? ""}
                onChange={(e) => setSelectedAgent(e.target.value)}
              >
                {gs.availableAgents.map((opt) => (
                  <option key={opt.id} value={opt.id} disabled={!opt.available}>
                    {opt.label}
                    {opt.available ? "" : " (unavailable)"}
                  </option>
                ))}
              </select>
            </div>
          )}
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
              board={gs.board}
              moveHistory={gs.moveHistory}
              humanPlayer={gs.humanPlayer}
            />
          )}
        </aside>
      </main>
    </div>
  );
}
