import { useState } from "react";
import Board from "./components/Board";
import AnalysisPanel from "./components/AnalysisPanel";
import RulesTheater from "./components/RulesTheater";
import { useGame } from "./hooks/useGame";
import { useShake } from "./hooks/useShake";
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
  const { jitter, trigger: triggerShake } = useShake();
  const [loserKey, setLoserKey] = useState(0);

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
          <span>
            <span className={humanIsWhite ? "piece p1" : "piece p2"} />
            You ({youGlyph} {humanIsWhite ? "white" : "black"})
          </span>
          <span>
            <span className={humanIsWhite ? "piece p2" : "piece p1"} />
            Agent ({agentGlyph} {humanIsWhite ? "black" : "white"})
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
            legalActions={legalActions}
            selectedPos={gs.selectedPos}
            onPositionClick={handlePositionClick}
            disabled={gs.status !== "waiting_human"}
            jitter={jitter}
            lastPlacedPos={gs.lastPlacedPos}
            lastMoveKey={gs.moveHistory.length}
          />
          <div className="controls">
            <button onClick={() => resetGame(gs.humanPlayer)} className="btn">
              New Game
            </button>
            <span className="controls-label">Play as</span>
            <button
              onClick={() => resetGame(1)}
              className={`btn${gs.humanPlayer === 1 ? " is-active-white" : ""}`}
            >
              ○ White
            </button>
            <button
              onClick={() => resetGame(2)}
              className={`btn${gs.humanPlayer === 2 ? " is-active-black" : ""}`}
            >
              ● Black
            </button>
            <span className="controls-label">Test</span>
            <button onClick={triggerShake} className="btn is-test">
              Shake
            </button>
            <button
              onClick={() => setLoserKey((k) => k + 1)}
              className="btn is-test"
            >
              Loser
            </button>
          </div>

          <div className="rules-block">
            <RulesTheater triggerKey={loserKey} />
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
