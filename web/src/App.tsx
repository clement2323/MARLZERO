import { useEffect, useRef, useState } from "react";
import Board from "./components/Board";
import AnalysisPanel from "./components/AnalysisPanel";
import RulesTheater from "./components/RulesTheater";
import { useGame } from "./hooks/useGame";
import { useShake } from "./hooks/useShake";
import "./App.css";

// Auto-trigger thresholds — kept here so they're easy to tweak from one spot.
// Mirror the resign-detection logic used during training (threshold -0.9 there,
// looser -0.7 here so the UI reacts before the position is hopeless).
const LOSING_VALUE_THRESHOLD = 0.7;     // |value| above this counts as "losing"
const LOSING_VALUE_STREAK = 3;          // consecutive plies above threshold
const LOSING_PIECES_DIFF = -3;          // pieces_diff <= this triggers shake
const MOVING_PHASE_PIECES_IN_HAND = 0;  // both hands empty = movement phase

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
  const { gs, legalActions, handlePositionClick, resetGame, setSelectedAgent } = useGame();
  const { jitter, trigger: triggerShake } = useShake();
  const [loserKey, setLoserKey] = useState(0);
  const losingValueStreakRef = useRef(0);
  const lastTickProcessed = useRef(0);
  const loserPlayedForGame = useRef(false);

  const playerPieces: [number, number] = [
    gs.board.filter((x) => x === 1).length,
    gs.board.filter((x) => x === 2).length,
  ];

  // Auto-trigger animations based on signals returned by the server.
  // Fires once per agent reply (responseTick) so we don't double-fire on
  // unrelated re-renders. The "is losing" logic is:
  //   value > +0.7 (agent winning, i.e. human losing) for 3 consecutive
  //   replies, OR pieces_diff <= -3 during the movement phase.
  useEffect(() => {
    if (gs.responseTick === lastTickProcessed.current) return;
    lastTickProcessed.current = gs.responseTick;

    // The server's value_estimate is from the agent's POV at the root state
    // BEFORE the agent's move — so value > 0 means agent is winning (= human
    // losing).  pieces_diff is from the human's POV on board_after.
    const humanLosingByValue = gs.valueEstimate > LOSING_VALUE_THRESHOLD;
    losingValueStreakRef.current = humanLosingByValue
      ? losingValueStreakRef.current + 1
      : 0;

    const inMovementPhase =
      gs.piecesInHand[0] <= MOVING_PHASE_PIECES_IN_HAND &&
      gs.piecesInHand[1] <= MOVING_PHASE_PIECES_IN_HAND;
    const humanLosingByPieces = inMovementPhase && gs.piecesDiff <= LOSING_PIECES_DIFF;

    const shouldShake =
      losingValueStreakRef.current >= LOSING_VALUE_STREAK || humanLosingByPieces;
    if (shouldShake) triggerShake();
  }, [gs.responseTick, gs.valueEstimate, gs.piecesDiff, gs.piecesInHand, triggerShake]);

  // Loser overlay on real game-over loss — fire exactly once per game.
  useEffect(() => {
    if (gs.status !== "game_over") {
      loserPlayedForGame.current = false;
      return;
    }
    if (loserPlayedForGame.current) return;
    const humanLost = gs.winner !== null && gs.winner !== gs.humanPlayer;
    if (humanLost) {
      loserPlayedForGame.current = true;
      setLoserKey((k) => k + 1);
    }
  }, [gs.status, gs.winner, gs.humanPlayer]);

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
                  <option
                    key={opt.id}
                    value={opt.id}
                    disabled={!opt.available}
                  >
                    {opt.label}
                    {opt.available ? "" : " (unavailable)"}
                  </option>
                ))}
              </select>
            </div>
          )}

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
