export interface BoardState {
  board: number[];        // 24 cells: 0=empty, 1=player1, 2=player2
  current_player: number;
  pieces_in_hand: number[];
  must_capture: boolean;
  game_over: boolean;
  winner: number | null;
  legal_actions: number[];
}

export interface MoveInfo {
  action: number;
  visit_prob: number;
  description: string;
}

export interface PlayResponse {
  action: number;
  description: string;
  top_moves: MoveInfo[];
  value_estimate: number;
  board_after: BoardState;
  using_network: boolean;
  agent_name: string;
}

export type GamePhase = "placing" | "moving" | "game_over";

export interface AgentOption {
  id: string;
  label: string;
  available: boolean;
}

export interface AgentsResponse {
  options: AgentOption[];
  default: string;
}
