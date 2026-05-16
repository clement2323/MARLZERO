export interface MoveInfo {
  action: number;
  visit_prob: number;
  description: string;
}

export interface BoardState {
  board: number[];          // 64 ints: 0=empty, 1=black(P1), 2=white(P2)
  current_player: number;   // 1 or 2
  game_over: boolean;
  winner: number | null;    // 1, 2, 0 (draw), or null (ongoing)
  legal_actions: number[];  // valid action indices for current player
  pass_count: number;
}

export interface PlayResponse {
  action: number;
  description: string;
  board_after: BoardState;
  top_moves: MoveInfo[];
  value_estimate: number;   // [-1, 1] from Black's POV
  using_network: boolean;
  agent_name: string;
}

export interface AgentOption {
  id: string;
  label: string;
  available: boolean;
}

export interface AgentsResponse {
  options: AgentOption[];
  default: string;
}
