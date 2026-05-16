import type { AgentsResponse, BoardState, PlayResponse } from "../types/game";

const BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8001";

export async function fetchNewGame(): Promise<BoardState> {
  const res = await fetch(`${BASE_URL}/new-game`);
  if (!res.ok) throw new Error(`/new-game failed: ${res.status}`);
  return res.json() as Promise<BoardState>;
}

export async function fetchAgents(): Promise<AgentsResponse> {
  const res = await fetch(`${BASE_URL}/agents`);
  if (!res.ok) throw new Error(`/agents failed: ${res.status}`);
  return res.json() as Promise<AgentsResponse>;
}

export async function fetchPlay(
  actions: number[],
  agent: string | null = null,
): Promise<PlayResponse> {
  const res = await fetch(`${BASE_URL}/play`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ actions, agent }),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error((detail as { detail: string }).detail ?? res.statusText);
  }
  return res.json() as Promise<PlayResponse>;
}

export async function fetchState(actions: number[]): Promise<BoardState> {
  const res = await fetch(`${BASE_URL}/state`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ actions }),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error((detail as { detail: string }).detail ?? res.statusText);
  }
  return res.json() as Promise<BoardState>;
}

export async function fetchHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${BASE_URL}/health`);
    return res.ok;
  } catch {
    return false;
  }
}
