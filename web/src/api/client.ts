import type { AgentsResponse, BoardState, PlayResponse } from "../types/game";

const BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export type Variant = "flying" | "no-flying";

export async function fetchNewGame(variant: Variant = "flying"): Promise<BoardState> {
  const res = await fetch(`${BASE_URL}/new-game?variant=${variant}`);
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
  variant: Variant = "flying",
): Promise<PlayResponse> {
  const res = await fetch(`${BASE_URL}/play`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ actions, agent, variant }),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error((detail as { detail: string }).detail ?? res.statusText);
  }
  return res.json() as Promise<PlayResponse>;
}

export async function fetchState(
  actions: number[],
  variant: Variant = "flying",
): Promise<BoardState> {
  const res = await fetch(`${BASE_URL}/state`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ actions, variant }),
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
