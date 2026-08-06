// Typed client for the PromptLens backend (FastAPI — see backend/main.py).
// Both endpoints take the same request body.

// Where the backend lives. Override at build time with VITE_API_URL if needed;
// otherwise default to the local uvicorn server.
export const API_BASE = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

export interface DiffRequest {
  prompt_a: string;
  prompt_b: string;
  top_n: number;
}

// One row of the feature diff: how strongly a feature fired in each prompt, and the change.
export interface DiffRow {
  feature_id: number;
  strength_a: number;
  strength_b: number;
  change: number; // strength_b - strength_a  (positive = stronger in B)
  label: string; // human-readable description from Neuronpedia
}

// A feature that fired in BOTH prompts.
export interface SharedRow {
  feature_id: number;
  strength_a: number;
  strength_b: number;
  label: string;
}

export interface DiffResponse {
  results: DiffRow[]; // top changed features, already sorted by |change| desc
  similarity: number; // cosine similarity of the two feature vectors, 0..1
  shared: SharedRow[];
}

// One named feature in a prompt's NLA verbalization.
export interface NlaFeature {
  feature_id: number;
  strength: number;
  label: string;
}

// The NLA round-trip result for a single prompt.
export interface NlaView {
  verbalization: NlaFeature[]; // top-N features describing the activation
  fidelity: number; // how well those features reconstruct the activation, 0..1
  sae_ceiling: number; // reconstruction from ALL features — an upper bound on fidelity
}

export interface VerbalizeResponse {
  prompt_a: NlaView;
  prompt_b: NlaView;
}

// POST JSON and throw a readable error if the server refuses or is unreachable
// (the UI catches this to show a friendly message).
async function postJson<T>(path: string, body: DiffRequest): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`${path} returned ${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export const postDiff = (req: DiffRequest) => postJson<DiffResponse>("/diff", req);
export const postVerbalize = (req: DiffRequest) =>
  postJson<VerbalizeResponse>("/verbalize", req);

// Link to a feature's Neuronpedia dashboard (the same SAE the backend labels against).
export const neuronpediaUrl = (id: number) =>
  `https://www.neuronpedia.org/gpt2-small/8-res-jb/${id}`;
