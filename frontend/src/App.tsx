import { useState } from "react";
import { postDiff, postVerbalize, API_BASE } from "./api";
import type { DiffResponse, VerbalizeResponse } from "./api";
import { SimilarityGauge, FeatureDiff, SharedFeatures, NlaPanel } from "./components";

// Prefill an example pair so a first-time visitor can just hit "Compare".
const EXAMPLE_A = "write a poem about spring";
const EXAMPLE_B = "write a legal lease contract";

function App() {
  const [promptA, setPromptA] = useState(EXAMPLE_A);
  const [promptB, setPromptB] = useState(EXAMPLE_B);
  const [topN, setTopN] = useState(10);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [diff, setDiff] = useState<DiffResponse | null>(null);
  const [nla, setNla] = useState<VerbalizeResponse | null>(null);

  async function runCompare() {
    setLoading(true);
    setError(null);
    const body = { prompt_a: promptA, prompt_b: promptB, top_n: topN };
    try {
      // /diff gives the similarity + feature diff + shared; /verbalize gives the
      // per-prompt NLA fidelity. Fire both at once.
      const [d, n] = await Promise.all([postDiff(body), postVerbalize(body)]);
      setDiff(d);
      setNla(n);
    } catch (e) {
      const detail = e instanceof Error ? e.message : String(e);
      setError(
        `Couldn't reach the backend at ${API_BASE}. Is it running? (${detail})`,
      );
      setDiff(null);
      setNla(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100">
      <div className="max-w-4xl mx-auto px-6 py-10 space-y-6">
        {/* Header */}
        <header>
          <h1 className="text-3xl font-bold">PromptLens</h1>
          <p className="text-slate-400 mt-1">
            Mechanistic prompt diffing — see which internal features of a language model
            change when you edit a prompt.
          </p>
        </header>

        {/* Input form */}
        <div className="bg-slate-800/60 rounded-xl p-5 border border-slate-700 space-y-4">
          <div className="grid md:grid-cols-2 gap-4">
            <label className="block">
              <span className="text-sm text-slate-400">Prompt A</span>
              <textarea
                className="mt-1 w-full h-24 rounded-lg bg-slate-900 border border-slate-700 p-3 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400"
                value={promptA}
                onChange={(e) => setPromptA(e.target.value)}
              />
            </label>
            <label className="block">
              <span className="text-sm text-slate-400">Prompt B</span>
              <textarea
                className="mt-1 w-full h-24 rounded-lg bg-slate-900 border border-slate-700 p-3 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400"
                value={promptB}
                onChange={(e) => setPromptB(e.target.value)}
              />
            </label>
          </div>

          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 text-sm text-slate-400">
              features
              <input
                type="number"
                min={1}
                max={50}
                className="w-16 rounded-lg bg-slate-900 border border-slate-700 px-2 py-1 text-slate-100"
                value={topN}
                onChange={(e) =>
                  setTopN(Math.max(1, Math.min(50, Number(e.target.value) || 1)))
                }
              />
            </label>
            <button
              onClick={runCompare}
              disabled={loading}
              className="rounded-lg bg-indigo-500 hover:bg-indigo-400 disabled:opacity-50 disabled:cursor-not-allowed px-5 py-2 font-medium"
            >
              {loading ? "Comparing…" : "Compare"}
            </button>
            {loading && (
              <span className="text-sm text-slate-500">
                first run loads GPT-2 + the SAE (~1 min)…
              </span>
            )}
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="rounded-lg bg-rose-950/60 border border-rose-800 text-rose-200 px-4 py-3 text-sm">
            {error}
          </div>
        )}

        {/* Results */}
        {diff && nla && (
          <div className="space-y-5">
            <SimilarityGauge value={diff.similarity} />
            <FeatureDiff rows={diff.results} />
            <SharedFeatures rows={diff.shared} />
            <NlaPanel a={nla.prompt_a} b={nla.prompt_b} />
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
