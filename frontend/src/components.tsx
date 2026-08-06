// Small presentational components for the diff dashboard. All styling is Tailwind
// utility classes on a dark (slate) theme; no component holds any state.

import type { ReactNode } from "react";
import { neuronpediaUrl } from "./api";
import type { DiffRow, SharedRow, NlaView } from "./api";

// Format a 0..1 value as a percentage string, e.g. 0.413 -> "41%".
const pct = (x: number) => `${Math.round(x * 100)}%`;

// A labelled section wrapper so the dashboard reads as a set of cards.
function Card({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="bg-slate-800/60 rounded-xl p-5 border border-slate-700">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-400 mb-3">
        {title}
      </h2>
      {children}
    </section>
  );
}

// How internally similar the two prompts are (cosine similarity of feature vectors).
export function SimilarityGauge({ value }: { value: number }) {
  return (
    <Card title="Internal similarity">
      <div className="flex items-center gap-4">
        <div className="flex-1 h-3 rounded-full bg-slate-700 overflow-hidden">
          <div
            className="h-full bg-indigo-400"
            style={{ width: pct(value) }}
          />
        </div>
        <span className="tabular-nums font-mono text-lg">{value.toFixed(2)}</span>
      </div>
      <p className="text-xs text-slate-500 mt-2">
        1.00 = identical internal features · 0.00 = completely different
      </p>
    </Card>
  );
}

// One feature in the diff: an arrow for direction, the label, a magnitude bar, the
// signed change, and a link to the feature's Neuronpedia page.
function DiffBar({ row, maxAbs }: { row: DiffRow; maxAbs: number }) {
  const up = row.change > 0; // stronger in prompt B
  const width = `${Math.max(2, (Math.abs(row.change) / maxAbs) * 100)}%`;
  return (
    <li className="flex items-center gap-3 py-1.5">
      <span className={up ? "text-emerald-400" : "text-rose-400"}>{up ? "▲" : "▼"}</span>
      <a
        href={neuronpediaUrl(row.feature_id)}
        target="_blank"
        rel="noreferrer"
        className="w-52 shrink-0 truncate text-sm text-slate-200 hover:text-indigo-300 hover:underline"
        title={`${row.label} — feature #${row.feature_id} (opens Neuronpedia)`}
      >
        {row.label}
      </a>
      <div className="flex-1 h-2.5 rounded bg-slate-700/60 overflow-hidden">
        <div
          className={`h-full ${up ? "bg-emerald-500" : "bg-rose-500"}`}
          style={{ width }}
        />
      </div>
      <span
        className={`w-14 text-right tabular-nums font-mono text-sm ${
          up ? "text-emerald-400" : "text-rose-400"
        }`}
      >
        {row.change > 0 ? "+" : ""}
        {row.change.toFixed(2)}
      </span>
    </li>
  );
}

// The full feature-diff list. `rows` is already sorted by |change| by the backend.
export function FeatureDiff({ rows }: { rows: DiffRow[] }) {
  // Longest bar = the biggest change; everything else is scaled relative to it.
  const maxAbs = Math.max(...rows.map((r) => Math.abs(r.change)), 1e-9);
  return (
    <Card title="What changed (A → B)">
      <ul className="divide-y divide-slate-700/50">
        {rows.map((r) => (
          <DiffBar key={r.feature_id} row={r} maxAbs={maxAbs} />
        ))}
      </ul>
      <p className="text-xs text-slate-500 mt-3">
        <span className="text-emerald-400">▲</span> stronger in B ·{" "}
        <span className="text-rose-400">▼</span> stronger in A
      </p>
    </Card>
  );
}

// Features that fired strongly in BOTH prompts, shown as chips linking to Neuronpedia.
export function SharedFeatures({ rows }: { rows: SharedRow[] }) {
  return (
    <Card title="Shared by both prompts">
      {rows.length === 0 ? (
        <p className="text-sm text-slate-500">No strongly shared features.</p>
      ) : (
        <div className="flex flex-wrap gap-2">
          {rows.map((r) => (
            <a
              key={r.feature_id}
              href={neuronpediaUrl(r.feature_id)}
              target="_blank"
              rel="noreferrer"
              className="rounded-full bg-slate-700 hover:bg-slate-600 px-3 py-1 text-sm text-slate-200"
              title={`feature #${r.feature_id} (opens Neuronpedia)`}
            >
              {r.label}
            </a>
          ))}
        </div>
      )}
    </Card>
  );
}

// NLA reconstruction meter for one prompt: how faithfully its top named features
// rebuild the activation (`fidelity`), with the SAE's best-possible reconstruction
// (`sae_ceiling`) drawn as a marker line.
function FidelityMeter({ title, view }: { title: string; view: NlaView }) {
  const topLabels = view.verbalization.slice(0, 3).map((f) => f.label).join(" · ");
  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span className="text-slate-300">{title}</span>
        <span className="tabular-nums font-mono text-slate-400">
          {view.fidelity.toFixed(2)}{" "}
          <span className="text-slate-600">/ ceiling {view.sae_ceiling.toFixed(2)}</span>
        </span>
      </div>
      <div className="relative h-3 rounded-full bg-slate-700 overflow-hidden">
        <div className="h-full bg-indigo-400" style={{ width: pct(view.fidelity) }} />
        {/* dashed marker for the SAE ceiling (upper bound on fidelity) */}
        <div
          className="absolute top-0 h-full w-0.5 bg-slate-300/70"
          style={{ left: pct(view.sae_ceiling) }}
        />
      </div>
      {topLabels && (
        <p className="text-xs text-slate-500 mt-1 truncate" title={topLabels}>
          top features: {topLabels}
        </p>
      )}
    </div>
  );
}

export function NlaPanel({ a, b }: { a: NlaView; b: NlaView }) {
  return (
    <Card title="NLA reconstruction fidelity">
      <div className="space-y-4">
        <FidelityMeter title="Prompt A" view={a} />
        <FidelityMeter title="Prompt B" view={b} />
      </div>
      <p className="text-xs text-slate-500 mt-3">
        How well each prompt's named features reconstruct its true activation. The marker
        is the SAE ceiling — the best possible reconstruction using every feature.
      </p>
    </Card>
  );
}
