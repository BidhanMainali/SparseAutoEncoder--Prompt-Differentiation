# PromptLens — Mechanistic Prompt Diffing

**See which internal features of a language model change when you edit a prompt.**

You give PromptLens two prompts. It runs both through GPT-2, reads the model's
*internal* representations with a **sparse autoencoder (SAE)**, and shows you which
human-interpretable features fired differently — turning an opaque activation vector
into a concrete answer like *"editing the prompt this way turned up the 'legal
language' feature and turned down the 'casual narration' feature."*

It can also run a **Natural Language Autoencoder** check on each prompt — describe the
activation with its top named features, rebuild the activation from those features
alone, and score how faithfully they capture it (see below).

---

## Why this exists (the research)

Modern language models pack thousands of concepts into each activation vector, in
superposition — you can't read them off directly. Anthropic's interpretability work
showed that a **sparse autoencoder** can decompose those dense activations into a
large dictionary of sparse, often human-interpretable **features** ("the French
language", "legal documents", "DNA sequences", …):

- **Towards Monosemanticity: Decomposing Language Models With Dictionary Learning** —
  Anthropic, 2023. https://transformer-circuits.pub/2023/monosemantic-features
- **Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet** —
  Anthropic, 2024. https://transformer-circuits.pub/2024/scaling-monosemanticity

PromptLens applies that idea to **prompt diffing**: instead of inspecting one prompt,
it compares the SAE feature fingerprints of two prompts.

The one weakness those papers name is that SAE outputs are *"still complex objects
that trained researchers need to carefully interpret"* — a list of numeric feature
IDs. Anthropic's follow-up tackles exactly that:

- **Natural Language Autoencoders (NLAs)** — Anthropic, May 2026.
  https://www.anthropic.com/research/natural-language-autoencoders

An NLA verbalizes an activation into language (the **Activation Verbalizer**),
reconstructs the activation from that description alone (the **Activation
Reconstructor**), and scores the round-trip by how faithfully the reconstruction
matches the original.

The `/verbalize` endpoint implements this loop directly on top of the SAE — no extra
model, no training. Because the SAE *is* an autoencoder, its two halves give us the
NLA round-trip for free:

- **Verbalizer (activation → language):** the top-N features that fire, each carrying a
  Neuronpedia label — those labels are the natural-language description.
- **Reconstructor (language → activation):** rebuild the activation from *only* those
  named features via the SAE's decoder (`sae.decode`).
- **Fidelity:** cosine similarity between the true activation and the reconstruction.
  1.0 means the named features fully explain the activation's direction; lower means
  the description left information behind. We also report the **SAE ceiling** — the
  reconstruction from *all* features — as an upper bound.

---

## How it works

```
prompt A ─┐
          ├─► GPT-2 small ─► residual stream @ layer 8 ─► SAE.encode() ─► top-k features
prompt B ─┘                                                                   │
                                                                              ▼
                          diff · cosine similarity · shared features ──► Neuronpedia labels

  NLA round-trip (/verbalize), per prompt:
      activation ─► SAE.encode ─► top-N named features ─► SAE.decode ─► reconstruction
                         └──────────────► cosine(reconstruction, activation) = fidelity
```

- **Model:** GPT-2 small via [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens).
- **SAE:** the open-source `gpt2-small-res-jb` SAE (Joseph Bloom) on the layer-8
  residual stream (`blocks.8.hook_resid_pre`), loaded via
  [SAELens](https://github.com/jbloomAus/SAELens). This is a community SAE that
  operationalizes the Anthropic methodology above — not Anthropic's own SAE.
- **Features:** for each prompt we take the top-50 SAE features at the final token
  (`backend/pipeline.py: get_features`).
- **Diff:** per-feature `strength_b - strength_a`, sorted by absolute change, plus a
  **cosine similarity** of the two feature vectors and the **shared** features strong
  in both (`backend/pipeline.py: diff_prompts`).
- **Labels:** each feature ID is given a human-readable description from
  [Neuronpedia](https://www.neuronpedia.org) (`backend/main.py: fetch_label`).
- **NLA reconstruction:** for each prompt, reconstruct the activation from only its
  top-N named features and score the fidelity — the round-trip described above
  (`backend/pipeline.py: verbalize_prompt`).

---

## Project layout

```
backend/
  pipeline.py      # GPT-2 + SAE: feature extraction, diff, and the NLA reconstruction (verbalize_prompt)
  main.py          # FastAPI app: /diff and /verbalize, Neuronpedia labelling
  requirements.txt
frontend/          # React 19 + TypeScript + Vite + Tailwind
  src/api.ts        # typed client for /diff and /verbalize
  src/components.tsx# dashboard widgets: similarity gauge, diff bars, fidelity meters
  src/App.tsx       # the diff dashboard (two prompts → Compare → results)
```

---

## API

Both endpoints accept the same body:

```json
{ "prompt_a": "write a poem about spring", "prompt_b": "write a legal lease contract", "top_n": 10 }
```

### `POST /diff`
Returns the labeled feature diff, the cosine similarity, and the shared features.

```json
{
  "results":    [{ "feature_id": 12345, "strength_a": 0.0, "strength_b": 6.2, "change": 6.2, "label": "legal / formal language" }],
  "similarity": 0.41,
  "shared":     [{ "feature_id": 678, "strength_a": 3.1, "strength_b": 2.9, "label": "references to time" }]
}
```

### `POST /verbalize`
The NLA round-trip, run on each prompt independently. For A and B it returns the
`verbalization` (the top-N named features that describe the activation), the `fidelity`
(how well those features reconstruct the activation), and the `sae_ceiling` (the
reconstruction from *all* features — an upper bound). `top_n` sets how many features
are named. Use `/diff` to compare the two prompts against each other.

```json
{
  "prompt_a": {
    "verbalization": [{ "feature_id": 678,   "strength": 5.4, "label": "nature / seasonal imagery" }],
    "fidelity": 0.71,
    "sae_ceiling": 0.94
  },
  "prompt_b": {
    "verbalization": [{ "feature_id": 12345, "strength": 6.2, "label": "legal / formal language" }],
    "fidelity": 0.68,
    "sae_ceiling": 0.93
  }
}
```

---

## Setup

### Backend

```bash
cd backend
python -m venv venv
# Windows: venv\Scripts\activate   ·   macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload          # http://127.0.0.1:8000
```

Create `backend/.env` (gitignored):

```
NEURONPEDIA_API_KEY=...     # optional — labels work unauthenticated, a key just raises rate limits
```

> First run downloads GPT-2 small and the SAE, so give it a minute. Everything runs on
> CPU by default.

### Frontend

```bash
cd frontend
npm install
npm run dev                        # http://localhost:5173 (CORS is pre-allowed for it)
```

Start the backend first, then open the dashboard at **http://localhost:5173**: enter
two prompts, click **Compare**, and you'll get the similarity gauge, the ▲/▼ feature
diff (each label links to Neuronpedia), the shared features, and per-prompt NLA fidelity
meters. The backend can also be driven directly via the FastAPI docs at
`http://127.0.0.1:8000/docs`.

Point the UI at a non-default backend with `VITE_API_URL` (defaults to
`http://127.0.0.1:8000`).

---

## References

1. Anthropic — *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning* (2023). https://transformer-circuits.pub/2023/monosemantic-features
2. Anthropic — *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet* (2024). https://transformer-circuits.pub/2024/scaling-monosemanticity
3. Anthropic — *Natural Language Autoencoders* (2026). https://www.anthropic.com/research/natural-language-autoencoders
4. Neuronpedia — feature dashboards and labels. https://www.neuronpedia.org
5. SAELens (SAE loading) · TransformerLens (model hooks). https://github.com/jbloomAus/SAELens · https://github.com/TransformerLensOrg/TransformerLens
