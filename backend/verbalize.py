"""NLA-inspired verbalizer for the SAE prompt diff.

Anthropic's Natural Language Autoencoders (NLAs) verbalize a model's internal
activation into plain English (the Activation Verbalizer, AV), then reconstruct
the activation from that text alone (the Activation Reconstructor, AR), and score
how faithful the round-trip is.

This module is a lightweight *approximation* of that idea on top of our SAE feature
diff: Claude plays the AV and AR over the already-labeled features. It is NOT a
trained AV/AR pair over raw activation vectors -- it operates on labeled SAE
features. The fidelity score is what keeps it honest: the English summary is only
trusted to the degree its reconstruction covers the features that were truly active.

The whole module degrades gracefully: with no `anthropic` package or no
ANTHROPIC_API_KEY, `verbalize()` returns an `enabled: False` stub and nothing breaks.
"""

import os
import json

try:
    from anthropic import Anthropic
except ImportError:  # keep the backend importable without the SDK installed
    Anthropic = None

# Haiku 4.5 is cheap and fast -- plenty for summarizing a handful of feature
# labels. Override with VERBALIZER_MODEL (e.g. claude-sonnet-4-6) for richer prose.
VERBALIZER_MODEL = os.getenv("VERBALIZER_MODEL", "claude-haiku-4-5-20251001")

_client = None


def _get_client():
    """Return a cached Anthropic client, or None if the SDK/key is unavailable."""
    global _client
    if _client is not None:
        return _client
    if Anthropic is None or not os.getenv("ANTHROPIC_API_KEY"):
        return None
    _client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    return _client


def _ask(client, prompt, max_tokens=400):
    """Single-turn Claude call; returns the concatenated text response."""
    msg = client.messages.create(
        model=VERBALIZER_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in msg.content if block.type == "text").strip()


def _parse_json_list(raw):
    """Best-effort extraction of a JSON array from a model response."""
    try:
        return json.loads(raw[raw.index("["):raw.rindex("]") + 1])
    except (ValueError, json.JSONDecodeError):
        return None


def _parse_json_obj(raw):
    """Best-effort extraction of a JSON object from a model response."""
    try:
        return json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return None


def verbalize_diff(client, labeled_diffs, labeled_shared):
    """AV step: turn the labeled feature diff into a plain-English summary."""
    up = [f"- {d['label']} ({d['change']:+.2f})" for d in labeled_diffs if d["change"] > 0]
    down = [f"- {d['label']} ({d['change']:+.2f})" for d in labeled_diffs if d["change"] < 0]
    shared = [f"- {s['label']}" for s in labeled_shared]

    prompt = f"""You are interpreting a language model's internal features (from a sparse autoencoder) to explain how its internal representation changes between two prompts, A and B.

Features that fired MORE strongly in prompt B:
{chr(10).join(up) or "- (none)"}

Features that fired MORE strongly in prompt A:
{chr(10).join(down) or "- (none)"}

Features active in BOTH prompts:
{chr(10).join(shared) or "- (none)"}

In 2-3 sentences, describe in plain English how the model's internal focus shifts when going from prompt A to prompt B. Refer to concepts, not feature numbers. Be concrete and specific."""
    return _ask(client, prompt)


def reconstruct_concepts(client, summary):
    """AR step: from the English summary ALONE, predict which concepts are active."""
    prompt = f"""The following is a plain-English description of how a language model's internal representation differs between two prompts:

\"\"\"{summary}\"\"\"

Based ONLY on this description, list the distinct concepts you would expect to be active in the model. Return a JSON array of short concept strings (max 12) and nothing else."""
    return _parse_json_list(_ask(client, prompt))


def score_fidelity(client, reconstructed, actual_labels):
    """Score how well the reconstructed concepts cover the truly-active features (0-1)."""
    prompt = f"""Ground-truth concepts that were actually active in the model:
{json.dumps(actual_labels, indent=2)}

Concepts reconstructed from a text description alone:
{json.dumps(reconstructed, indent=2)}

What fraction of the ground-truth concepts are semantically captured by the reconstructed concepts? Treat paraphrases and near-synonyms as matches. Return only a JSON object: {{"fidelity": <0..1>, "matched": <int>, "total": <int>}}."""
    return _parse_json_obj(_ask(client, prompt))


def verbalize(labeled_diffs, labeled_shared):
    """Full NLA-inspired round-trip: AV -> AR -> fidelity score.

    Returns a dict with the summary, reconstructed concepts, and fidelity, or an
    `enabled: False` stub when the Claude API is unavailable/unconfigured.
    """
    client = _get_client()
    if client is None:
        return {
            "enabled": False,
            "summary": None,
            "reconstructed_concepts": None,
            "fidelity": None,
            "note": "Set ANTHROPIC_API_KEY and install `anthropic` to enable verbalization.",
        }

    summary = verbalize_diff(client, labeled_diffs, labeled_shared)
    reconstructed = reconstruct_concepts(client, summary)

    # Ground truth = every actually-active label, de-duplicated but order-preserved.
    seen = set()
    actual_labels = [
        row["label"]
        for row in labeled_diffs + labeled_shared
        if not (row["label"] in seen or seen.add(row["label"]))
    ]
    fidelity = score_fidelity(client, reconstructed, actual_labels) if reconstructed else None

    return {
        "enabled": True,
        "model": VERBALIZER_MODEL,
        "summary": summary,
        "reconstructed_concepts": reconstructed,
        "fidelity": fidelity,
    }
