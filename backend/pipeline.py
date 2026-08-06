import math
import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE

# Load once at module import time
print("Loading GPT-2 small...")
torch.set_grad_enabled(False)
model = HookedTransformer.from_pretrained("gpt2", device="cpu")

print("Loading SAE...")
sae, _, _ = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.8.hook_resid_pre",
    device="cpu"
)
print("Pipeline ready.")


def get_features(prompt, top_k=50):
    """Return dict of {feature_id: strength} for the top-k features at layer 8."""
    tokens = model.to_tokens(prompt)
    _, cache = model.run_with_cache(
        tokens,
        names_filter="blocks.8.hook_resid_pre"
    )
    activations = cache["blocks.8.hook_resid_pre"]
    feature_activations = sae.encode(activations)
    last_token_features = feature_activations[0, -1]
    values, indices = last_token_features.topk(top_k)
    return dict(zip(indices.tolist(), values.tolist()))


def cosine_similarity(features_a: dict, features_b: dict) -> float:
    """Cosine similarity between two sparse feature vectors (0.0 = orthogonal, 1.0 = identical)."""
    all_ids = set(features_a.keys()) | set(features_b.keys())
    dot = sum(features_a.get(fid, 0.0) * features_b.get(fid, 0.0) for fid in all_ids)
    mag_a = math.sqrt(sum(v ** 2 for v in features_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in features_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def get_shared_features(features_a: dict, features_b: dict, top_n: int = 5) -> list:
    """Find features that fired strongly in both prompts, sorted by minimum strength."""
    # Only consider features present in both
    shared_ids = set(features_a.keys()) & set(features_b.keys())
    shared = []
    for fid in shared_ids:
        sa = features_a[fid]
        sb = features_b[fid]
        shared.append({
            "feature_id": fid,
            "strength_a": sa,
            "strength_b": sb,
            # min strength = how strong it is in the weaker prompt (both must be at least this strong)
            "min_strength": min(sa, sb),
        })
    # Sort by min_strength descending — features that are strong in BOTH prompts first
    shared.sort(key=lambda row: row["min_strength"], reverse=True)
    return shared[:top_n]


def diff_prompts(prompt_a, prompt_b, top_k=50):
    """Compare two prompts. Return diffs, similarity score, and shared features."""
    features_a = get_features(prompt_a, top_k=top_k)
    features_b = get_features(prompt_b, top_k=top_k)

    all_feature_ids = set(features_a.keys()) | set(features_b.keys())

    diffs = []
    for feat_id in all_feature_ids:
        strength_a = features_a.get(feat_id, 0.0)
        strength_b = features_b.get(feat_id, 0.0)
        change = strength_b - strength_a
        diffs.append({
            "feature_id": feat_id,
            "strength_a": strength_a,
            "strength_b": strength_b,
            "change": change,
        })

    diffs.sort(key=lambda row: abs(row["change"]), reverse=True)

    similarity = cosine_similarity(features_a, features_b)
    shared = get_shared_features(features_a, features_b)

    return {
        "diffs": diffs,
        "similarity": similarity,
        "shared": shared,
    }


# ---------------------------------------------------------------------------
# Natural Language Autoencoder (NLA) — a simplified, local implementation
# ---------------------------------------------------------------------------
# Anthropic's NLA idea: describe a model's internal activation in natural
# language, then prove the description is faithful by RECONSTRUCTING the original
# activation from it and measuring how close the reconstruction is.
#
# A sparse autoencoder already hands us both halves of that loop:
#   - Verbalizer   (activation -> language): the handful of features that fire
#     most strongly. Each has a human-readable label (fetched from Neuronpedia in
#     the API layer), so those labels ARE the natural-language description.
#   - Reconstructor (language -> activation): rebuild the activation from ONLY
#     those named features, using the SAE's decoder (sae.decode).
#   - Fidelity: cosine similarity between the true activation and the
#     reconstruction. 1.0 = the named features fully explain the activation's
#     direction; lower = the description left information behind.
# ---------------------------------------------------------------------------


def _cosine(a, b):
    """Cosine similarity between two 1-D activation tensors.

    Scale-invariant, so it doesn't matter if the SAE's reconstruction comes out at
    a different overall magnitude than the original activation.
    """
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def verbalize_prompt(prompt, top_n=10):
    """Run one prompt through the NLA round-trip and score how faithful it is.

    Returns a dict:
        {
          "features":    {feature_id: strength, ...}  # the top-N features that make
                                                       # up the natural-language description
          "fidelity":    float  # cosine(reconstruction from the top-N features, true activation)
          "sae_ceiling": float  # cosine(reconstruction from ALL SAE features, true activation);
                                 # the best this SAE can do, i.e. an upper bound on fidelity
        }
    """
    # 1. Run the model and grab the layer-8 residual-stream activation of the LAST
    #    token -- the same point in the network the rest of the pipeline analyses.
    tokens = model.to_tokens(prompt)
    _, cache = model.run_with_cache(tokens, names_filter="blocks.8.hook_resid_pre")
    activation = cache["blocks.8.hook_resid_pre"][0, -1]          # shape: [d_model]

    # 2. Encode that activation into the SAE's feature space. Most of the ~24k
    #    features sit at ~0; only a few are meaningfully active. (unsqueeze/squeeze
    #    just add and remove the batch dimension the SAE expects.)
    features = sae.encode(activation.unsqueeze(0)).squeeze(0)     # shape: [d_sae]

    # 3. VERBALIZE: keep only the top-N strongest features. These are the ones we
    #    can name in English, so they stand in for the natural-language summary.
    values, indices = features.topk(top_n)
    named = torch.zeros_like(features)   # start from all-zeros...
    named[indices] = values              # ...and switch on just the named features

    # 4. RECONSTRUCT: decode back into activation space -- once from the named
    #    features only, and once from every feature (the SAE's own ceiling).
    named_recon = sae.decode(named.unsqueeze(0)).squeeze(0)       # from the top-N only
    full_recon = sae.decode(features.unsqueeze(0)).squeeze(0)     # from all features

    # 5. SCORE: how well does each reconstruction match the true activation?
    return {
        "features": dict(zip(indices.tolist(), values.tolist())),
        "fidelity": _cosine(named_recon, activation),
        "sae_ceiling": _cosine(full_recon, activation),
    }