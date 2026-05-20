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


def diff_prompts(prompt_a, prompt_b, top_k=50):
    """Compare two prompts. Return a sorted list of dicts."""
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
    return diffs