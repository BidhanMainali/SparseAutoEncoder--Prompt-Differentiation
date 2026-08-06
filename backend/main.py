import os
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from pipeline import diff_prompts
from verbalize import verbalize

load_dotenv()
NEURONPEDIA_API_KEY = os.getenv("NEURONPEDIA_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class DiffRequest(BaseModel):
    prompt_a: str
    prompt_b: str
    top_n: int = 10


def fetch_label(feature_id):
    url = f"https://www.neuronpedia.org/api/feature/gpt2-small/8-res-jb/{feature_id}"
    headers = {"X-Api-Key": NEURONPEDIA_API_KEY} if NEURONPEDIA_API_KEY else {}
    try:
        r = requests.get(url, headers=headers, timeout=5)
        data = r.json()
        explanations = data.get("explanations", [])
        if explanations:
            return explanations[0].get("description", "no label")
        return "no label"
    except Exception:
        return "no label"


@app.get("/")
def read_root():
    return {"message": "PromptLens backend is alive"}


def _labeled(req: DiffRequest):
    """Run the SAE diff and attach Neuronpedia labels.

    Shared by /diff and /verbalize. Returns (labeled_diffs, labeled_shared, similarity).
    """
    output = diff_prompts(req.prompt_a, req.prompt_b)

    # Label the top-changed features (the diff)
    labeled_diffs = []
    for row in output["diffs"][:req.top_n]:
        labeled_diffs.append({
            "feature_id": row["feature_id"],
            "strength_a": round(row["strength_a"], 2),
            "strength_b": round(row["strength_b"], 2),
            "change": round(row["change"], 2),
            "label": fetch_label(row["feature_id"]),
        })

    # Label the top shared features
    labeled_shared = []
    for row in output["shared"]:
        labeled_shared.append({
            "feature_id": row["feature_id"],
            "strength_a": round(row["strength_a"], 2),
            "strength_b": round(row["strength_b"], 2),
            "label": fetch_label(row["feature_id"]),
        })

    return labeled_diffs, labeled_shared, round(output["similarity"], 4)


@app.post("/diff")
def diff(req: DiffRequest):
    labeled_diffs, labeled_shared, similarity = _labeled(req)
    return {
        "results": labeled_diffs,
        "similarity": similarity,
        "shared": labeled_shared,
    }


@app.post("/verbalize")
def verbalize_endpoint(req: DiffRequest):
    """NLA-inspired: the /diff payload plus a plain-English summary of the diff
    and a round-trip fidelity score (see verbalize.py)."""
    labeled_diffs, labeled_shared, similarity = _labeled(req)
    return {
        "results": labeled_diffs,
        "similarity": similarity,
        "shared": labeled_shared,
        "nla": verbalize(labeled_diffs, labeled_shared),
    }