import os
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from pipeline import diff_prompts

load_dotenv()
NEURONPEDIA_API_KEY = os.getenv("sk-np-ptVxI4ZDMQjtARYs3FoioHYb0bNvsqCFZ6D2503dpj40")

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
    try:
        r = requests.get(url, timeout=5)
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


@app.post("/diff")
def diff(req: DiffRequest):
    results = diff_prompts(req.prompt_a, req.prompt_b)

    labeled = []
    for row in results[:req.top_n]:
        labeled.append({
            "feature_id": row["feature_id"],
            "strength_a": round(row["strength_a"], 2),
            "strength_b": round(row["strength_b"], 2),
            "change": round(row["change"], 2),
            "label": fetch_label(row["feature_id"]),
        })

    return {"results": labeled}