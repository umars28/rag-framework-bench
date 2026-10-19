from __future__ import annotations

import json
import os
import urllib.request
from typing import Sequence

from .shared import Chunk

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("RAGBENCH_OLLAMA_MODEL", "qwen3:8b")


def prompt(question: str, chunks: Sequence[Chunk]) -> str:
    context = "\n\n".join(f"[{i}] {c.content}" for i, c in enumerate(chunks, start=1))
    return (
        "Answer the question using only the numbered Kubernetes documentation excerpts below. "
        "Cite excerpts by number. If they do not contain the answer, say so.\n\n"
        f"{context}\n\nQuestion: {question}\nAnswer:"
    )


def answer(question: str, chunks: Sequence[Chunk], model: str = OLLAMA_MODEL) -> dict:
    payload = {"model": model, "prompt": prompt(question, chunks), "stream": False, "think": False,
               "options": {"temperature": 0}}
    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        body = json.loads(response.read())
    return {
        "model": model,
        "answer": body.get("response", "").strip(),
        "prompt_tokens": body.get("prompt_eval_count"),
        "output_tokens": body.get("eval_count"),
        "seconds": round(body.get("total_duration", 0) / 1e9, 2),
    }
