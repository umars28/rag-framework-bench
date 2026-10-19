from __future__ import annotations

import base64
import json
import os
import time
import urllib.parse
import urllib.request
from typing import Any, Callable

TRUTHY = {"1", "true", "yes", "on"}


def langsmith_status() -> str:
    wants = os.environ.get("LANGSMITH_TRACING", "").lower() in TRUTHY
    has_key = bool(os.environ.get("LANGSMITH_API_KEY"))
    if wants and not has_key:
        os.environ["LANGSMITH_TRACING"] = "false"
        return "requested, but LANGSMITH_API_KEY is not set, so it stays off"
    return "on" if wants else "off"


def flush_langsmith() -> None:
    from langchain_core.tracers.langchain import wait_for_all_tracers

    wait_for_all_tracers()


def langfuse_configured() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


def langfuse_client():
    if not langfuse_configured():
        return None
    from langfuse import get_client

    return get_client()


def instrument_llamaindex() -> bool:
    try:
        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    except ImportError:
        return False
    LlamaIndexInstrumentor().instrument()
    return True


class _ScorerSpan:
    def __init__(self, client, scorer) -> None:
        self.client = client
        self.scorer = scorer

    def predict(self, pairs, **kwargs: Any):
        with self.client.start_as_current_observation(
            name="cross-encoder", as_type="span", metadata={"pairs": len(pairs)}
        ):
            return self.scorer.predict(pairs, **kwargs)


class Traced:
    def __init__(self, pipeline, client, session_id: str) -> None:
        self.pipeline = pipeline
        self.client = client
        self.session_id = session_id
        self.name = pipeline.name
        if pipeline.name == "baseline":
            pipeline.dense = self._child("dense", pipeline.dense)
            pipeline.lexical = self._child("bm25", pipeline.lexical)
            pipeline.scorer = _ScorerSpan(client, pipeline.scorer)
        if hasattr(pipeline, "callbacks"):
            from langfuse.langchain import CallbackHandler

            pipeline.callbacks.append(CallbackHandler())

    def _child(self, name: str, fn: Callable[[str, int], list[str]]):
        def wrapped(query: str, k: int) -> list[str]:
            with self.client.start_as_current_observation(
                name=name, as_type="retriever", input={"query": query, "k": k}
            ) as span:
                ids = fn(query, k)
                span.update(output={"top": ids[:5]})
                return ids

        return wrapped

    def _run(self, stage: str, query: str) -> list[str]:
        from langfuse import propagate_attributes

        with self.client.start_as_current_observation(
            name=f"{self.name}.{stage}", as_type="retriever", input={"query": query}
        ) as span:
            with propagate_attributes(
                session_id=self.session_id, tags=[self.name, stage], trace_name=f"{self.name}.{stage}"
            ):
                ids = getattr(self.pipeline, stage)(query)
            span.update(output={"top": ids[:10]})
        return ids

    def hybrid(self, query: str) -> list[str]:
        return self._run("hybrid", query)

    def reranked(self, query: str) -> list[str]:
        return self._run("reranked", query)


def _get(path: str, params: dict[str, Any]) -> dict:
    base = os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST", "http://localhost:3300")
    token = base64.b64encode(
        f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}".encode()
    ).decode()
    url = f"{base.rstrip('/')}{path}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"Authorization": f"Basic {token}"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def stored_traces(session_id: str, expected: int, timeout: float = 120.0) -> list[dict]:
    deadline = time.monotonic() + timeout
    traces: list[dict] = []
    while time.monotonic() < deadline:
        traces = _get("/api/public/traces", {"sessionId": session_id, "limit": 100})["data"]
        if len(traces) >= expected:
            break
        time.sleep(3)
    return traces


def observation_counts(trace_ids: list[str]) -> dict[str, int]:
    return {tid: len(_get(f"/api/public/traces/{tid}", {}).get("observations", [])) for tid in trace_ids}
