from __future__ import annotations

import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ragbench import tracing
from ragbench.evaluate import builder
from ragbench.shared import Settings

SMALL = Settings(dense_k=5, lexical_k=5, fused_k=8, final_k=5)


def _reset_langsmith_caches():
    import langsmith.run_trees as run_trees
    import langsmith.utils as utils

    utils.get_env_var.cache_clear()
    run_trees._CLIENT = None


@pytest.fixture
def capture_server():
    received: list[tuple[str, str, bytes]] = []

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, body: bytes = b"{}") -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            received.append(("GET", self.path, b""))
            self._reply()

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length)
            if self.headers.get("Content-Encoding") == "zstd":
                import zstandard

                body = zstandard.ZstdDecompressor().decompressobj().decompress(body)
            received.append(("POST", self.path, body))
            self._reply()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", received
    server.shutdown()


def test_langsmith_off_without_key(monkeypatch, chunks, fakes):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    assert tracing.langsmith_status().startswith("requested")
    _reset_langsmith_caches()
    pipeline = builder("langchain")(chunks, settings=SMALL, **fakes["langchain"])
    assert pipeline.reranked("memory limit")
    tracing.flush_langsmith()


def test_langsmith_emits_langchain_runs_to_the_configured_endpoint(monkeypatch, chunks, fakes, capture_server):
    url, received = capture_server
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_fake_offline_key")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", url)
    monkeypatch.setenv("LANGSMITH_PROJECT", "ragbench-test")
    _reset_langsmith_caches()
    assert tracing.langsmith_status() == "on"

    pipeline = builder("langchain")(chunks, settings=SMALL, **fakes["langchain"])
    pipeline.reranked("container OOMKilled memory limit")
    tracing.flush_langsmith()
    _reset_langsmith_caches()

    posted = b"".join(body for method, path, body in received if method == "POST" and "/runs" in path)
    names = set(re.findall(rb'"name":"([^"]+)"', posted))
    assert {b"langchain", b"EnsembleRetriever", b"VectorStoreRetriever", b"BM25Retriever"} <= names


def test_langfuse_is_skipped_without_keys():
    assert tracing.langfuse_configured() is False
    assert tracing.langfuse_client() is None
