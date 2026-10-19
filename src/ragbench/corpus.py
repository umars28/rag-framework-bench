from __future__ import annotations

import gzip
import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
CORPUS_FILE = DATA_DIR / "corpus" / "documents.jsonl.gz"
GOLDEN_FILE = DATA_DIR / "golden.yaml"

SOURCE_REPO = "https://github.com/kubernetes/website"
SOURCE_LICENCE = "CC BY 4.0"
SOURCE_PATHS = ["content/en/docs/tasks", "content/en/docs/concepts"]
DOC_URL_BASE = "https://kubernetes.io/docs"
PATH_PREFIX = "content/en/docs/"

_SHORTCODE = re.compile(r"\{\{[%<].*?[%>]\}\}", re.DOTALL)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_CAPTION = re.compile(r"^\s*\{\{.*$", re.M)
_MULTI_BLANK = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    url: str
    path: str
    content: str


@dataclass(frozen=True)
class Query:
    id: str
    text: str
    facet: str
    relevant: dict[str, int] = field(default_factory=dict)


class LabelError(ValueError):
    pass


def _clean(text: str) -> str:
    text = _HTML_COMMENT.sub("", text)
    text = _SHORTCODE.sub("", text)
    text = _CAPTION.sub("", text)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()


def _doc_url(relative: Path) -> str:
    slug = relative.with_suffix("").as_posix()
    slug = slug.removeprefix(PATH_PREFIX).removesuffix("/_index")
    return f"{DOC_URL_BASE}/{slug}/"


def fetch_source(target: Path, ref: str) -> str:
    target = Path(target)
    git = ["git", "-C", str(target)]
    if not (target / ".git").exists():
        target.mkdir(parents=True, exist_ok=True)
        subprocess.run([*git, "init", "-q"], check=True)
        subprocess.run([*git, "remote", "add", "origin", SOURCE_REPO], check=True)
        subprocess.run([*git, "sparse-checkout", "set", *SOURCE_PATHS], check=True)
    subprocess.run(
        [*git, "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", ref], check=True
    )
    subprocess.run([*git, "checkout", "-q", "--force", "FETCH_HEAD"], check=True)
    out = subprocess.run([*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    return out.stdout.strip()


def load_documents(source_root: Path) -> list[Document]:
    import frontmatter

    root = Path(source_root)
    docs: list[Document] = []
    for sub in SOURCE_PATHS:
        for path in sorted((root / sub).rglob("*.md")):
            post = frontmatter.load(path)
            title = str(post.get("title") or "").strip()
            body = _clean(post.content)
            if not title or len(body) < 400:
                continue
            relative = path.relative_to(root)
            docs.append(
                Document(
                    doc_id=hashlib.sha1(relative.as_posix().encode()).hexdigest()[:12],
                    title=title,
                    url=_doc_url(relative),
                    path=relative.as_posix(),
                    content=body,
                )
            )
    return sorted(docs, key=lambda d: d.path)


def save_documents(docs: list[Document], commit: str, target: Path = CORPUS_FILE) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(asdict(d), sort_keys=True) + "\n" for d in docs)
    with gzip.GzipFile(target, "wb", mtime=0) as fh:
        fh.write(payload.encode())
    (target.parent / "SOURCE.txt").write_text(
        f"source: {SOURCE_REPO}\ncommit: {commit}\npaths: {' '.join(SOURCE_PATHS)}\n"
        f"licence: {SOURCE_LICENCE}\ndocuments: {len(docs)}\n"
        f"characters: {sum(len(d.content) for d in docs)}\n"
    )
    return target


def load_corpus(path: Path = CORPUS_FILE) -> list[Document]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found, run `ragbench fetch` first")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        return [Document(**json.loads(line)) for line in fh if line.strip()]


def load_golden(docs: list[Document], path: Path = GOLDEN_FILE) -> list[Query]:
    raw = yaml.safe_load(Path(path).read_text())
    by_path = {d.path.removeprefix(PATH_PREFIX): d.doc_id for d in docs}
    queries: list[Query] = []
    missing: list[str] = []
    for entry in raw["queries"]:
        resolved: dict[str, int] = {}
        for doc_path, grade in (entry.get("relevant") or {}).items():
            if doc_path not in by_path:
                missing.append(f"{entry['id']} -> {doc_path}")
                continue
            resolved[by_path[doc_path]] = int(grade)
        queries.append(Query(entry["id"], entry["query"], entry.get("facet", "general"), resolved))
    if missing:
        raise LabelError("labels reference documents not in the corpus:\n  " + "\n  ".join(missing))
    return queries
