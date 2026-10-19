from __future__ import annotations

import importlib.metadata as md
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from .corpus import REPO_ROOT

PACKAGE_DIR = Path(__file__).resolve().parent
IMPL_FILES = {
    "baseline": "impl_baseline.py",
    "langchain": "impl_langchain.py",
    "llamaindex": "impl_llamaindex.py",
}
SHARED_FILES = ["shared.py"]


def lines_of_code(path: Path) -> int:
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def code_size() -> dict[str, int]:
    sizes = {name: lines_of_code(PACKAGE_DIR / file) for name, file in IMPL_FILES.items()}
    sizes["shared"] = sum(lines_of_code(PACKAGE_DIR / f) for f in SHARED_FILES)
    return sizes


def _roots(extra: str) -> list[str]:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]
    return [Requirement(r).name for r in project["optional-dependencies"][extra]]


def closure(roots: list[str]) -> set[str]:
    seen: set[str] = set()
    stack = [canonicalize_name(r) for r in roots]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        try:
            requires = md.requires(name) or []
        except md.PackageNotFoundError:
            continue
        seen.add(name)
        for raw in requires:
            req = Requirement(raw)
            if req.marker and not req.marker.evaluate({"extra": ""}):
                continue
            stack.append(canonicalize_name(req.name))
    return seen


def installed_bytes(name: str) -> int:
    files = md.distribution(name).files or []
    total = 0
    for f in files:
        path = Path(f.locate())
        if path.is_file():
            total += path.stat().st_size
    return total


def dependency_footprint() -> dict[str, dict[str, float]]:
    base = closure(_roots("baseline"))
    out: dict[str, dict[str, float]] = {}
    for impl in IMPL_FILES:
        packages = closure(_roots(impl))
        extra = packages - base
        out[impl] = {
            "packages": len(packages),
            "install_mb": round(sum(installed_bytes(p) for p in packages) / 1e6, 1),
            "packages_beyond_baseline": len(extra),
            "mb_beyond_baseline": round(sum(installed_bytes(p) for p in extra) / 1e6, 1),
            "beyond_baseline": sorted(extra),
        }
    return out
