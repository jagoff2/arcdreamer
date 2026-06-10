from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any


def _external_patterns() -> list[str]:
    return [
        "from_" + "pretrained",
        "transform" + "ers",
        "huggingface" + "_hub",
        "Auto" + "Model",
        "Auto" + "Tokenizer",
        ".".join(["torch", "hub"]),
        "hf_" + "hub_download",
        "load_state_dict" + "_from_url",
        "sentence" + "piece",
        "tik" + "token",
        "open" + "ai",
        "anth" + "ropic",
        "google.generative" + "ai",
        "co" + "here",
        "ll" + "ama",
        "mistr" + "al",
        "qw" + "en",
        "be" + "rt",
        "t5",
    ]


def _forbidden_import_roots() -> set[str]:
    return {
        "transform" + "ers",
        "huggingface_hub",
        "token" + "izers",
        "sentence" + "piece",
        "tik" + "token",
        "open" + "ai",
        "anth" + "ropic",
        "co" + "here",
    }


def _iter_python_files() -> list[Path]:
    files: list[Path] = []
    for root in [Path("src"), Path("tests"), Path("audit")]:
        if root.exists():
            files.extend(sorted(root.rglob("*.py")))
    return files


def _import_findings(path: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return [{"path": str(path), "line": exc.lineno, "match": "syntax_error", "kind": "parse"}]
    forbidden_roots = _forbidden_import_roots()
    for node in ast.walk(tree):
        names: list[str] = []
        line = getattr(node, "lineno", None)
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            root = name.split(".")[0]
            if root in forbidden_roots:
                findings.append({"path": str(path), "line": line, "match": name, "kind": "forbidden_import"})
    return findings


def _text_findings(path: Path) -> list[dict[str, Any]]:
    if path == Path(__file__).relative_to(Path.cwd()):
        return []
    if not path.parts or path.parts[0] != "src":
        return []
    text = path.read_text(encoding="utf-8", errors="ignore")
    lowered = text.lower()
    findings: list[dict[str, Any]] = []
    for pattern in _external_patterns():
        needle = pattern.lower()
        if needle in lowered:
            findings.append({"path": str(path), "line": None, "match": pattern, "kind": "external_pattern"})
    return findings


def _checkpoint_findings() -> list[dict[str, Any]]:
    allowed = {Path("frozen/recurrent_latent_fast.pt")}
    ignored_parts = {".git", "__pycache__", ".pytest_cache", ".venv", "venv", "env", "runs"}
    suffixes = {".pt", ".pth", ".ckpt", ".safetensors", ".bin"}
    findings: list[dict[str, Any]] = []
    for path in Path(".").rglob("*"):
        if not path.is_file():
            continue
        if any(part in ignored_parts for part in path.parts):
            continue
        normalized = Path(*path.parts)
        if normalized in allowed:
            continue
        if path.suffix.lower() in suffixes:
            findings.append({"path": str(path), "line": None, "match": path.suffix, "kind": "unexpected_weight_file"})
    return findings


def _prompt_loop_findings() -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    runtime = Path("src/run_unbroken.py")
    if not runtime.exists():
        return [{"path": str(runtime), "line": None, "match": "missing_runtime", "kind": "prompt_loop"}]
    text = runtime.read_text(encoding="utf-8")
    required = ["model.step(observation, z)", "world.step(action)", "private_token = output[\"private_logits\"].argmax"]
    for item in required:
        if item not in text:
            findings.append({"path": str(runtime), "line": None, "match": item, "kind": "runtime_missing_expected_loop"})
    blocked = ["world.step(language", "lang_in = language", "prompt", "transcript"]
    lowered = text.lower()
    for item in blocked:
        if item.lower() in lowered:
            findings.append({"path": str(runtime), "line": None, "match": item, "kind": "prompt_loop_pattern"})
    return findings


def run_scan() -> dict[str, Any]:
    python_files = _iter_python_files()
    findings: list[dict[str, Any]] = []
    for path in python_files:
        findings.extend(_import_findings(path))
        findings.extend(_text_findings(path))
    findings.extend(_checkpoint_findings())
    findings.extend(_prompt_loop_findings())
    return {
        "passes": len(findings) == 0,
        "python_files_scanned": [str(path) for path in python_files],
        "findings": findings,
        "checks": {
            "no_external_or_pretrained_imports": not any(item["kind"] == "forbidden_import" for item in findings),
            "no_external_or_pretrained_text_patterns": not any(item["kind"] == "external_pattern" for item in findings),
            "no_unexpected_weight_files": not any(item["kind"] == "unexpected_weight_file" for item in findings),
            "runtime_not_prompt_loop": not any("prompt" in item["kind"] for item in findings),
        },
    }


def main() -> None:
    report = run_scan()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
