from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path
from typing import Any


_RULE_ID = re.compile(r"^[a-z]{2,6}-[a-f0-9]{6,8}$")
_NAMED_ID = re.compile(r"^(game|level|puzzle|task|fixture)[-_]?[a-z0-9][a-z0-9_-]*$", re.IGNORECASE)
_HEX_SUFFIX = re.compile(r"[-_]([0-9a-f]{6,10})$", re.IGNORECASE)


NAME_TOKENS_POLICY = (
    "policy",
    "solution",
    "answer",
    "solver",
    "lookup",
    "script",
    "sequence",
    "action",
)

NAME_TOKENS_ID = ("game", "puzzle", "level", "task", "fixture")
NAME_TOKENS_ID_VAR = ("game_id", "puzzle_id", "level_id", "task_id", "fixture_id", "level")
NAME_TOKENS_ALLOW_GENERAL = ("vocab", "grammar", "metric", "metrics", "curriculum", "test", "tests")

DB_HINT_TOKENS = ("symbolic", "solver", "solution", "answer", "policy", "lookup", "knowledge", "database", "db")
DB_PATH_HINTS = ("_symbolic_", "_solver_", "_solution_", "_answer_", "_policy_", "_lookup_", "_db", "knowledge", "solver")
DB_EXTS = {".json", ".jsonl", ".yaml", ".yml", ".pkl", ".npz", ".npy", ".pt", ".pth", ".safetensors", ".csv", ".tsv"}


def _name_has_token(name: str, tokens: tuple[str, ...]) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in tokens)


def _looks_like_id_key(value: str) -> bool:
    normalized = value.lower()
    return bool(_RULE_ID.fullmatch(normalized) or _NAMED_ID.fullmatch(normalized) or _HEX_SUFFIX.search(normalized))


def _is_action_sequence(value: ast.AST) -> bool:
    return isinstance(value, (ast.List, ast.Tuple)) and value.elts and all(_is_scalar_constant(item) for item in value.elts)


def _is_scalar_constant(node: ast.AST) -> bool:
    return isinstance(node, (ast.Constant,)) and not isinstance(node.value, (dict, list, tuple, set))


def _is_name(node: ast.AST) -> str | None:
    return node.id if isinstance(node, ast.Name) else None


def _assignment_targets(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        targets = [node.target]
    else:
        return []
    out: list[str] = []
    for target in targets:
        name = _is_name(target)
        if name:
            out.append(name)
    return out


def _assigned_names(node: ast.Assign | ast.AnnAssign) -> list[str]:
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    else:
        targets = [node.target]
    names: list[str] = []
    for target in targets:
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            names.extend(item.id for item in target.elts if isinstance(item, ast.Name))
    return names


def _dict_values_keyed_by_level_ids(node: ast.Dict) -> bool:
    if not node.keys:
        return False
    keys = [key for key in node.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)]
    if not keys:
        return False
    return any(_looks_like_id_key(str(key.value)) for key in keys)


def _is_general_vocab_name(name: str) -> bool:
    return _name_has_token(name, NAME_TOKENS_ALLOW_GENERAL)


def _is_action_return(block: list[ast.stmt]) -> bool:
    for stmt in block:
        for nested in ast.walk(stmt):
            if isinstance(nested, ast.Return):
                if isinstance(nested.value, (ast.List, ast.Tuple)):
                    return True
                if isinstance(nested.value, ast.Call) and _name_has_token(ast.unparse(nested.value.func), NAME_TOKENS_POLICY):
                    return True
            if isinstance(nested, (ast.Assign, ast.AnnAssign)):
                names = _assigned_names(nested)
                for name in names:
                    if _name_has_token(name, NAME_TOKENS_POLICY):
                        return True
    return False


def _is_target_id_variable(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return _name_has_token(node.id, NAME_TOKENS_ID_VAR)
    return False


def _slice_is_id_key(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return _looks_like_id_key(node.value)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return any(
            isinstance(item, ast.Constant)
            and isinstance(item.value, str)
            and _looks_like_id_key(item.value)
            for item in node.elts
        )
    if isinstance(node, ast.Name):
        return _name_has_token(node.id, NAME_TOKENS_ID_VAR)
    return False


def _classify_dict_map(name: str, node: ast.Dict) -> str | None:
    if _is_general_vocab_name(name):
        return None

    has_id_keys = _dict_values_keyed_by_level_ids(node)
    if not has_id_keys:
        return None

    if _name_has_token(name, ("solution", "answer")):
        return "hardcoded_solution_map"
    if _name_has_token(name, ("solver", "lookup")):
        return "solver_lookup_table"
    if _name_has_token(name, ("policy", "action", "sequence")) and any(_is_action_sequence(value) for value in node.values if value):
        return "action_sequence_by_id"
    return None


def _collect_compare_id_literals(node: ast.Compare) -> bool:
    if not _is_target_id_variable(node.left):
        return False
    operations = (ast.Eq, ast.NotEq, ast.In, ast.NotIn)
    for op, comparator in zip(node.ops, node.comparators):
        if not isinstance(op, operations):
            return False
        if isinstance(op, (ast.Eq, ast.NotEq)):
            if not (
                isinstance(comparator, ast.Constant)
                and isinstance(comparator.value, str)
                and _looks_like_id_key(comparator.value)
            ):
                return False
        elif isinstance(comparator, (ast.List, ast.Tuple, ast.Set)):
            if not any(
                isinstance(item, ast.Constant) and isinstance(item.value, str) and _looks_like_id_key(item.value)
                for item in comparator.elts
            ):
                return False
        else:
            return False
    return True


def _is_id_control_node(node: ast.If) -> bool:
    tests: list[ast.AST] = [node.test]
    for test in tests:
        if isinstance(test, ast.BoolOp):
            tests.extend(test.values)
        if isinstance(test, ast.IfExp):
            tests.append(test.test)
        if isinstance(test, ast.Compare) and _collect_compare_id_literals(test):
            if _is_action_return(node.body):
                return True
    return _scan_nested_id_control(node)


def _scan_nested_id_control(node: ast.If) -> bool:
    for test in ast.walk(node.test):
        if isinstance(test, ast.Compare) and _collect_compare_id_literals(test) and _is_action_return(node.body):
            return True
    for handler in node.orelse:
        if isinstance(handler, ast.If):
            if _is_id_control_node(handler):
                return True
    return False


def _extract_path_constant(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if not isinstance(node, ast.Call):
        return None
    if isinstance(node.func, ast.Name) and node.func.id in {"open", "Path"}:
        return _extract_path_constant(node.args[0]) if node.args else None
    if isinstance(node.func, ast.Attribute) and node.func.attr in {"read_text", "read_bytes"}:
        return _extract_path_constant(node.func.value)
    if isinstance(node.func, ast.Attribute) and node.func.attr == "open":
        return _extract_path_constant(node.func.value)
    return None


def _is_loader_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Name):
        return node.func.id in {"load", "loads"}
    if isinstance(node.func, ast.Attribute):
        return node.func.attr in {"load", "loads"}
    return False


def _looks_like_symbolic_db(path: str, var_name: str) -> bool:
    lower = path.lower()
    extension_ok = Path(lower).suffix in DB_EXTS
    if not extension_ok:
        return False
    if any(token in lower for token in ("report", "manifest", "fixture", "fixtures", "trace", "traces", "game", "games")):
        return False
    if not _name_has_token(var_name, tuple(DB_HINT_TOKENS)):
        return False
    return any(token in lower for token in DB_PATH_HINTS)


def _scan_file(path: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return [
            {
                "path": str(path),
                "line": exc.lineno or 0,
                "kind": "parse_error",
                "name": path.name,
                "match": "syntax_error",
            }
        ]

    id_maps: set[str] = set()
    for node in ast.walk(tree):
        for target in _assignment_targets(node):
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Dict):
                kind = _classify_dict_map(target, node.value)
                if kind:
                    findings.append(
                        {
                            "path": str(path),
                            "line": int(getattr(node, "lineno", 0) or 0),
                            "kind": kind,
                            "name": target,
                            "match": _looks_like_id_key(str(target)) if _looks_like_id_key(target) else "id_keyed_map",
                        }
                    )
                    id_maps.add(target)

            if isinstance(node.value, ast.Call):
                path_literal = _extract_path_constant(node.value)
                if path_literal and _looks_like_symbolic_db(path_literal, target):
                    findings.append(
                        {
                            "path": str(path),
                            "line": int(getattr(node, "lineno", 0) or 0),
                            "kind": "external_symbolic_memory_db",
                            "name": target,
                            "db_path": path_literal,
                        }
                    )
                if _is_loader_call(node.value):
                    path_literal = None
                    for arg in node.value.args:
                        path_literal = _extract_path_constant(arg)
                        if path_literal:
                            break
                    if path_literal and _looks_like_symbolic_db(path_literal, target):
                        findings.append(
                            {
                                "path": str(path),
                                "line": int(getattr(node, "lineno", 0) or 0),
                                "kind": "external_symbolic_memory_db",
                                "name": target,
                                "db_path": path_literal,
                            }
                        )

    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            if _is_id_control_node(node):
                findings.append(
                    {
                        "path": str(path),
                        "line": int(getattr(node, "lineno", 0) or 0),
                        "kind": "scripted_policy_by_id",
                        "match": "id_branch_control",
                    }
                )

        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            map_name = node.value.id
            if map_name in id_maps and _slice_is_id_key(node.slice):
                findings.append(
                    {
                        "path": str(path),
                        "line": int(getattr(node, "lineno", 0) or 0),
                        "kind": "solver_lookup_table",
                        "name": map_name,
                    }
                )

    return findings


def scan_file_for_hardcoding(path: str | Path) -> list[dict[str, Any]]:
    return _scan_file(Path(path))


def run_scan(source_root: str | Path = Path("src")) -> dict[str, Any]:
    root = Path(source_root)
    findings: list[dict[str, Any]] = []
    scanned: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if not path.is_file():
            continue
        scanned.append(str(path))
        findings.extend(_scan_file(path))

    return {
        "passes": not findings,
        "files_scanned": scanned,
        "findings": findings,
        "counts": dict(Counter(item["kind"] for item in findings)),
    }


if __name__ == "__main__":
    print(run_scan())
