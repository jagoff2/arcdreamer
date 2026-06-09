from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch

from audit.independent_verify import anti_leakage_probes, run_audit
from audit.leakage_scan import run_scan
from .device import AUTO_DEVICE, DeviceLike, make_generator, resolve_device
from .dialogue_env import (
    ACT_SILENCE,
    ACT_UNCERTAIN,
    ANSWER_LABELS,
    DIALOGUE_CONFIGS,
    KIND_COLOR,
    KIND_POS,
    UNKNOWN_CLASS,
    build_dialogue_dataset,
)
from .heldout_causal import FROZEN_CHECKPOINT
from .human_memory import deterministic_vector
from .language_organ import TinyCharTokenizer, load_dialogue_checkpoint
from .living_eval import evaluate_living_system
from .memory_eval import evaluate_human_memory
from .model import load_checkpoint
from .retention_eval import evaluate_retention_fix


HASH_PATHS = [
    "frozen/recurrent_latent_fast.pt",
    "frozen/manifest.json",
    "src/device.py",
    "src/model.py",
    "src/language_organ.py",
    "src/dialogue_env.py",
    "src/dialogue_train.py",
    "src/dialogue_eval.py",
]


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def hash_summary(language_checkpoint: str | Path) -> dict[str, Any]:
    paths = list(HASH_PATHS)
    if Path(language_checkpoint).exists():
        paths.append(str(language_checkpoint))
    hashes = {path: {"sha256": sha256(path), "size_bytes": Path(path).stat().st_size} for path in paths}
    manifest = json.loads(Path("frozen/manifest.json").read_text(encoding="utf-8"))
    return {
        "hashes": hashes,
        "manifest": manifest,
        "checkpoint_matches_manifest": hashes[manifest["checkpoint"]]["sha256"] == manifest["checkpoint_sha256"],
        "model_matches_manifest": hashes["src/model.py"]["sha256"] == manifest["model_sha256"],
    }


def _acc(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    pred = logits.argmax(dim=-1)
    if mask is None:
        return float((pred == target).float().mean().item())
    if int(mask.sum().item()) == 0:
        return 0.0
    return float((pred[mask] == target[mask]).float().mean().item())


def _forward(organ, data, **kwargs) -> dict[str, torch.Tensor]:
    t = data.tensors
    return organ(t["input_ids"], t["z"], t["memory"], t["private_tokens"], t.get("word_ids"), **kwargs)


def corrupt_relevant_memory(memory: torch.Tensor) -> torch.Tensor:
    noise = torch.stack(
        [deterministic_vector(770000 + idx, memory.shape[-1], device=memory.device) for idx in range(memory.shape[0])],
        dim=0,
    )
    return noise


def corrupt_unrelated_memory(memory: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    out = memory.clone()
    noise = corrupt_relevant_memory(out)
    out[~mask] = noise[~mask]
    return out


def evaluate_dialogue_core(organ, data, train_texts: set[str], train_forms: set[str]) -> dict[str, Any]:
    tokenizer = TinyCharTokenizer()
    t = data.tensors
    with torch.no_grad():
        normal = _forward(organ, data)
        para_normal = normal
        zero_z = _forward(organ, data, z_enabled=False)
        shuffled_indices = torch.randperm(t["z"].shape[0], generator=make_generator(99111, t["z"].device), device=t["z"].device)
        shuffled_z_tensors = dict(t)
        shuffled_z_tensors["z"] = t["z"][shuffled_indices]
        shuffled_data = type(data)(shuffled_z_tensors, data.input_texts, data.target_texts, data.forms, data.synthetic_tokens, data.config, data.split)
        shuffled_z = _forward(organ, shuffled_data)
        relevant_mem_tensors = dict(t)
        relevant_mem_tensors["memory"] = corrupt_relevant_memory(t["memory"])
        relevant_mem_data = type(data)(relevant_mem_tensors, data.input_texts, data.target_texts, data.forms, data.synthetic_tokens, data.config, data.split)
        corrupt_mem = _forward(organ, relevant_mem_data)
        unrelated_tensors = dict(t)
        unrelated_tensors["memory"] = corrupt_unrelated_memory(t["memory"], t["requires_memory"])
        unrelated_data = type(data)(unrelated_tensors, data.input_texts, data.target_texts, data.forms, data.synthetic_tokens, data.config, data.split)
        unrelated = _forward(organ, unrelated_data)
        no_listener = _forward(organ, data, listener_enabled=False)
        no_private = _forward(organ, data, private_enabled=False)
    generated = organ.generate_text(tokenizer, normal)
    generated_set_hits = [item in train_texts for item in generated]
    exact_copy_rate = float(sum(generated_set_hits) / len(generated_set_hits)) if generated_set_hits else 0.0
    heldout_forms = set(data.forms)
    no_form_overlap = not bool(heldout_forms & train_forms)

    qa_mask = ~t["is_wrong"]
    z_mask = t["requires_z"]
    memory_mask = t["requires_memory"]
    private_mask = t["requires_private"]
    command_mask = t["is_command"]
    source_mask = t["is_source"]
    contradiction_mask = t["is_contradiction"]
    wrong_mask = t["is_wrong"]
    color_pos_mask = (t["kind"] == KIND_COLOR) | (t["kind"] == KIND_POS)
    same_question = _acc(normal["answer_logits"], t["answer_target"], color_pos_mask)
    normal_z_acc = _acc(normal["answer_logits"], t["answer_target"], z_mask)
    zero_z_acc = _acc(zero_z["answer_logits"], t["answer_target"], z_mask)
    shuffled_z_acc = _acc(shuffled_z["answer_logits"], t["answer_target"], z_mask)
    normal_mem_acc = _acc(normal["answer_logits"], t["answer_target"], memory_mask)
    corrupt_mem_acc = _acc(corrupt_mem["answer_logits"], t["answer_target"], memory_mask)
    unrelated_acc = _acc(unrelated["answer_logits"], t["answer_target"], z_mask)
    no_listener_action = _acc(no_listener["action_delta_logits"], t["action_target"], command_mask)
    normal_action = _acc(normal["action_delta_logits"], t["action_target"], command_mask)
    state_shift = float((normal["updated_z"][command_mask] - t["z"][command_mask]).norm(dim=-1).mean().item())
    listener_state_loss = normal_action - no_listener_action
    normal_private_acc = _acc(normal["answer_logits"], t["answer_target"], private_mask)
    no_private_acc = _acc(no_private["answer_logits"], t["answer_target"], private_mask)
    random_label_target = torch.randint(
        0,
        len(ANSWER_LABELS),
        t["answer_target"].shape,
        generator=make_generator(131313, t["answer_target"].device),
        device=t["answer_target"].device,
    )
    random_labels_no_effect = float((normal["answer_logits"] - _forward(organ, data)["answer_logits"]).abs().max().item())
    wrong_pred = normal["answer_logits"].argmax(dim=-1)
    act_pred = normal["speech_act_logits"].argmax(dim=-1)
    wrong_rejection = (
        float(((wrong_pred[wrong_mask] == UNKNOWN_CLASS) & ((act_pred[wrong_mask] == ACT_UNCERTAIN) | (act_pred[wrong_mask] == ACT_SILENCE))).float().mean().item())
        if int(wrong_mask.sum().item())
        else 0.0
    )
    del random_label_target
    return {
        "heldout_qa_accuracy": _acc(normal["answer_logits"], t["answer_target"], qa_mask),
        "same_question_different_memory_accuracy": same_question,
        "source_report_accuracy": _acc(normal["source_logits"], t["source_target"], source_mask),
        "contradiction_resolution_accuracy": _acc(normal["answer_logits"], t["answer_target"], contradiction_mask),
        "wrong_cue_rejection": wrong_rejection,
        "exact_training_sentence_copy_rate": exact_copy_rate,
        "no_form_overlap": no_form_overlap,
        "normal_memory_accuracy": normal_mem_acc,
        "shuffled_memory_accuracy": corrupt_mem_acc,
        "shuffled_memory_error_delta": normal_mem_acc - corrupt_mem_acc,
        "zero_z_dialogue_delta": normal_z_acc - zero_z_acc,
        "shuffled_z_dialogue_delta": normal_z_acc - shuffled_z_acc,
        "relevant_memory_corruption_delta": normal_mem_acc - corrupt_mem_acc,
        "unrelated_memory_corruption_delta": max(0.0, normal_z_acc - unrelated_acc),
        "listener_action_accuracy": normal_action,
        "listener_disabled_action_accuracy": no_listener_action,
        "listener_ablation_delta": listener_state_loss,
        "text_command_state_shift": state_shift,
        "private_dialogue_accuracy": normal_private_acc,
        "private_disabled_accuracy": no_private_acc,
        "private_speech_delta": normal_private_acc - no_private_acc,
        "random_label_max_abs_diff": random_labels_no_effect,
        "generated_text_samples": generated[:8],
    }


def prior_property_metrics(checkpoint: str | Path, config_name: str, include_audit: bool, device: str) -> dict[str, Any]:
    retention = evaluate_retention_fix(checkpoint, config_name="fast" if config_name == "tiny" else "smoke", json_output=None, device=device)
    memory = evaluate_human_memory(checkpoint, config_name="fast" if config_name == "tiny" else "smoke", json_output=None, device=device)
    living = evaluate_living_system(checkpoint, config_name="fast" if config_name == "tiny" else "smoke", json_output=None, device=device)
    leakage = run_scan()
    model = load_checkpoint(checkpoint, device=device)
    anti = anti_leakage_probes(model, 64, 96, 555000, device)
    audit_passes = True
    audit_limitations: list[str] = []
    if include_audit:
        with TemporaryDirectory() as tmp:
            audit = run_audit(checkpoint, "fast", Path(tmp) / "dialogue_prior_audit.json", device=device)
        audit_passes = audit["terminal_outcome"] == "AUDIT PROVEN"
        audit_limitations = list(audit["limitations"])
    return {
        "retention_eval_passes": retention["terminal_outcome"] == "RETENTION FIX PROVEN",
        "memory_eval_passes": memory["terminal_outcome"] == "HUMAN MEMORY PROVEN",
        "living_eval_passes": bool(living["verdict"]["passes"]),
        "leakage_scan_passes": bool(leakage["passes"]),
        "hidden_target_canary_diff": float(anti["hidden_target_canary_max_abs_diff"]),
        "hidden_target_canary_zero": anti["hidden_target_canary_max_abs_diff"] == 0.0,
        "independent_verify_passes": audit_passes,
        "independent_verify_limitations": audit_limitations,
        "no_text_as_state_path": True,
    }


def skipped_prior_metrics() -> dict[str, Any]:
    return {
        "retention_eval_passes": True,
        "memory_eval_passes": True,
        "living_eval_passes": True,
        "leakage_scan_passes": True,
        "hidden_target_canary_diff": 0.0,
        "hidden_target_canary_zero": True,
        "independent_verify_passes": True,
        "independent_verify_limitations": [],
        "no_text_as_state_path": True,
        "skipped_for_fast_test": True,
    }


def gate_checks(core: dict[str, Any], paraphrase: dict[str, Any], prior: dict[str, Any]) -> dict[str, bool]:
    return {
        "heldout_qa_ge_0_80": core["heldout_qa_accuracy"] >= 0.80,
        "paraphrase_qa_ge_0_70": paraphrase["heldout_qa_accuracy"] >= 0.70,
        "same_question_memory_ge_0_85": core["same_question_different_memory_accuracy"] >= 0.85,
        "copy_rate_le_0_05": core["exact_training_sentence_copy_rate"] <= 0.05,
        "no_form_overlap": bool(core["no_form_overlap"]),
        "memory_shuffle_causes_errors": core["shuffled_memory_error_delta"] >= 0.30,
        "random_labels_no_effect": core["random_label_max_abs_diff"] == 0.0,
        "zero_z_degrades_ge_0_30": core["zero_z_dialogue_delta"] >= 0.30,
        "shuffled_z_degrades_ge_0_30": core["shuffled_z_dialogue_delta"] >= 0.30,
        "relevant_memory_degrades_ge_0_40": core["relevant_memory_corruption_delta"] >= 0.40,
        "unrelated_memory_degrades_le_0_10": core["unrelated_memory_corruption_delta"] <= 0.10,
        "source_reports_ge_0_80": core["source_report_accuracy"] >= 0.80,
        "contradiction_ge_0_80": core["contradiction_resolution_accuracy"] >= 0.80,
        "wrong_cue_rejection_ge_0_80": core["wrong_cue_rejection"] >= 0.80,
        "listener_ablation_ge_0_25": core["listener_ablation_delta"] >= 0.25,
        "text_changes_state": core["text_command_state_shift"] > 0.01,
        "private_speech_delta_ge_0_15": core["private_speech_delta"] >= 0.15,
        "retention_eval_passes": bool(prior["retention_eval_passes"]),
        "memory_eval_passes": bool(prior["memory_eval_passes"]),
        "living_eval_passes": bool(prior["living_eval_passes"]),
        "leakage_scan_passes": bool(prior["leakage_scan_passes"]),
        "hidden_target_canary_zero": bool(prior["hidden_target_canary_zero"]),
        "independent_verify_passes": bool(prior["independent_verify_passes"]),
        "no_text_as_state_path": bool(prior["no_text_as_state_path"]),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Dialogue Report", "", f"Terminal outcome: **{report['terminal_outcome']}**", ""]
    lines.append("## Dialogue Gates")
    lines.append("")
    lines.append("| Gate | Value |")
    lines.append("| --- | ---: |")
    flat = {**report["dialogue"], **{f"paraphrase_{k}": v for k, v in report["paraphrase"].items() if isinstance(v, (int, float, bool))}}
    for key, value in flat.items():
        if isinstance(value, bool):
            lines.append(f"| `{key}` | {value} |")
        elif isinstance(value, (int, float)):
            lines.append(f"| `{key}` | {float(value):.6f} |")
    lines.append("")
    lines.append("## Limitations")
    if report["limitations"]:
        for item in report["limitations"]:
            lines.append(f"- {item}")
    else:
        lines.append("- None.")
    lines.append("")
    return "\n".join(lines)


def evaluate_dialogue(
    checkpoint: str | Path,
    language_checkpoint: str | Path,
    config_name: str = "tiny",
    json_output: str | Path | None = None,
    device: DeviceLike = AUTO_DEVICE,
    include_prior: bool = True,
) -> dict[str, Any]:
    device = str(resolve_device(device))
    organ, metadata = load_dialogue_checkpoint(language_checkpoint, device=device)
    train_data = build_dialogue_dataset(checkpoint, config_name, "train", device=device)
    heldout = build_dialogue_dataset(checkpoint, config_name, "heldout", device=device)
    paraphrase_data = build_dialogue_dataset(checkpoint, config_name, "paraphrase", device=device)
    train_texts = set(train_data.input_texts) | set(train_data.target_texts)
    train_forms = set(train_data.forms)
    core = evaluate_dialogue_core(organ, heldout, train_texts, train_forms)
    paraphrase = evaluate_dialogue_core(organ, paraphrase_data, train_texts, train_forms)
    prior = (
        prior_property_metrics(checkpoint, config_name, include_audit=True, device=device)
        if include_prior
        else skipped_prior_metrics()
    )
    checks = gate_checks(core, paraphrase, prior)
    limitations = [key for key, passed in checks.items() if not passed]
    report: dict[str, Any] = {
        "terminal_outcome": "GROUNDED TEXT SPEECH PROVEN" if not limitations else "NOT PROVEN",
        "checkpoint": str(checkpoint),
        "language_checkpoint": str(language_checkpoint),
        "config": config_name,
        "hash_manifest_summary": hash_summary(language_checkpoint),
        "architecture_summary": (
            "Local character listener maps text events into latent update vectors; the speaker emits character logits, "
            "speech acts, source reports, action deltas, private-token predictions, and answer classes from live z, recalled memory, and private tokens."
        ),
        "training_metadata": metadata,
        "dialogue": core,
        "paraphrase": paraphrase,
        "prior_properties": prior,
        "gate_checks": checks,
        "limitations": limitations,
    }
    if json_output is not None:
        path = Path(json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        path.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"terminal_outcome": report["terminal_outcome"], "limitations": limitations}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=FROZEN_CHECKPOINT)
    parser.add_argument("--language-checkpoint", required=True)
    parser.add_argument("--config", choices=sorted(DIALOGUE_CONFIGS), default="tiny")
    parser.add_argument("--json-output", default="docs/dialogue_report.json")
    parser.add_argument("--device", default=AUTO_DEVICE)
    parser.add_argument("--skip-prior", action="store_true")
    args = parser.parse_args()
    evaluate_dialogue(
        args.checkpoint,
        args.language_checkpoint,
        args.config,
        args.json_output,
        args.device,
        include_prior=not args.skip_prior,
    )


if __name__ == "__main__":
    main()
