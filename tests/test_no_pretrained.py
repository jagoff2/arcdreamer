from pathlib import Path
import ast

import torch

from src.model import RecurrentLatentModel


def test_source_does_not_use_external_model_loaders_or_apis() -> None:
    forbidden = {
        "from_" + "pretrained",
        "transformers",
        "huggingface" + "_hub",
        "Auto" + "Model",
        "Auto" + "Tokenizer",
        ".".join(["torch", "hub"]),
        "hf_" + "hub_download",
        "load_state_dict" + "_from_url",
        "sentencepiece",
        "tiktoken",
        "open" + "ai",
        "anthropic",
        "google.generativeai",
        "cohere",
    }
    for root in [Path("src")]:
        paths = list(root.rglob("*.py"))
        assert paths
        for path in paths:
            text = path.read_text(encoding="utf-8")
            lowered = text.lower()
            for item in forbidden:
                assert item.lower() not in lowered, f"{item} found in {path}"


def test_source_imports_do_not_reference_pretrained_ecosystems() -> None:
    forbidden_roots = {
        "transformers",
        "huggingface_hub",
        "tokenizers",
        "sentencepiece",
        "tiktoken",
        "openai",
        "anthropic",
        "cohere",
    }
    for path in Path("src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif node.module:
                    names = [node.module]
                for name in names:
                    root = name.split(".")[0]
                    assert root not in forbidden_roots, f"{name} imported in {path}"


def test_no_downloaded_checkpoints_are_committed() -> None:
    blocked_parts = {".git", "runs", "__pycache__", ".pytest_cache", ".venv", "venv", "env"}
    allowed_local_checkpoints = {
        Path("frozen/recurrent_latent_fast.pt"),
        Path("frozen/external_base_v1.pt"),
    }
    for path in Path(".").rglob("*"):
        if not path.is_file():
            continue
        if any(part in blocked_parts for part in path.parts):
            continue
        if path in allowed_local_checkpoints:
            continue
        assert path.suffix.lower() not in {".pt", ".pth", ".ckpt", ".safetensors", ".bin"}


def test_no_external_weight_download_helpers_in_tests_or_config() -> None:
    forbidden = {".".join(["torch", "hub"]), "hf_" + "hub_download", "load_state_dict" + "_from_url"}
    for root in [Path("tests")]:
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8").lower()
            for item in forbidden:
                assert item.lower() not in text, f"{item} found in {path}"


        text = path.read_text(encoding="utf-8")
        for item in forbidden:
            assert item not in text, f"{item} found in {path}"


def test_parameters_are_local_random_initialization() -> None:
    torch.manual_seed(1)
    first = RecurrentLatentModel()
    torch.manual_seed(2)
    second = RecurrentLatentModel()
    first_params = torch.cat([param.detach().flatten() for param in first.parameters()])
    second_params = torch.cat([param.detach().flatten() for param in second.parameters()])
    assert first_params.numel() > 0
    assert not torch.allclose(first_params, second_params)
