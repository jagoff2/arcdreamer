from __future__ import annotations

import argparse
import json

from arcagi.envs.arc_adapter import arc_toolkit_available, list_arc_games


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arcagi")
    subparsers = parser.add_subparsers(dest="command", required=False)
    subparsers.add_parser("status", help="Print a short project status summary.")

    train_parser = subparsers.add_parser("train-synthetic", help="Train the synthetic world model and language stack.")
    train_parser.add_argument("--epochs", type=int, default=8)
    train_parser.add_argument("--episodes-per-epoch", type=int, default=96)
    train_parser.add_argument("--checkpoint-path", type=str, default="artifacts/synthetic_hybrid.pt")
    train_parser.add_argument("--init-checkpoint-path", type=str, default="")
    train_parser.add_argument("--seed", type=int, default=7)
    train_parser.add_argument("--curriculum", type=str, default="staged")

    eval_parser = subparsers.add_parser("eval-synthetic", help="Evaluate an agent on the synthetic adaptation benchmark.")
    eval_parser.add_argument("--agent", type=str, default="graph")
    eval_parser.add_argument("--checkpoint-path", type=str, default="")
    eval_parser.add_argument("--episodes-per-family", type=int, default=3)
    eval_parser.add_argument("--seed", type=int, default=17)

    subparsers.add_parser("list-arc-games", help="List available ARC toolkit games when the optional SDK is installed.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command in (None, "status"):
        print("arcagi: scaffold initialized, core modules in progress")
        return 0
    if args.command == "train-synthetic":
        from arcagi.training.synthetic import SyntheticTrainingConfig, train_synthetic

        metrics = train_synthetic(
            SyntheticTrainingConfig(
                epochs=args.epochs,
                episodes_per_epoch=args.episodes_per_epoch,
                checkpoint_path=args.checkpoint_path,
                init_checkpoint_path=args.init_checkpoint_path,
                seed=args.seed,
                curriculum=args.curriculum,
            )
        )
        print(json.dumps(metrics, indent=2))
        return 0
    if args.command == "eval-synthetic":
        from arcagi.evaluation.harness import evaluate_synthetic

        result = evaluate_synthetic(
            agent_name=args.agent,
            checkpoint_path=args.checkpoint_path or None,
            episodes_per_family=args.episodes_per_family,
            seed=args.seed,
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "list-arc-games":
        result = {
            "available": arc_toolkit_available(),
            "games": list_arc_games(),
        }
        print(json.dumps(result, indent=2))
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2
