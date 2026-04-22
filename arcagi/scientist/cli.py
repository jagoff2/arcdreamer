"""Command-line entry points for the scientist patch."""

from __future__ import annotations

import argparse
import json

from . import HiddenRuleGridEnv, ScientistAgent, SyntheticConfig, run_episode


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the ARC-AGI-3 scientist agent on the synthetic hidden-rule smoke test.")
    parser.add_argument("--size", type=int, default=7, help="Synthetic grid size.")
    parser.add_argument("--max-steps", type=int, default=80, help="Maximum episode steps.")
    parser.add_argument("--seed", type=int, default=7, help="Environment and agent seed.")
    parser.add_argument("--no-key", action="store_true", help="Disable the hidden key requirement.")
    args = parser.parse_args(argv)

    env = HiddenRuleGridEnv(
        SyntheticConfig(
            size=args.size,
            requires_key=not args.no_key,
            seed=args.seed,
            max_steps=args.max_steps,
        )
    )
    agent = ScientistAgent()
    result = run_episode(env, agent, max_steps=args.max_steps, seed=args.seed)
    print(
        json.dumps(
            {
                "steps": result.steps,
                "total_reward": result.total_reward,
                "terminated": result.terminated,
                "final_info": dict(result.final_info),
                "diagnostics": result.diagnostics,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
