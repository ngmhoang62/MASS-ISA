from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch

from src.config import DEFAULT_CONFIG_PATH, get_config, get_run_defaults
from src.evaluation import aggregate_seed_metrics
from src.training import train


def parse_args():
    parser = argparse.ArgumentParser(description="Run MASS experiments.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config.yaml.")
    parser.add_argument("--variant", nargs="+", help="One or more model variants from config.yaml.")
    parser.add_argument("--dataset", nargs="+", help="One or more datasets from config.yaml.")
    parser.add_argument("--seed", nargs="+", type=int, help="One or more random seeds.")
    parser.add_argument(
        "--ablation",
        nargs="+",
        help="One or more ablation/representation settings from config.yaml (default: full).",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print each resolved configuration without starting training.",
    )
    return parser.parse_args()


def resolve_runs(args):
    defaults = get_run_defaults(args.config)
    return (
        args.variant or [defaults["default_variant"]],
        args.dataset or [defaults["default_dataset"]],
        args.seed or [defaults["default_seed"]],
        args.ablation or [defaults.get("default_ablation", "full")],
    )


def main():
    args = parse_args()
    variants, datasets, seeds, ablations = resolve_runs(args)

    for variant, dataset, ablation in itertools.product(variants, datasets, ablations):
        seed_metrics = []
        aggregate_path = None
        for seed in seeds:
            config = get_config(
                args.config,
                variant=variant,
                dataset=dataset,
                seed=seed,
                ablation=ablation,
            )
            aggregate_path = config.aggregate_results_path
            if args.print_config:
                print(json.dumps(config.as_serializable_dict(), indent=2))
                continue
            seed_metrics.append(train(config))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if seed_metrics:
            aggregate = aggregate_seed_metrics(seed_metrics, aggregate_path)
            print(json.dumps(aggregate["metrics"], indent=2))


if __name__ == "__main__":
    main()
