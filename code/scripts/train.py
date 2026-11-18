#!/usr/bin/env python
"""Train one baseline cell.

Example:
    python scripts/train.py --config configs/energy_two_stage.yaml
    python scripts/train.py --config configs/residual_joint.yaml --seed 1 --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plate_pinn.config import apply_smoke_overrides, load_config  # noqa: E402
from plate_pinn.train import Trainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--seed", type=int, default=None, help="Override the seed.")
    parser.add_argument("--output", default=None, help="Override the output directory.")
    parser.add_argument("--device", default=None, help="cuda, cuda:0 or cpu.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use tiny data and step counts to verify the code path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.device is not None:
        cfg["device"] = args.device
    if args.smoke:
        cfg = apply_smoke_overrides(cfg)

    output = args.output or cfg["output_dir"]
    trainer = Trainer(cfg, output)
    metrics = trainer.train()

    print(json.dumps(metrics, indent=2))
    print(f"\nResults written to: {Path(output).resolve()}")


if __name__ == "__main__":
    main()
