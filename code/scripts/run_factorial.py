#!/usr/bin/env python
"""Run the full 2x2 factorial (four configs x seeds).

Example:
    python scripts/run_factorial.py --seeds 0 1 2
    python scripts/run_factorial.py --seeds 0 --smoke
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [
    "energy_two_stage",
    "energy_joint",
    "residual_two_stage",
    "residual_joint",
]
STABILIZED_CONFIGS = [
    "energy_two_stage_stabilized",
    "energy_joint_stabilized",
    "residual_two_stage",
    "residual_joint",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--stabilized",
        action="store_true",
        help="Use the mixed energy + residual configs for the energy cells.",
    )
    parser.add_argument("--runs-dir", default="runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    configs = STABILIZED_CONFIGS if args.stabilized else CONFIGS
    commands: list[list[str]] = []

    for name in configs:
        for seed in args.seeds:
            output = runs_dir / f"{name}_seed{seed}"
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "train.py"),
                "--config",
                str(ROOT / "configs" / f"{name}.yaml"),
                "--seed",
                str(seed),
                "--output",
                str(output),
            ]
            if args.smoke:
                cmd.append("--smoke")
            commands.append(cmd)

    if args.dry_run:
        for cmd in commands:
            print(" ".join(cmd))
        return

    for cmd in commands:
        print("\n>>>", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    print("\n=== Summary ===")
    print(
        f"{'cell':<22}{'seed':>5}{'rel_T1':>10}{'rel_T1c':>10}"
        f"{'time_s':>10}{'t_target':>10}"
    )
    for name in configs:
        for seed in args.seeds:
            path = runs_dir / f"{name}_seed{seed}" / "metrics.json"
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as f:
                m = json.load(f)
            rel = m.get("final_rel_l2_T1")
            rel_c = m.get("final_rel_l2_T1_centered")
            total = m.get("total_time_s")
            t_target = m.get("time_to_target_s")
            rel_s = "n/a" if rel is None else f"{rel:.4f}"
            rel_c_s = "n/a" if rel_c is None else f"{rel_c:.4f}"
            total_s = "n/a" if total is None else f"{total:.1f}"
            target_s = "n/a" if t_target is None else f"{t_target:.1f}"
            print(
                f"{name:<22}{seed:>5}{rel_s:>10}{rel_c_s:>10}"
                f"{total_s:>10}{target_s:>10}"
            )


if __name__ == "__main__":
    main()
