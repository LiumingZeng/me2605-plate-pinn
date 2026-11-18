#!/usr/bin/env python
"""Small correctness and end-to-end smoke checks.

Run from the code/ directory:
    python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plate_pinn.config import apply_smoke_overrides, load_config  # noqa: E402
from plate_pinn.models import HardBoundaryMLP  # noqa: E402
from plate_pinn.problem import (  # noqa: E402
    T1_true,
    biharmonic,
    grad_scalar,
    laplacian,
    sample_boundary,
    sample_interior,
    w_true,
)
from plate_pinn.train import Trainer  # noqa: E402

CONFIGS = [
    "energy_two_stage",
    "energy_joint",
    "residual_two_stage",
    "residual_joint",
]


def check_analytic_pair() -> None:
    torch.manual_seed(0)
    x = (2.0 * torch.rand(256, 2) - 1.0).requires_grad_(True)
    w = w_true(x[:, 0:1], x[:, 1:2])
    T = T1_true(x[:, 0:1], x[:, 1:2], nu=0.3, consistent=True)
    residual = biharmonic(w, x) + laplacian(T, x) / (1.0 - 0.3)
    max_abs = residual.abs().max().item()
    print(f"[check] analytic PDE residual max |r| = {max_abs:.3e}")
    # float32 + high-order derivatives: 1e-3 is a safe pointwise tolerance
    assert max_abs < 1e-3, "analytic pair does not satisfy the PDE"


def check_hard_boundary() -> None:
    torch.manual_seed(0)
    net = HardBoundaryMLP([16, 16])
    points, normals = sample_boundary(32, seed=0, device=torch.device("cpu"))
    w = net(points)
    g = grad_scalar(w, points)
    dn = (g * normals).sum(dim=1, keepdim=True)
    print(
        f"[check] hard BC: max |w|={w.abs().max().item():.3e}, "
        f"max |dw/dn|={dn.abs().max().item():.3e}"
    )
    assert w.abs().max().item() < 1e-6
    assert dn.abs().max().item() < 1e-5


def check_training_cells() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for name in CONFIGS:
        cfg = load_config(ROOT / "configs" / f"{name}.yaml")
        cfg = apply_smoke_overrides(cfg)
        cfg["device"] = device
        cfg["seed"] = 0
        output = ROOT / "runs" / "smoke" / name
        print(f"\n[smoke] {name} on {device}")
        trainer = Trainer(cfg, output)
        metrics = trainer.train()
        print(
            f"[smoke] {name}: rel_T1={metrics['final_rel_l2_T1']:.4f}, "
            f"time={metrics['total_time_s']:.2f}s"
        )


def main() -> None:
    check_analytic_pair()
    check_hard_boundary()
    check_training_cells()
    print("\nAll smoke checks passed.")


if __name__ == "__main__":
    main()
