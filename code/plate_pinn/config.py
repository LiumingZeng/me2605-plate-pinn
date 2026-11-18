"""Configuration loading and small overrides for smoke tests."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "seed": 0,
    "device": "cuda",
    "output_dir": "runs/default",
    "problem": {
        "nu": 0.3,
        "D": 1.0,
        "kappa": 1.0,
        # If true, add a polynomial correction to the thesis T1 so that the
        # analytic pair satisfies D*biharm(w) + kappa/(1-nu)*lap(T1) = 0.
        "consistent": True,
    },
    "data": {
        "n_obs": 200,
        "obs_range": 0.95,
        "noise_level": 0.01,
        "n_interior": 2048,
        "n_boundary_per_side": 50,
    },
    "model": {
        "w_hidden": [32, 32, 32, 32],
        "T_hidden": [128, 128, 128, 128],
        "activation": "tanh",
    },
    "train": {
        "mode": "two_stage",       # two_stage | joint
        "loss": "energy",          # energy | residual
        "steps_stage1": 3000,
        "steps_stage2": 8000,
        "steps_joint": 11000,
        "lr_stage1": 1.0e-2,
        "lr_stage2": 1.0e-2,
        "lr_joint": 1.0e-2,
        "optimizer_stage1": "adam",
        "optimizer_stage2": "adamw",
        "optimizer_joint": "adamw",
        "weight_decay": 1.0e-4,
        "lambda_data": 1.0,
        "lambda_physics": 1.0,
        "lambda_bc_T": 10.0,
        # For loss="energy": 0.0 is the literal thesis potential-energy
        # objective; a positive value adds this weight times the PDE residual
        # penalty, which stabilizes the underdetermined T subproblem.
        "energy_pde_weight": 0.0,
        # The Neumann inverse problem determines T1 only up to a constant.
        # A positive weight adds (mean(T_pred) - mean_prior)^2. If
        # mean_T1_prior is null, the analytic benchmark mean is used (this is
        # an extra benchmark prior, not available in a real inverse problem).
        "mean_prior_weight": 0.0,
        "mean_T1_prior": None,
        "scheduler": "plateau",     # plateau | none
        "scheduler_factor": 0.5,
        "scheduler_patience": 500,
        "clip_grad_norm": 1.0,
        "log_every": 50,
        "eval_every": 200,
        "target_error": 0.1571,     # thesis reference: 15.71% relative L2 in T1
    },
}


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, filling missing keys from DEFAULTS."""
    cfg = copy.deepcopy(DEFAULTS)
    with open(path, "r", encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}
    return _deep_update(cfg, user)


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def apply_smoke_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    """Small, fast settings used to verify the code path end to end."""
    cfg = copy.deepcopy(cfg)
    cfg["data"].update(
        n_obs=64,
        n_interior=128,
        n_boundary_per_side=8,
        noise_level=0.01,
    )
    cfg["model"].update(w_hidden=[16, 16], T_hidden=[32, 32])
    cfg["train"].update(
        steps_stage1=10,
        steps_stage2=10,
        steps_joint=20,
        scheduler_patience=5,
        eval_every=5,
    )
    return cfg
