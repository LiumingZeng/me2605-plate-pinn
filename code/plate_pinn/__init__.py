"""Plate PINN baseline for the ME2605 final project.

Four training cells are supported through config:
  - loss: energy | residual
  - training: two_stage | joint
"""

from .problem import (
    T1_true,
    biharmonic,
    laplacian,
    make_grid,
    relative_l2,
    sample_boundary,
    sample_interior,
    sample_observations,
    w_true,
)
from .models import HardBoundaryMLP, MLP, make_T_net, make_w_net

__all__ = [
    "T1_true",
    "w_true",
    "biharmonic",
    "laplacian",
    "make_grid",
    "relative_l2",
    "sample_boundary",
    "sample_interior",
    "sample_observations",
    "HardBoundaryMLP",
    "MLP",
    "make_T_net",
    "make_w_net",
]
