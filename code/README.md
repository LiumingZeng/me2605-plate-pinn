# Baseline and experiments

Status: the baseline is being refactored from the author's thesis implementation into a clean,
reproducible PyTorch codebase. This folder will contain the code for the 2x2 factorial study.

## Problem

Clamped isotropic square plate on `[-1, 1]^2`. The unknown is the linear through-thickness
temperature component `T1(x, y)`; observations are 200 sparse deflections with 1% Gaussian noise,
plus the heat-flux boundary condition. The forward model is

```
D * laplacian^2(w) = -1/(1 - nu) * laplacian(M_T),
M_T = alpha * E * integral_{-h/2}^{h/2} T z dz = alpha * E * h^3 / 12 * T1 + o(h^5),
w = dw/dn = 0 on the boundary.
```

Two tanh MLPs are used: `w`-net `(2 -> 4 x 32 -> 1)` and `T`-net `(2 -> 4 x 128 -> 1)`.

## Planned layout

```
plate_pinn/
  models.py        # w-net, T-net, hard clamped boundary constraint
  losses.py        # energy functional, fourth-order residual, data and boundary terms
  train.py         # joint and two-stage training loops
  diagnostics.py   # Hessian/NTK spectra, gradient cosine similarity, FFT error spectrum
configs/
  energy_two_stage.yaml
  energy_joint.yaml
  residual_two_stage.yaml
  residual_joint.yaml
scripts/
  reproduce_baseline.py
  run_factorial.py
  run_diagnostics.py
```

## Reproduction targets (thesis, RTX 5070 12GB)

| Recipe | Reported time | Relative L2 error in T1 |
|---|---|---|
| Energy + two-stage | 550.62 s | 15.71% |
| PDE residual + joint | 1488.15 s | 81.21% |

The first step is to reproduce these numbers under an equal optimizer budget, then run the two
missing cells (`energy + joint`, `residual + two-stage`).

## Requirements

See `requirements.txt`. A CUDA-capable PyTorch build is required (the thesis used an RTX 5070
12GB; PyTorch 2.7.1+cu128 works).
