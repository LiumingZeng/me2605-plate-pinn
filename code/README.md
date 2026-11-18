# Baseline and experiments

Runnable PyTorch baseline for the ME2605 final project. It implements the four cells of the
proposed 2x2 study:

|  | Two-stage | Joint |
|---|---|---|
| **Energy functional** | `configs/energy_two_stage.yaml` | `configs/energy_joint.yaml` |
| **PDE residual** | `configs/residual_two_stage.yaml` | `configs/residual_joint.yaml` |

## Problem

Clamped isotropic square plate on `[-1, 1]^2`. The unknown is the linear through-thickness
temperature component `T1(x, y)`; observations are 200 sparse deflections with 1% Gaussian noise,
plus the heat-flux boundary condition. The forward model is

```
D * biharm(w) = -kappa / (1 - nu) * lap(T1),
w = dw/dn = 0 on the boundary,
```

where `biharm(w) = w_xxxx + 2 w_xxyy + w_yyyy` and `kappa` collects
`alpha * E * h^3 / 12`. The default dimensionless setup uses `D = kappa = 1`, `nu = 0.3`.

Two tanh MLPs are used:

- `w`-net: `2 -> 4 x 32 -> 1`, with the hard clamped multiplier `(x^2 - 1)^2 (y^2 - 1)^2`.
- `T`-net: `2 -> 4 x 128 -> 1`.

## Analytic benchmark note

The thesis reports the pair

```
w      = (x^2 - 1)^2 (y^2 - 1)^2
T1     = -1/5 (-x^6 - y^6 + 15 x^4 y^2 + 15 x^2 y^4 - 10 x^4 - 10 y^4 + 10 x^2 + 10 y^2)
         -1/5 (4 x^6 - 20 x^4 + 60 x^2)
         -1/5 (4 y^6 - 20 y^4 + 60 y^2)
```

but this `T1` does **not** exactly satisfy `D*biharm(w) + kappa/(1-nu)*lap(T1) = 0`. With the
default `problem.consistent: true`, the code adds the polynomial correction

```
delta_T1 = 12 nu x^4 y^2 + 12 nu x^2 y^4 - 72 nu x^2 y^2 + 40 nu y^2
           - 18 x^2 y^4 + 36 x^2 y^2 + 6/5 y^6 - 12 y^2,
```

which makes the analytic pair exact for any `nu` (with `D = kappa = 1`). Set
`problem.consistent: false` (or use `configs/thesis_reference.yaml`) to reproduce the
uncorrected thesis polynomial. The corrected benchmark is the default because a consistent pair
is needed for a clean energy-vs-residual comparison.

## T1 constant null space and stabilization

The heat-flux condition is a Neumann condition, so the inverse problem determines `T1` only up
to an additive constant. The residual loss is insensitive to that constant; the raw relative
`L2` error in `T1` can therefore be dominated by a mean offset even when the shape is accurate.
Two knobs handle this:

- `train.mean_prior_weight` (default `0.0`): adds
  `weight * (mean(T_pred) - mean_prior)^2`. If `train.mean_T1_prior` is null, `mean_prior` is
  computed from the analytic benchmark (an extra benchmark prior, not available in a real
  inverse problem). A value around `0.1` fixes the constant without hurting the shape.
- `train.energy_pde_weight` (default `0.0`): for `loss: energy`, adds this weight times the
  PDE residual penalty. The literal potential-energy objective is **linear in T1** when `w` is
  frozen, so it is unbounded below and cannot determine the shape by itself. A positive weight
  (for example `1.0`) turns it into a mixed energy + residual loss and stabilizes training.

All runs report both `final_rel_l2_T1` (raw) and `final_rel_l2_T1_centered` (shape, with the
mean removed), plus `final_mean_error_T1`. For the 2x2 study, use the centered error as the
shape metric and report the mean error separately.

## Quick start

```bash
# from the code/ directory
python -m venv D:/venvs/me2605-pinn
D:/venvs/me2605-pinn/Scripts/python.exe -m pip install --upgrade pip
D:/venvs/me2605-pinn/Scripts/python.exe -m pip install -r requirements.txt

# correctness + end-to-end smoke checks (tiny nets, 10-20 steps, all four cells)
D:/venvs/me2605-pinn/Scripts/python.exe scripts/smoke_test.py

# one cell
D:/venvs/me2605-pinn/Scripts/python.exe scripts/train.py \
    --config configs/energy_two_stage.yaml --seed 0

# the full 2x2 with three seeds
D:/venvs/me2605-pinn/Scripts/python.exe scripts/run_factorial.py --seeds 0 1 2
```

On Linux/macOS, use `python` instead of the Windows venv path.

## Configs

Defaults live in `plate_pinn/config.py`; each YAML only overrides `output_dir` and
`train.mode`/`train.loss`. Useful knobs:

- `train.steps_stage1`, `train.steps_stage2`, `train.steps_joint`
- `train.lr_stage1`, `train.lr_stage2`, `train.lr_joint`
- `train.lambda_data`, `train.lambda_physics`, `train.lambda_bc_T`
- `train.scheduler`, `train.scheduler_patience`, `train.clip_grad_norm`
- `data.n_obs`, `data.noise_level`, `data.n_interior`, `data.n_boundary_per_side`
- `problem.consistent` (see the note above)

Example override:

```yaml
output_dir: runs/energy_two_stage_lr5e3
train:
  mode: two_stage
  loss: energy
  lr_stage2: 0.005
```

## Outputs

Each run writes to its output directory:

```
config.yaml        resolved config
metrics.json       final errors, stage times, time-to-target, seed, param counts
history.csv        per-step losses and periodic grid errors
model_w.pt         w-network state dict
model_T.pt         T-network state dict
predictions.npz    grid, w_pred, T_pred, w_ref, T_ref
```

`metrics.json` includes `time_to_target_s`: the first wall-clock time at which the grid
relative `L2` error in `T1` is at or below `train.target_error` (default `0.1571`, the thesis
reference). This is the primary metric for the 2x2 comparison.

## Code layout

```
plate_pinn/
  config.py      defaults, YAML loading, smoke overrides
  problem.py     analytic solutions, sampling, autograd derivatives, metrics
  models.py      MLP and hard-boundary MLP
  losses.py      data, heat-flux, energy and residual losses
  train.py       two-stage and joint training loops, logging, saving
configs/         four 2x2 configs, two stabilized energy variants,
                 and thesis_reference.yaml
scripts/         train.py, run_factorial.py, smoke_test.py
```

## Baseline reference (thesis, RTX 5070 12GB)

| Recipe | Reported time | Relative L2 error in T1 |
|---|---|---|
| Energy + two-stage | 550.62 s | 15.71% |
| PDE residual + joint | 1488.15 s | 81.21% |

These numbers are references from the thesis. Because the analytic correction changes the
benchmark and the thesis times may cover different phases, the code re-measures the end-to-end
baseline (stage I + stage II) under an equal optimizer budget. Report all speedups relative to
the re-measured baseline.

## Preliminary sanity results

These are single-seed, 2000-step runs with the full networks. They are **not** the final
baseline; they only show that the code paths work and illustrate the null-space issue.

| cell | raw T1 error | centered T1 error | w error |
|---|---|---|---|
| residual + two-stage, mean prior 0.1 | 3.96% | 8.18% | 0.06% |
| mixed energy + residual (PDE weight 1.0) + two-stage, mean prior 0.1 | 3.25% | 3.48% | 0.06% |
| mixed energy + residual (PDE weight 1.0) + joint, mean prior 0.1 | 5.90% | 11.98% | 13.98% |

The pure potential-energy loss (PDE weight 0) is unbounded in stage II and does not learn the
shape; this is why the stabilized variant is provided.

## Remaining work

- The Hessian/NTK/gradient-conflict/FFT diagnostics from the proposal are not implemented yet;
  `history.csv` already records the losses and grid errors needed for the first plots.
- Add plotting utilities for the prediction/error maps and the spectra.
- Run the four cells with three seeds and fill the 2x2 table.
