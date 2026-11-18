# ME2605 Final Project — Disentangling Energy and Residual Losses in PINNs for Fourth-Order Plate Bending

**Course:** ME2605 Training Neural Networks, Fall 2026  
**Author:** Liuming Zeng

## Research question

The baseline solves a thermoelastic thin-plate inverse problem with a two-stage physics-informed
neural network (PINN). It reports a large gap between two recipes:

| Recipe | Time | Relative error in $T_1$ |
|---|---|---|
| Energy functional + two-stage training | 550.62 s | 15.71% |
| PDE residual + joint training | 1488.15 s | 81.21% |

These recipes differ in **two** factors at once: the loss (energy vs. PDE residual) and the
training scheme (two-stage vs. joint). This project disentangles them with a 2×2 factorial
design, diagnoses the dominant factor using Hessian/NTK spectra, gradient-conflict metrics and
error spectra, and tests whether a simple intervention can close the gap.

## Planned experiments

- **2×2 factorial:** {energy, residual} × {joint, two-stage}, same networks, data and optimizer
  budget, 3 seeds.
- **Diagnostics:** Gauss–Newton/Hessian spectrum (Lanczos), empirical NTK, gradient cosine
  similarity, FFT error spectrum, loss-landscape slices.
- **Interventions:** loss normalization, mixed second-order formulation for the residual,
  optimizer/schedule/precision changes.
- **Success target:** best recipe reaches 15.71% in ≤ 495 s (≥10% faster than the 550.62 s
  baseline).

## Repository structure

```
proposal/    Proposal (LaTeX + PDF)
code/        Baseline and experiments (to be added)
```

## Baseline

The baseline is the two-stage energy PINN from the author's master's thesis:

> L. Zeng, *Analysis and Solution of Thermoelastic Deformation Problem for Thin Plates*,
> M.Sc. thesis, Sun Yat-sen University, 2026.

## Status

Proposal stage. Code and experiment configs will be added during the project.
