"""Loss terms for the four training cells."""

from __future__ import annotations

import torch
import torch.nn as nn

from .problem import (
    T1_true,
    biharmonic,
    grad_scalar,
    laplacian,
    second_derivatives,
)


def data_loss(
    w_net: nn.Module, x_obs: torch.Tensor, w_obs: torch.Tensor
) -> torch.Tensor:
    return torch.mean((w_net(x_obs) - w_obs) ** 2)


def heat_bc_loss(
    T_net: nn.Module,
    x_bc: torch.Tensor,
    normals: torch.Tensor,
    nu: float,
    consistent: bool,
) -> torch.Tensor:
    """Match the normal derivative of T1 to the analytic heat-flux data."""
    T_pred = T_net(x_bc)
    grad_pred = grad_scalar(T_pred, x_bc)
    dn_pred = (grad_pred * normals).sum(dim=1, keepdim=True)

    T_ref = T1_true(x_bc[:, 0:1], x_bc[:, 1:2], nu=nu, consistent=consistent)
    grad_ref = grad_scalar(T_ref, x_bc)
    dn_ref = (grad_ref * normals).sum(dim=1, keepdim=True)

    return torch.mean((dn_pred - dn_ref) ** 2)


def energy_loss(
    w_net: nn.Module,
    T_net: nn.Module,
    x_int: torch.Tensor,
    nu: float,
    D: float,
    kappa: float,
) -> torch.Tensor:
    """Total potential energy Pi[w, T1] (mean over interior points).

    The load is q = -kappa/(1-nu) * lap(T1), so varying the energy gives
        D * biharm(w) + kappa/(1-nu) * lap(T1) = 0.
    """
    w = w_net(x_int)
    w_xx, w_yy, w_xy = second_derivatives(w, x_int)

    T = T_net(x_int)
    lap_T = laplacian(T, x_int)
    q = -(kappa / (1.0 - nu)) * lap_T

    density = (
        0.5
        * D
        * (
            w_xx**2
            + w_yy**2
            + 2.0 * nu * w_xx * w_yy
            + 2.0 * (1.0 - nu) * w_xy**2
        )
        - q * w
    )
    return torch.mean(density)


def residual_loss(
    w_net: nn.Module,
    T_net: nn.Module,
    x_int: torch.Tensor,
    nu: float,
    D: float,
    kappa: float,
) -> torch.Tensor:
    """Mean squared fourth-order PDE residual."""
    w = w_net(x_int)
    bih = biharmonic(w, x_int)

    T = T_net(x_int)
    lap_T = laplacian(T, x_int)

    residual = D * bih + (kappa / (1.0 - nu)) * lap_T
    return torch.mean(residual**2)


def physics_loss(
    loss_type: str,
    w_net: nn.Module,
    T_net: nn.Module,
    x_int: torch.Tensor,
    nu: float,
    D: float,
    kappa: float,
) -> torch.Tensor:
    if loss_type == "energy":
        return energy_loss(w_net, T_net, x_int, nu, D, kappa)
    if loss_type == "residual":
        return residual_loss(w_net, T_net, x_int, nu, D, kappa)
    raise ValueError(f"Unknown loss type: {loss_type}")
