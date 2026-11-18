"""Analytic benchmark, sampling, automatic derivatives and metrics."""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Analytic solution
# ---------------------------------------------------------------------------


def w_true(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Clamped deflection w = (x^2 - 1)^2 (y^2 - 1)^2 on [-1, 1]^2."""
    return (x**2 - 1.0) ** 2 * (y**2 - 1.0) ** 2


def thesis_T1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """T1 polynomial reported in the thesis (chapter 5)."""
    return (
        -0.2
        * (
            -x**6
            - y**6
            + 15.0 * x**4 * y**2
            + 15.0 * x**2 * y**4
            - 10.0 * x**4
            - 10.0 * y**4
            + 10.0 * x**2
            + 10.0 * y**2
        )
        - 0.2 * (4.0 * x**6 - 20.0 * x**4 + 60.0 * x**2)
        - 0.2 * (4.0 * y**6 - 20.0 * y**4 + 60.0 * y**2)
    )


def delta_T1(x: torch.Tensor, y: torch.Tensor, nu: float) -> torch.Tensor:
    """Polynomial correction making the analytic pair satisfy the PDE.

    The thesis T1 does not exactly satisfy
        D * biharm(w) + kappa / (1 - nu) * lap(T1) = 0
    for w = (x^2 - 1)^2 (y^2 - 1)^2.  Adding this delta gives an exact pair
    (with D = kappa = 1) for any Poisson ratio nu.
    """
    return (
        12.0 * nu * x**4 * y**2
        + 12.0 * nu * x**2 * y**4
        - 72.0 * nu * x**2 * y**2
        + 40.0 * nu * y**2
        - 18.0 * x**2 * y**4
        + 36.0 * x**2 * y**2
        + 1.2 * y**6
        - 12.0 * y**2
    )


def T1_true(
    x: torch.Tensor,
    y: torch.Tensor,
    nu: float = 0.3,
    consistent: bool = True,
) -> torch.Tensor:
    """True temperature component T1.

    consistent=True gives a pair (w_true, T1_true) satisfying the plate PDE
    exactly; consistent=False reproduces the uncorrected thesis polynomial.
    """
    value = thesis_T1(x, y)
    if consistent:
        value = value + delta_T1(x, y, nu)
    return value


# ---------------------------------------------------------------------------
# Automatic derivatives
# ---------------------------------------------------------------------------


def grad_scalar(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Gradient of a scalar-output tensor y with respect to x."""
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]


def second_derivatives(
    u: torch.Tensor, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return u_xx, u_yy, u_xy for u(x) with x of shape (N, 2)."""
    g = grad_scalar(u, x)
    u_x, u_y = g[:, 0:1], g[:, 1:2]
    u_xx = grad_scalar(u_x, x)[:, 0:1]
    u_xy = grad_scalar(u_x, x)[:, 1:2]
    u_yy = grad_scalar(u_y, x)[:, 1:2]
    return u_xx, u_yy, u_xy


def laplacian(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    u_xx, u_yy, _ = second_derivatives(u, x)
    return u_xx + u_yy


def biharmonic(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Return u_xxxx + 2 u_xxyy + u_yyyy."""
    u_xx, u_yy, _ = second_derivatives(u, x)

    u_xxx = grad_scalar(u_xx, x)[:, 0:1]
    u_xxxx = grad_scalar(u_xxx, x)[:, 0:1]

    u_xxy = grad_scalar(u_xx, x)[:, 1:2]
    u_xxyy = grad_scalar(u_xxy, x)[:, 1:2]

    u_yyy = grad_scalar(u_yy, x)[:, 1:2]
    u_yyyy = grad_scalar(u_yyy, x)[:, 1:2]

    return u_xxxx + 2.0 * u_xxyy + u_yyyy


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_observations(
    n_obs: int,
    obs_range: float,
    noise_level: float,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse deflection observations with Gaussian noise."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = 2.0 * obs_range * torch.rand(n_obs, 2, generator=g) - obs_range
    w = w_true(x[:, 0:1], x[:, 1:2])
    if noise_level > 0.0:
        sigma = w.std(unbiased=False)
        w = w + noise_level * sigma * torch.randn(n_obs, 1, generator=g)
    return x.to(device), w.to(device)


def sample_interior(
    n_interior: int, seed: int, device: torch.device
) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    x = 2.0 * torch.rand(n_interior, 2, generator=g) - 1.0
    return x.to(device).requires_grad_(True)


def sample_boundary(
    n_per_side: int, seed: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Boundary points on the four sides of [-1, 1]^2 and outward normals."""
    g = torch.Generator(device="cpu").manual_seed(seed + 2)
    t = torch.rand(n_per_side, generator=g)

    bottom = torch.stack([2.0 * t - 1.0, -torch.ones_like(t)], dim=1)
    top = torch.stack([2.0 * t - 1.0, torch.ones_like(t)], dim=1)
    left = torch.stack([-torch.ones_like(t), 2.0 * t - 1.0], dim=1)
    right = torch.stack([torch.ones_like(t), 2.0 * t - 1.0], dim=1)

    points = torch.cat([bottom, top, left, right], dim=0)
    normals = torch.cat(
        [
            torch.tensor([[0.0, -1.0]]).repeat(n_per_side, 1),
            torch.tensor([[0.0, 1.0]]).repeat(n_per_side, 1),
            torch.tensor([[-1.0, 0.0]]).repeat(n_per_side, 1),
            torch.tensor([[1.0, 0.0]]).repeat(n_per_side, 1),
        ],
        dim=0,
    )
    return points.to(device).requires_grad_(True), normals.to(device)


def make_grid(n: int, device: torch.device) -> tuple[torch.Tensor, tuple[int, int]]:
    x = torch.linspace(-1.0, 1.0, n, device=device)
    xx, yy = torch.meshgrid(x, x, indexing="ij")
    points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    return points, (n, n)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def relative_l2(pred: torch.Tensor, true: torch.Tensor) -> float:
    return (
        torch.linalg.vector_norm(pred - true) / torch.linalg.vector_norm(true)
    ).item()


def mse(pred: torch.Tensor, true: torch.Tensor) -> float:
    return torch.mean((pred - true) ** 2).item()
