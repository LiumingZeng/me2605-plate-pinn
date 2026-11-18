"""Network definitions."""

from __future__ import annotations

import torch
import torch.nn as nn


class Sin(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "sin":
        return Sin()
    if name == "softplus":
        return nn.Softplus(beta=1.0)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation: {name}")


class MLP(nn.Module):
    """Fully connected network with Xavier-normal weights and zero bias."""

    def __init__(
        self,
        in_dim: int,
        hidden: list[int],
        out_dim: int,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for width in hidden:
            layers.append(nn.Linear(prev, width))
            layers.append(get_activation(activation))
            prev = width
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HardBoundaryMLP(nn.Module):
    """MLP multiplied by (x^2 - 1)^2 (y^2 - 1)^2.

    The multiplier vanishes, together with its first derivatives, on the
    boundary of [-1, 1]^2, so w = dw/dn = 0 is enforced exactly.
    """

    def __init__(self, hidden: list[int], activation: str = "tanh") -> None:
        super().__init__()
        self.net = MLP(2, hidden, 1, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        bx = (x[:, 0:1] ** 2 - 1.0) ** 2
        by = (x[:, 1:2] ** 2 - 1.0) ** 2
        return bx * by * raw


def make_w_net(cfg: dict) -> HardBoundaryMLP:
    return HardBoundaryMLP(
        hidden=list(cfg["model"]["w_hidden"]),
        activation=cfg["model"]["activation"],
    )


def make_T_net(cfg: dict) -> MLP:
    return MLP(
        in_dim=2,
        hidden=list(cfg["model"]["T_hidden"]),
        out_dim=1,
        activation=cfg["model"]["activation"],
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
