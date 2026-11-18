"""Training loop for the four baseline cells."""

from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .config import save_config
from .losses import data_loss, heat_bc_loss, physics_loss, residual_loss
from .models import count_parameters, make_T_net, make_w_net
from .problem import (
    T1_true,
    make_grid,
    mse,
    relative_l2,
    sample_boundary,
    sample_interior,
    sample_observations,
    w_true,
)

HISTORY_FIELDS = [
    "stage",
    "step",
    "time_s",
    "loss",
    "data_loss",
    "physics_loss",
    "bc_loss",
    "rel_l2_w",
    "rel_l2_T1",
    "rel_l2_T1_centered",
    "mean_error_T1",
    "mse_w",
    "mse_T1",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(self, cfg: dict, output_dir: str | Path) -> None:
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cfg["output_dir"] = str(self.output_dir)

        self.seed = int(cfg["seed"])
        set_seed(self.seed)

        self.device = self._resolve_device(cfg["device"])
        self.w_net = make_w_net(cfg).to(self.device)
        self.T_net = make_T_net(cfg).to(self.device)

        self.nu = float(cfg["problem"]["nu"])
        self.D = float(cfg["problem"]["D"])
        self.kappa = float(cfg["problem"]["kappa"])
        self.consistent = bool(cfg["problem"]["consistent"])
        if self.consistent and (self.D != 1.0 or self.kappa != 1.0):
            raise ValueError(
                "consistent=True assumes D = kappa = 1; "
                "set problem.consistent=false or use the default constants."
            )

        data_cfg = cfg["data"]
        self.x_obs, self.w_obs = sample_observations(
            n_obs=int(data_cfg["n_obs"]),
            obs_range=float(data_cfg["obs_range"]),
            noise_level=float(data_cfg["noise_level"]),
            seed=self.seed,
            device=self.device,
        )
        self.x_int = sample_interior(
            n_interior=int(data_cfg["n_interior"]),
            seed=self.seed,
            device=self.device,
        )
        self.x_bc, self.normals = sample_boundary(
            n_per_side=int(data_cfg["n_boundary_per_side"]),
            seed=self.seed,
            device=self.device,
        )
        self.grid, self.grid_shape = make_grid(101, self.device)

        prior = cfg["train"].get("mean_T1_prior")
        if prior is None:
            with torch.no_grad():
                T_ref_grid = T1_true(
                    self.grid[:, 0:1],
                    self.grid[:, 1:2],
                    nu=self.nu,
                    consistent=self.consistent,
                )
                prior = float(T_ref_grid.mean())
        self.mean_prior = float(prior)

        self.history: list[dict] = []
        self.stage_times: dict[str, float] = {}
        self.start_time: float | None = None
        self.time_to_target: float | None = None
        self.step_to_target: int | None = None
        self.final_rel_l2_w: float | None = None
        self.final_rel_l2_T1: float | None = None
        self.final_rel_l2_T1_centered: float | None = None
        self.final_mean_error_T1: float | None = None
        self.final_mse_w: float | None = None
        self.final_mse_T1: float | None = None

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(name: str) -> torch.device:
        if name.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA is not available; falling back to CPU.")
            return torch.device("cpu")
        return torch.device(name)

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _elapsed(self) -> float:
        self._sync()
        if self.start_time is None:
            return 0.0
        return time.perf_counter() - self.start_time

    def _make_optimizer(
        self, params, name: str, lr: float
    ) -> torch.optim.Optimizer:
        name = name.lower()
        weight_decay = float(self.cfg["train"]["weight_decay"])
        if name == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        if name == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        if name == "sgd":
            return torch.optim.SGD(
                params, lr=lr, momentum=0.9, weight_decay=weight_decay
            )
        raise ValueError(f"Unknown optimizer: {name}")

    def _make_scheduler(self, optimizer: torch.optim.Optimizer):
        if self.cfg["train"]["scheduler"] == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=float(self.cfg["train"]["scheduler_factor"]),
                patience=int(self.cfg["train"]["scheduler_patience"]),
            )
        return None

    def _clip(self, params: list[torch.nn.Parameter]) -> None:
        max_norm = self.cfg["train"]["clip_grad_norm"]
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(params, float(max_norm))

    def _physics(self, loss_type: str) -> torch.Tensor:
        physics = physics_loss(
            loss_type,
            self.w_net,
            self.T_net,
            self.x_int,
            self.nu,
            self.D,
            self.kappa,
        )
        if loss_type == "energy":
            weight = float(self.cfg["train"].get("energy_pde_weight", 0.0))
            if weight > 0.0:
                physics = physics + weight * residual_loss(
                    self.w_net,
                    self.T_net,
                    self.x_int,
                    self.nu,
                    self.D,
                    self.kappa,
                )
        return physics

    def _mean_prior_loss(self) -> torch.Tensor | None:
        weight = float(self.cfg["train"].get("mean_prior_weight", 0.0))
        if weight <= 0.0:
            return None
        T_int = self.T_net(self.x_int)
        return weight * (T_int.mean() - self.mean_prior) ** 2

    # ------------------------------------------------------------------
    # Logging and evaluation
    # ------------------------------------------------------------------

    def _log_train(
        self,
        stage: str,
        step: int,
        loss: torch.Tensor,
        data: torch.Tensor | None = None,
        physics: torch.Tensor | None = None,
        bc: torch.Tensor | None = None,
    ) -> None:
        self.history.append(
            {
                "stage": stage,
                "step": step,
                "time_s": self._elapsed(),
                "loss": float(loss.detach().cpu()),
                "data_loss": "" if data is None else float(data.detach().cpu()),
                "physics_loss": "" if physics is None else float(physics.detach().cpu()),
                "bc_loss": "" if bc is None else float(bc.detach().cpu()),
                "rel_l2_w": "",
                "rel_l2_T1": "",
                "rel_l2_T1_centered": "",
                "mean_error_T1": "",
                "mse_w": "",
                "mse_T1": "",
            }
        )

    @torch.no_grad()
    def _predict(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        w_pred = self.w_net(self.grid)
        T_pred = self.T_net(self.grid)
        w_ref = w_true(self.grid[:, 0:1], self.grid[:, 1:2])
        T_ref = T1_true(
            self.grid[:, 0:1],
            self.grid[:, 1:2],
            nu=self.nu,
            consistent=self.consistent,
        )
        return w_pred, T_pred, w_ref, T_ref

    def _evaluate(
        self, stage: str, step: int, check_target: bool = True
    ) -> tuple[float, float]:
        self.w_net.eval()
        self.T_net.eval()
        w_pred, T_pred, w_ref, T_ref = self._predict()
        rel_w = relative_l2(w_pred, w_ref)
        rel_T = relative_l2(T_pred, T_ref)
        # The Neumann inverse problem determines T1 only up to a constant.
        # Report the shape error separately from the mean offset.
        T_pred_c = T_pred - T_pred.mean()
        T_ref_c = T_ref - T_ref.mean()
        rel_T_c = relative_l2(T_pred_c, T_ref_c)
        mean_error_T = (T_pred.mean() - T_ref.mean()).item()
        mse_w = mse(w_pred, w_ref)
        mse_T = mse(T_pred, T_ref)
        self.w_net.train()
        self.T_net.train()

        self.final_rel_l2_w = rel_w
        self.final_rel_l2_T1 = rel_T
        self.final_rel_l2_T1_centered = rel_T_c
        self.final_mean_error_T1 = mean_error_T
        self.final_mse_w = mse_w
        self.final_mse_T1 = mse_T

        self.history.append(
            {
                "stage": stage,
                "step": step,
                "time_s": self._elapsed(),
                "loss": "",
                "data_loss": "",
                "physics_loss": "",
                "bc_loss": "",
                "rel_l2_w": rel_w,
                "rel_l2_T1": rel_T,
                "rel_l2_T1_centered": rel_T_c,
                "mean_error_T1": mean_error_T,
                "mse_w": mse_w,
                "mse_T1": mse_T,
            }
        )

        if (
            check_target
            and self.time_to_target is None
            and rel_T <= float(self.cfg["train"]["target_error"])
        ):
            self.time_to_target = self._elapsed()
            self.step_to_target = step
        return rel_w, rel_T

    # ------------------------------------------------------------------
    # Training stages
    # ------------------------------------------------------------------

    def _train_stage1(self) -> None:
        self._current_stage = "stage1"
        train_cfg = self.cfg["train"]
        steps = int(train_cfg["steps_stage1"])
        eval_every = max(1, int(train_cfg["eval_every"]))
        log_every = max(1, int(train_cfg.get("log_every", 50)))
        params = list(self.w_net.parameters())
        optimizer = self._make_optimizer(
            params,
            str(train_cfg["optimizer_stage1"]),
            float(train_cfg["lr_stage1"]),
        )

        self.w_net.train()
        self._sync()
        t0 = time.perf_counter()
        for step in range(1, steps + 1):
            optimizer.zero_grad(set_to_none=True)
            loss = data_loss(self.w_net, self.x_obs, self.w_obs)
            loss.backward()
            self._clip(params)
            optimizer.step()

            if step % log_every == 0 or step == steps:
                self._log_train("stage1", step, loss, data=loss)
            if step % eval_every == 0 or step == steps:
                self._evaluate("stage1", step, check_target=False)

        self._sync()
        self.stage_times["stage1"] = time.perf_counter() - t0

    def _train_stage2(self) -> None:
        self._current_stage = "stage2"
        train_cfg = self.cfg["train"]
        steps = int(train_cfg["steps_stage2"])
        eval_every = max(1, int(train_cfg["eval_every"]))
        log_every = max(1, int(train_cfg.get("log_every", 50)))
        loss_type = str(train_cfg["loss"])
        lambda_physics = float(train_cfg["lambda_physics"])
        lambda_bc = float(train_cfg["lambda_bc_T"])

        for param in self.w_net.parameters():
            param.requires_grad_(False)

        params = list(self.T_net.parameters())
        optimizer = self._make_optimizer(
            params,
            str(train_cfg["optimizer_stage2"]),
            float(train_cfg["lr_stage2"]),
        )
        scheduler = self._make_scheduler(optimizer)

        self.T_net.train()
        self._sync()
        t0 = time.perf_counter()
        for step in range(1, steps + 1):
            optimizer.zero_grad(set_to_none=True)
            physics = self._physics(loss_type)
            bc = heat_bc_loss(
                self.T_net,
                self.x_bc,
                self.normals,
                self.nu,
                self.consistent,
            )
            loss = lambda_physics * physics + lambda_bc * bc
            mean_penalty = self._mean_prior_loss()
            if mean_penalty is not None:
                loss = loss + mean_penalty
            if not torch.isfinite(loss):
                print(f"Non-finite loss at stage2 step {step}; stopping stage.")
                break

            loss.backward()
            self._clip(params)
            optimizer.step()
            if scheduler is not None:
                scheduler.step(loss.detach())

            if step % log_every == 0 or step == steps:
                self._log_train(
                    "stage2", step, loss, physics=physics, bc=bc
                )
            if step % eval_every == 0 or step == steps:
                self._evaluate("stage2", step, check_target=True)

        self._sync()
        self.stage_times["stage2"] = time.perf_counter() - t0

    def _train_joint(self) -> None:
        self._current_stage = "joint"
        train_cfg = self.cfg["train"]
        steps = int(train_cfg["steps_joint"])
        eval_every = max(1, int(train_cfg["eval_every"]))
        log_every = max(1, int(train_cfg.get("log_every", 50)))
        loss_type = str(train_cfg["loss"])
        lambda_data = float(train_cfg["lambda_data"])
        lambda_physics = float(train_cfg["lambda_physics"])
        lambda_bc = float(train_cfg["lambda_bc_T"])

        params = list(self.w_net.parameters()) + list(self.T_net.parameters())
        optimizer = self._make_optimizer(
            params,
            str(train_cfg["optimizer_joint"]),
            float(train_cfg["lr_joint"]),
        )
        scheduler = self._make_scheduler(optimizer)

        self.w_net.train()
        self.T_net.train()
        self._sync()
        t0 = time.perf_counter()
        for step in range(1, steps + 1):
            optimizer.zero_grad(set_to_none=True)
            data = data_loss(self.w_net, self.x_obs, self.w_obs)
            physics = self._physics(loss_type)
            bc = heat_bc_loss(
                self.T_net,
                self.x_bc,
                self.normals,
                self.nu,
                self.consistent,
            )
            loss = lambda_data * data + lambda_physics * physics + lambda_bc * bc
            mean_penalty = self._mean_prior_loss()
            if mean_penalty is not None:
                loss = loss + mean_penalty
            if not torch.isfinite(loss):
                print(f"Non-finite loss at joint step {step}; stopping.")
                break

            loss.backward()
            self._clip(params)
            optimizer.step()
            if scheduler is not None:
                scheduler.step(loss.detach())

            if step % log_every == 0 or step == steps:
                self._log_train(
                    "joint", step, loss, data=data, physics=physics, bc=bc
                )
            if step % eval_every == 0 or step == steps:
                self._evaluate("joint", step, check_target=True)

        self._sync()
        self.stage_times["joint"] = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def train(self) -> dict:
        mode = str(self.cfg["train"]["mode"])
        self.start_time = time.perf_counter()

        if mode == "two_stage":
            self._train_stage1()
            self._train_stage2()
        elif mode == "joint":
            self._train_joint()
        else:
            raise ValueError(f"Unknown training mode: {mode}")

        self._evaluate(self._current_stage, -1, check_target=True)
        metrics = self._metrics()
        self._save(metrics)
        return metrics

    def _metrics(self) -> dict:
        return {
            "seed": self.seed,
            "device": str(self.device),
            "mode": self.cfg["train"]["mode"],
            "loss": self.cfg["train"]["loss"],
            "n_params_w": count_parameters(self.w_net),
            "n_params_T": count_parameters(self.T_net),
            "stage1_time_s": self.stage_times.get("stage1"),
            "stage2_time_s": self.stage_times.get("stage2"),
            "joint_time_s": self.stage_times.get("joint"),
            "total_time_s": self._elapsed(),
            "final_rel_l2_w": self.final_rel_l2_w,
            "final_rel_l2_T1": self.final_rel_l2_T1,
            "final_rel_l2_T1_centered": self.final_rel_l2_T1_centered,
            "final_mean_error_T1": self.final_mean_error_T1,
            "final_mse_w": self.final_mse_w,
            "final_mse_T1": self.final_mse_T1,
            "mean_prior": self.mean_prior,
            "target_error": float(self.cfg["train"]["target_error"]),
            "time_to_target_s": self.time_to_target,
            "step_to_target": self.step_to_target,
            "config": self.cfg,
        }

    def _save(self, metrics: dict) -> None:
        save_config(self.cfg, self.output_dir / "config.yaml")
        with open(self.output_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        with open(
            self.output_dir / "history.csv", "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
            writer.writeheader()
            for row in self.history:
                writer.writerow(row)

        torch.save(self.w_net.state_dict(), self.output_dir / "model_w.pt")
        torch.save(self.T_net.state_dict(), self.output_dir / "model_T.pt")

        w_pred, T_pred, w_ref, T_ref = self._predict()
        np.savez(
            self.output_dir / "predictions.npz",
            grid=self.grid.detach().cpu().numpy(),
            grid_shape=np.array(self.grid_shape),
            w_pred=w_pred.detach().cpu().numpy(),
            T_pred=T_pred.detach().cpu().numpy(),
            w_ref=w_ref.detach().cpu().numpy(),
            T_ref=T_ref.detach().cpu().numpy(),
        )
