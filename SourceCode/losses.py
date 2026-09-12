#Author - Dipanwita Thakur
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# BASIC REGRESSION LOSSES
# =============================================================================

class MSELoss(nn.Module):
    """Standard mean squared error loss."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        prediction = prediction.float()
        target = target.float()

        return F.mse_loss(
            prediction,
            target,
            reduction=self.reduction,
        )


class MAELoss(nn.Module):
    """Mean absolute error loss."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        prediction = prediction.float()
        target = target.float()

        return F.l1_loss(
            prediction,
            target,
            reduction=self.reduction,
        )


class SmoothL1Loss(nn.Module):
    """
    Smooth L1 / Huber-style regression loss.

    More robust than MSE to occasional large target errors.
    """

    def __init__(
        self,
        beta: float = 0.1,
        reduction: str = "mean",
    ):
        super().__init__()

        self.beta = beta
        self.reduction = reduction

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        prediction = prediction.float()
        target = target.float()

        return F.smooth_l1_loss(
            prediction,
            target,
            beta=self.beta,
            reduction=self.reduction,
        )


# =============================================================================
# STABLE REGRESSION LOSS
# =============================================================================

class StableRegressionLoss(nn.Module):
    """
    Stable regression loss for MAST-Fuse.

    The loss combines:

        1. SmoothL1 regression loss
        2. optional MSE component
        3. optional prediction-range penalty

    This is appropriate for the MAST-Fuse synthetic aphid target,
    where the target is approximately normalized to [0, 1].

    Parameters
    ----------
    smooth_l1_weight:
        Weight of SmoothL1 loss.

    mse_weight:
        Weight of MSE loss.

    range_penalty_weight:
        Penalizes predictions below min_target or above max_target.

    min_target:
        Lower expected target bound.

    max_target:
        Upper expected target bound.

    beta:
        SmoothL1 transition parameter.

    Example
    -------
    criterion = StableRegressionLoss(
        smooth_l1_weight=1.0,
        mse_weight=0.25,
        range_penalty_weight=0.05,
    )

    loss = criterion(prediction, target)
    """

    def __init__(
        self,
        smooth_l1_weight: float = 1.0,
        mse_weight: float = 0.25,
        range_penalty_weight: float = 0.05,
        min_target: float = 0.0,
        max_target: float = 1.0,
        beta: float = 0.1,
    ):
        super().__init__()

        self.smooth_l1_weight = float(smooth_l1_weight)
        self.mse_weight = float(mse_weight)
        self.range_penalty_weight = float(range_penalty_weight)

        self.min_target = float(min_target)
        self.max_target = float(max_target)

        self.beta = float(beta)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        # -------------------------------------------------------------
        # Ensure compatible dtype and shape
        # -------------------------------------------------------------

        prediction = prediction.float()
        target = target.float()

        prediction = prediction.reshape(-1)
        target = target.reshape(-1)

        if prediction.shape != target.shape:
            raise ValueError(
                "Prediction and target shapes do not match: "
                f"{tuple(prediction.shape)} vs "
                f"{tuple(target.shape)}"
            )

        # -------------------------------------------------------------
        # Protect against NaN / Inf
        # -------------------------------------------------------------

        if not torch.isfinite(prediction).all():
            raise ValueError(
                "Model prediction contains NaN or Inf."
            )

        if not torch.isfinite(target).all():
            raise ValueError(
                "Target contains NaN or Inf."
            )

        # -------------------------------------------------------------
        # SmoothL1
        # -------------------------------------------------------------

        smooth_l1 = F.smooth_l1_loss(
            prediction,
            target,
            beta=self.beta,
            reduction="mean",
        )

        # -------------------------------------------------------------
        # MSE
        # -------------------------------------------------------------

        mse = F.mse_loss(
            prediction,
            target,
            reduction="mean",
        )

        # -------------------------------------------------------------
        # Range penalty
        #
        # Only predictions outside [min_target, max_target]
        # are penalized.
        # -------------------------------------------------------------

        below = F.relu(
            self.min_target - prediction
        )

        above = F.relu(
            prediction - self.max_target
        )

        range_penalty = (
            below.pow(2).mean()
            +
            above.pow(2).mean()
        )

        # -------------------------------------------------------------
        # Total loss
        # -------------------------------------------------------------

        loss = (
            self.smooth_l1_weight * smooth_l1
            +
            self.mse_weight * mse
            +
            self.range_penalty_weight * range_penalty
        )

        return loss

    def components(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> dict:
        """
        Return individual loss components for logging/debugging.
        """

        prediction = prediction.float().reshape(-1)
        target = target.float().reshape(-1)

        smooth_l1 = F.smooth_l1_loss(
            prediction,
            target,
            beta=self.beta,
            reduction="mean",
        )

        mse = F.mse_loss(
            prediction,
            target,
            reduction="mean",
        )

        below = F.relu(
            self.min_target - prediction
        )

        above = F.relu(
            prediction - self.max_target
        )

        range_penalty = (
            below.pow(2).mean()
            +
            above.pow(2).mean()
        )

        total = (
            self.smooth_l1_weight * smooth_l1
            +
            self.mse_weight * mse
            +
            self.range_penalty_weight * range_penalty
        )

        return {
            "total": total.detach(),
            "smooth_l1": smooth_l1.detach(),
            "mse": mse.detach(),
            "range_penalty": range_penalty.detach(),
        }


# =============================================================================
# WEIGHTED REGRESSION LOSS
# =============================================================================

class WeightedRegressionLoss(nn.Module):
    """
    Weighted combination of MSE and MAE.

    Useful when you want explicit control over robustness.
    """

    def __init__(
        self,
        mse_weight: float = 0.5,
        mae_weight: float = 0.5,
    ):
        super().__init__()

        self.mse_weight = float(mse_weight)
        self.mae_weight = float(mae_weight)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        prediction = prediction.float().reshape(-1)
        target = target.float().reshape(-1)

        mse = F.mse_loss(
            prediction,
            target,
        )

        mae = F.l1_loss(
            prediction,
            target,
        )

        return (
            self.mse_weight * mse
            +
            self.mae_weight * mae
        )


# =============================================================================
# HUBER LOSS
# =============================================================================

class HuberRegressionLoss(nn.Module):
    """Huber regression loss."""

    def __init__(
        self,
        delta: float = 1.0,
    ):
        super().__init__()

        self.delta = float(delta)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        prediction = prediction.float()
        target = target.float()

        return F.huber_loss(
            prediction,
            target,
            delta=self.delta,
            reduction="mean",
        )


# =============================================================================
# LOSS FACTORY
# =============================================================================

def get_loss(
    name: str = "stable",
    **kwargs,
) -> nn.Module:
    """
    Construct a loss function by name.

    Supported names:

        stable
        smooth_l1
        mse
        mae
        huber
        weighted

    Example
    -------
    criterion = get_loss("stable")
    """

    name = name.lower().strip()

    if name in {
        "stable",
        "stable_regression",
        "stable_regression_loss",
    }:
        return StableRegressionLoss(**kwargs)

    if name in {
        "smooth_l1",
        "smoothl1",
        "huber_smooth",
    }:
        return SmoothL1Loss(**kwargs)

    if name == "mse":
        return MSELoss(**kwargs)

    if name in {
        "mae",
        "l1",
    }:
        return MAELoss(**kwargs)

    if name == "huber":
        return HuberRegressionLoss(**kwargs)

    if name in {
        "weighted",
        "weighted_regression",
    }:
        return WeightedRegressionLoss(**kwargs)

    raise ValueError(
        f"Unknown loss '{name}'. "
        "Supported losses: "
        "stable, smooth_l1, mse, mae, huber, weighted."
    )


# =============================================================================
# TEST
# =============================================================================

def test_losses():

    print("=" * 80)
    print("TESTING MAST-Fuse LOSSES")
    print("=" * 80)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # -------------------------------------------------------------
    # Synthetic regression data
    # -------------------------------------------------------------

    torch.manual_seed(42)

    target = torch.tensor(
        [
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            1.00,
        ],
        dtype=torch.float32,
        device=device,
    )

    prediction = torch.tensor(
        [
            0.12,
            0.20,
            0.55,
            0.70,
            0.95,
            0.98,
        ],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    # -------------------------------------------------------------
    # Stable regression loss
    # -------------------------------------------------------------

    criterion = StableRegressionLoss(
        smooth_l1_weight=1.0,
        mse_weight=0.25,
        range_penalty_weight=0.05,
    ).to(device)

    loss = criterion(
        prediction,
        target,
    )

    print()
    print("StableRegressionLoss")
    print(f"  Loss: {loss.item():.8f}")

    components = criterion.components(
        prediction,
        target,
    )

    print("  Components:")

    for key, value in components.items():
        print(
            f"    {key:<16}: "
            f"{value.item():.8f}"
        )

    # -------------------------------------------------------------
    # Backward test
    # -------------------------------------------------------------

    loss.backward()

    print()
    print("Gradient test")

    if prediction.grad is None:
        raise RuntimeError(
            "Gradient was not generated."
        )

    if not torch.isfinite(
        prediction.grad
    ).all():

        raise RuntimeError(
            "Loss generated NaN/Inf gradients."
        )

    print(
        "  Gradient shape:",
        tuple(prediction.grad.shape),
    )

    # -------------------------------------------------------------
    # Test factory
    # -------------------------------------------------------------

    print()
    print("Loss factory")

    names = [
        "stable",
        "smooth_l1",
        "mse",
        "mae",
        "huber",
        "weighted",
    ]

    for name in names:

        criterion = get_loss(name).to(device)

        prediction_test = prediction.detach()

        value = criterion(
            prediction_test,
            target,
        )

        print(
            f"  {name:<12}: "
            f"{value.item():.8f}"
        )

    print()
    print("=" * 80)
    print("LOSS TEST PASSED")
    print("=" * 80)


if __name__ == "__main__":
    test_losses()
