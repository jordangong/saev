import dataclasses
import typing

import beartype
import torch
from jaxtyping import Bool, Float, jaxtyped
from torch import Tensor

from .. import config


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True, slots=True)
class Loss:
    """The loss term for an autoencoder training batch."""

    @property
    def loss(self) -> Float[Tensor, ""]:
        """Total loss."""
        raise NotImplementedError()

    def metrics(self) -> dict[str, object]:
        raise NotImplementedError()


@jaxtyped(typechecker=beartype.beartype)
class Objective(torch.nn.Module):
    def forward(
        self,
        x: Float[Tensor, "batch d_model"],
        f_x: Float[Tensor, "batch d_sae"],
        x_hat: Float[Tensor, "batch d_model"],
    ) -> Loss:
        raise NotImplementedError()


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True, slots=True)
class VanillaLoss(Loss):
    """The vanilla loss terms for an training batch."""

    mse: Float[Tensor, ""]
    """Reconstruction loss (mean squared error)."""
    sparsity: Float[Tensor, ""]
    """Sparsity loss, typically lambda * L1."""
    ghost_grad: Float[Tensor, ""] | None
    """Ghost gradient loss, if any."""
    l0: Float[Tensor, ""]
    """L0 magnitude of hidden activations."""
    l1: Float[Tensor, ""]
    """L1 magnitude of hidden activations."""
    dead_neuron_mask: Bool[Tensor, " d_sae"] | None
    """Mask indicating neurons are dead or not"""

    @property
    def loss(self) -> Float[Tensor, ""]:
        """Total loss."""
        if self.ghost_grad is None:
            return self.mse + self.sparsity
        else:
            return self.mse + self.sparsity + self.ghost_grad

    def metrics(self) -> dict[str, object]:
        metrics =  {
            "loss": self.loss.item(),
            "mse": self.mse.item(),
            "l0": self.l0.item(),
            "l1": self.l1.item(),
            "sparsity": self.sparsity.item(),
        }

        if self.ghost_grad is not None:
            metrics["ghost_grad"] = self.ghost_grad.item()

        return metrics


@jaxtyped(typechecker=beartype.beartype)
class VanillaObjective(Objective):
    def __init__(self, cfg: config.Vanilla):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        x: Float[Tensor, "batch d_model"],
        f_x: Float[Tensor, "batch d_sae"],
        x_hat: Float[Tensor, "batch d_model"],
        h_pre: Float[Tensor, "batch d_sae"] | None = None,
        W_dec: torch.nn.Parameter | None = None,
        n_fwd_passes_since_fired: Float[Tensor, " d_sae"] | None = None,
    ) -> VanillaLoss:
        # Some values of x and x_hat can be very large. We can calculate a safe MSE
        # print(x_hat.shape, x.shape)
        mse_loss = mean_squared_err(x_hat, x)

        ghost_loss = None
        dead_neuron_mask = None
        if (
            self.cfg.ghost_grads
            and self.training
            and n_fwd_passes_since_fired is not None
        ):
            dead_neuron_mask = (n_fwd_passes_since_fired > self.cfg.dead_feature_window)
        if (
            self.cfg.ghost_grads
            and self.training
            and h_pre is not None
            and W_dec is not None
            and dead_neuron_mask is not None
            and dead_neuron_mask.sum() > 0
        ):
            # ghost protocol

            # 1.
            residual = x - x_hat
            l2_norm_residual = torch.norm(residual, dim=-1)

            # 2.
            feature_acts_dead_neurons_only = torch.exp(h_pre[:, dead_neuron_mask])
            ghost_out = feature_acts_dead_neurons_only @ W_dec[dead_neuron_mask, :]
            l2_norm_ghost_out = torch.norm(ghost_out, dim=-1)
            norm_scaling_factor = l2_norm_residual / (1e-6 + l2_norm_ghost_out * 2)
            ghost_out *= norm_scaling_factor[:, None].detach()

            # 3.
            ghost_loss = mean_squared_err(ghost_out, residual.detach())
            mse_rescaling_factor = (mse_loss / (ghost_loss + 1e-6)).detach()
            ghost_loss *= mse_rescaling_factor
            ghost_loss = ghost_loss.mean()
 
        mse_loss = mse_loss.mean()
        l0 = (f_x > 0).float().sum(axis=1).mean(axis=0)
        l1 = f_x.sum(axis=1).mean(axis=0)
        sparsity_loss = self.cfg.sparsity_coeff * l1

        return VanillaLoss(
            mse_loss,
            sparsity_loss,
            ghost_loss,
            l0,
            l1,
            dead_neuron_mask
        )


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True, slots=True)
class MatryoshkaLoss(Loss):
    """The composite loss terms for an training batch."""

    @property
    def loss(self) -> Float[Tensor, ""]:
        raise NotImplementedError()


@jaxtyped(typechecker=beartype.beartype)
class MatryoshkaObjective(Objective):
    """Torch module for calculating the matryoshka loss for an SAE."""

    def __init__(self, cfg: config.Matryoshka):
        super().__init__()
        self.cfg = cfg

    def forward(self) -> "MatryoshkaLoss.Loss":
        raise NotImplementedError()


@beartype.beartype
def get_objective(cfg: config.Objective) -> Objective:
    if isinstance(cfg, config.Vanilla):
        return VanillaObjective(cfg)
    elif isinstance(cfg, config.Matryoshka):
        return MatryoshkaObjective(cfg)
    else:
        typing.assert_never(cfg)


@jaxtyped(typechecker=beartype.beartype)
def ref_mean_squared_err(
    x_hat: Float[Tensor, "*d"], x: Float[Tensor, "*d"], norm: bool = True
) -> Float[Tensor, "*d"]:
    mse_loss = torch.pow((x_hat - x.float()), 2)

    if norm:
        mse_loss /= (x**2).sum(dim=-1, keepdim=True)
    return mse_loss


@jaxtyped(typechecker=beartype.beartype)
def mean_squared_err(
    x_hat: Float[Tensor, "*batch d"], x: Float[Tensor, "*batch d"], norm: bool = True
) -> Float[Tensor, "*batch d"]:
    upper = x.abs().max()
    x = x / upper
    x_hat = x_hat / upper

    mse = (x_hat - x) ** 2
    # (sam): I am now realizing that we normalize by the L2 norm of x.
    if norm:
        mse /= (x**2).sum(dim=-1, keepdim=True) + 1e-12
        return mse

    return mse * upper * upper
