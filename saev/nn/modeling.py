"""
Neural network architectures for sparse autoencoders.
"""

import dataclasses
import io
import json
import logging
import os
import pathlib
import subprocess
import typing

import beartype
import einops
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor

from .. import __version__, config


@torch.no_grad()
def geometric_median(
    points: torch.Tensor, max_iterations: int = 100, tolerance: float = 1e-6
) -> torch.Tensor:
    """
    Calculate the geometric median of a set of points using Weiszfeld's algorithm.

    The geometric median is a robust estimator of centrality that minimizes the sum
    of Euclidean distances to all points in the set.

    Args:
        points: A tensor of shape (n, d) containing n points in d-dimensional space.
        max_iterations: Maximum number of iterations for the algorithm. Default is 100.
        tolerance: Convergence threshold for early stopping. Default is 1e-6.

    Returns:
        A tensor of shape (d,) representing the geometric median.
    """
    # Use mean as initial estimate
    center = points.mean(axis=0)

    for _ in range(max_iterations):
        distances = torch.norm(points - center, dim=1, keepdim=True)
        # Avoid division by zero
        distances = torch.clamp(distances, min=1e-8)
        weights = 1.0 / distances
        new_center = torch.sum(weights * points, dim=0) / torch.sum(weights)

        center_shift = torch.norm(new_center - center) / torch.norm(center)
        center = new_center

        if center_shift < tolerance:
            break

    return center


@jaxtyped(typechecker=beartype.beartype)
class SparseAutoencoder(torch.nn.Module):
    """
    Sparse auto-encoder (SAE) using L1 sparsity penalty.
    """

    def __init__(self, cfg: config.SparseAutoencoder):
        super().__init__()

        self.cfg = cfg

        self.W_enc = torch.nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(cfg.d_vit, cfg.d_sae))
        )
        self.b_enc = torch.nn.Parameter(torch.zeros(cfg.d_sae))

        self.W_dec = torch.nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(cfg.d_sae, cfg.d_vit))
        )
        self.b_dec = torch.nn.Parameter(torch.zeros(cfg.d_vit))

        self.activation = get_activation(cfg)

        self.logger = logging.getLogger(f"sae(seed={cfg.seed})")

    def forward(
        self, x: Float[Tensor, "batch d_model"]
    ) -> tuple[
        Float[Tensor, "batch d_model"],
        Float[Tensor, "batch d_sae"],
        Float[Tensor, "batch d_sae"],
        torch.nn.Parameter,
    ]:
        """
        Given x, calculates the reconstructed x_hat and the intermediate activations f_x.

        Arguments:
            x: a batch of ViT activations.
        """

        # Remove encoder bias as per Anthropic
        h_pre = (
            einops.einsum(
                x - self.b_dec, self.W_enc, "... d_vit, d_vit d_sae -> ... d_sae"
            )
            + self.b_enc
        )
        f_x = self.activation(h_pre)
        x_hat = self.decode(f_x)

        return x_hat, f_x, h_pre, self.W_dec

    def decode(
        self, f_x: Float[Tensor, "batch d_sae"]
    ) -> Float[Tensor, "batch d_model"]:
        x_hat = (
            einops.einsum(f_x, self.W_dec, "... d_sae, d_sae d_vit -> ... d_vit")
            + self.b_dec
        )
        return x_hat

    @torch.no_grad()
    def init_b_dec(self, vit_acts: Float[Tensor, "n d_vit"]):
        if self.cfg.n_reinit_samples <= 0:
            self.logger.info("Skipping init_b_dec.")
            return
        previous_b_dec = self.b_dec.clone().cpu()
        vit_acts = vit_acts[: self.cfg.n_reinit_samples]
        assert len(vit_acts) == self.cfg.n_reinit_samples

        if self.cfg.use_geometric_median:
            self.logger.info("Using geometric median for b_dec initialization")
            result = geometric_median(vit_acts)
        else:
            # Use arithmetic mean
            self.logger.info("Using arithmetic mean for b_dec initialization")
            result = vit_acts.mean(axis=0)

        previous_distances = torch.norm(vit_acts - previous_b_dec, dim=-1)
        distances = torch.norm(vit_acts - result, dim=-1)
        self.logger.info(
            "Prev dist: %.3f; new dist: %.3f",
            previous_distances.median(axis=0).values.mean().item(),
            distances.median(axis=0).values.mean().item(),
        )
        self.b_dec.data = result.to(self.b_dec.dtype).to(self.b_dec.device)

    @torch.no_grad()
    def normalize_w_dec(self):
        """
        Set W_dec to unit-norm columns.
        """
        if self.cfg.normalize_w_dec:
            self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

    @torch.no_grad()
    def remove_parallel_grads(self):
        """
        Update grads so that they remove the parallel component
            (d_sae, d_vit) shape
        """
        if not self.cfg.remove_parallel_grads:
            return

        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_sae d_vit, d_sae d_vit -> d_sae",
        )

        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_sae, d_sae d_vit -> d_sae d_vit",
        )


@beartype.beartype
def get_activation(cfg: config.SparseAutoencoder) -> torch.nn.Module:
    if isinstance(cfg, config.Relu):
        return torch.nn.ReLU()
    elif isinstance(cfg, config.JumpRelu):
        raise NotImplementedError()
    else:
        typing.assert_never(cfg)


@beartype.beartype
def dump(fpath: str, sae: SparseAutoencoder):
    """
    Save an SAE checkpoint to disk along with configuration, using the [trick from equinox](https://docs.kidger.site/equinox/examples/serialisation).

    Arguments:
        fpath: filepath to save checkpoint to.
        sae: sparse autoencoder checkpoint to save.
    """
    header = {
        "schema": 1,
        "cfg": dataclasses.asdict(sae.cfg),
        "cls": sae.cfg.__class__.__name__,
        "commit": current_git_commit() or "unknown",
        "lib": __version__,
    }

    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, "wb") as fd:
        header_str = json.dumps(header)
        fd.write((header_str + "\n").encode("utf-8"))
        torch.save(sae.state_dict(), fd)


@beartype.beartype
def load(fpath: str, *, device: str = "cpu") -> SparseAutoencoder:
    """
    Loads a sparse autoencoder from disk.

    `fpath` should be a path to a file saved with :func:`dump`.

    `device` is the device to load the model onto. Defaults to `"cpu"`.

    Returns an instance of :class:`SparseAutoencoder` with the loaded weights.
    """
    with open(fpath, "rb") as fd:
        header = json.loads(fd.readline())
        buffer = io.BytesIO(fd.read())

    if "schema" not in header:
        # Original, pre-schema stuff.
        for keyword in ("sparsity_coeff", "ghost_grads"):
            header.pop(keyword, None)
        cfg = config.Relu(**header)
    elif header["schema"] == 1:
        cls = getattr(config, header["cls"])  # default for v0
        cfg = cls(**header["cfg"])
    else:
        raise ValueError(f"Unknown schema version: {header['schema']}")

    model = SparseAutoencoder(cfg)
    model.load_state_dict(torch.load(buffer, weights_only=True, map_location=device))
    return model


@beartype.beartype
def current_git_commit() -> str | None:
    """
    Best-effort short SHA of the repo containing *this* file.

    Returns `None` when
    * `git` executable is missing,
    * we’re not inside a git repo (e.g. installed wheel),
    * or any git call errors out.
    """
    try:
        # Walk up until we either hit a .git dir or the FS root
        here = pathlib.Path(__file__).resolve()
        for parent in (here, *here.parents):
            if (parent / ".git").exists():
                break
        else:  # no .git found
            return None

        result = subprocess.run(
            ["git", "-C", str(parent), "rev-parse", "--short", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
        return result.stdout.strip() or None
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
