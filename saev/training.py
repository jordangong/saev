"""
Trains many SAEs in parallel to amortize the cost of loading a single batch of data over many SAE training runs.
"""

import dataclasses
import json
import logging
import os.path

import beartype
import einops
import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

import wandb

from . import activations, config, helpers, nn

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("train")


@torch.no_grad()
def init_b_dec_batched(saes: torch.nn.ModuleList, dataset: activations.Dataset):
    n_samples = max(sae.cfg.n_reinit_samples for sae in saes)
    if not n_samples:
        return
    # Pick random samples using first SAE's seed.
    if dataset.cfg.shuffled:
        perm = np.arange(len(dataset))
    else:
        perm = np.random.default_rng(seed=saes[0].cfg.seed).permutation(len(dataset))
    perm = perm[:n_samples]
    samples = [
        dataset[p.item()]
        for p in helpers.progress(
            perm, every=25_000, desc="examples to re-init b_dec"
        )
    ]
    vit_acts = torch.stack([sample["act"] for sample in samples])
    for sae in saes:
        sae.init_b_dec(vit_acts[: sae.cfg.n_reinit_samples])


@beartype.beartype
def make_saes(
    cfgs: list[tuple[config.SparseAutoencoder, config.Objective]],
) -> tuple[torch.nn.ModuleList, torch.nn.ModuleList, list[dict[str, object]]]:
    saes, objectives, param_groups = [], [], []
    for sae_cfg, obj_cfg in cfgs:
        sae = nn.SparseAutoencoder(sae_cfg)
        saes.append(sae)
        # Use an empty LR because our first step is warmup.
        param_groups.append({"params": sae.parameters(), "lr": 0.0})
        objectives.append(nn.get_objective(obj_cfg))

    return torch.nn.ModuleList(saes), torch.nn.ModuleList(objectives), param_groups


##################
# Parallel Wandb #
##################


MetricQueue = list[tuple[int, dict[str, object]]]


class ParallelWandbRun:
    """
    Inspired by https://community.wandb.ai/t/is-it-possible-to-log-to-multiple-runs-simultaneously/4387/3.
    """

    def __init__(
        self, project: str, cfgs: list[config.Train], mode: str, tags: list[str]
    ):
        cfg, *cfgs = cfgs
        self.project = project
        self.cfgs = cfgs
        self.mode = mode
        self.tags = tags

        self.live_run = wandb.init(
            project=project, config=dataclasses.asdict(cfg), mode=mode, tags=tags
        )

        self.metric_queues: list[MetricQueue] = [[] for _ in self.cfgs]

    def log(self, metrics: list[dict[str, object]], *, step: int):
        metric, *metrics = metrics
        self.live_run.log(metric, step=step)
        for queue, metric in zip(self.metric_queues, metrics):
            queue.append((step, metric))

    def finish(self) -> list[str]:
        ids = [self.live_run.id]
        # Log the rest of the runs.
        self.live_run.finish()

        for queue, cfg in zip(self.metric_queues, self.cfgs):
            run = wandb.init(
                project=self.project,
                config=dataclasses.asdict(cfg),
                mode=self.mode,
                tags=self.tags + ["queued"],
            )
            for step, metric in queue:
                run.log(metric, step=step)
            ids.append(run.id)
            run.finish()

        return ids


@beartype.beartype
def main(cfgs: list[config.Train], load_checkpoint: str | None = None) -> list[str]:
    if load_checkpoint:
        # Skip training and load checkpoint for evaluation only
        logger.info(
            "Skipping training and loading checkpoint from '%s'", load_checkpoint
        )
        cfg = cfgs[0]

        # Load the checkpoint
        saes = nn.load(load_checkpoint)
        objectives = nn.get_objective(cfg.objective)

        # Wrap with ModuleLists
        saes = torch.nn.ModuleList([saes])
        objectives = torch.nn.ModuleList([objectives])

        # Initialize wandb run for logging evaluation results
        mode = "online" if cfg.track else "disabled"
        tags = [cfg.tag] if cfg.tag else []
        run = ParallelWandbRun(cfg.wandb_project, cfgs, mode, tags)
        steps = 0
    else:
        # Normal training flow
        saes, objectives, run, steps = train(cfgs)
    # Cheap(-ish) evaluation
    eval_metrics = evaluate(cfgs, saes, objectives)
    metrics = [metric.for_wandb() for metric in eval_metrics]
    run.log(metrics, step=steps)
    ids = run.finish()

    for cfg, id, metric, sae in zip(cfgs, ids, eval_metrics, saes):
        logger.info(
            "Checkpoint %s has %d dense features (%.1f)",
            id,
            metric.n_dense,
            metric.n_dense / sae.cfg.d_sae * 100,
        )
        logger.info(
            "Checkpoint %s has %d dead features (%.1f%%)",
            id,
            metric.n_dead,
            metric.n_dead / sae.cfg.d_sae * 100,
        )
        logger.info(
            "Checkpoint %s has %d *almost* dead (<1e-7) features (%.1f)",
            id,
            metric.n_almost_dead,
            metric.n_almost_dead / sae.cfg.d_sae * 100,
        )

        if not load_checkpoint:
            ckpt_fpath = os.path.join(cfg.ckpt_path, id, "sae.pt")
            nn.dump(ckpt_fpath, sae)
            logger.info("Dumped checkpoint to '%s'.", ckpt_fpath)
            cfg_fpath = os.path.join(cfg.ckpt_path, id, "config.json")
            with open(cfg_fpath, "w") as fd:
                json.dump(dataclasses.asdict(cfg), fd, indent=4)

    return ids


@beartype.beartype
def train(
    cfgs: list[config.Train],
) -> tuple[torch.nn.ModuleList, torch.nn.ModuleList, ParallelWandbRun, int]:
    """
    Explicitly declare the optimizer, schedulers, dataloader, etc outside of `main` so that all the variables are dropped from scope and can be garbage collected.
    """
    if len(split_cfgs(cfgs)) != 1:
        raise ValueError("Configs are not parallelizeable: {cfgs}.")

    logger.info("Parallelizing %d runs.", len(cfgs))

    cfg = cfgs[0]
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        if cfg.span_all_devices:
            num_gpus = torch.cuda.device_count()
        else:
            num_gpus = 1
        logger.info("Using %d GPUs.", num_gpus)
    else:
        num_gpus = 0

    # Assert that shard_root is not None when training
    assert cfg.data.shard_root is not None, "shard_root must not be None when training"

    dataset = activations.Dataset(cfg.data)
    saes, objectives, param_groups = make_saes([(c.sae, c.objective) for c in cfgs])
    init_b_dec_batched(saes, dataset)

    mode = "online" if cfg.track else "disabled"
    tags = [cfg.tag] if cfg.tag else []
    run = ParallelWandbRun(cfg.wandb_project, cfgs, mode, tags)

    optimizer = torch.optim.Adam(param_groups, fused=True)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.sae_batch_size,
        num_workers=cfg.n_workers,
        pin_memory=cfg.pin_memory,
        prefetch_factor=cfg.prefetch_factor,
        shuffle=cfg.shuffle,
    )

    dataloader = BatchLimiter(dataloader, cfg.n_patches)

    # Create learning rate schedulers based on config
    lr_schedulers = []
    for c in cfgs:
        if c.lr_scheduler == "warmup":
            lr_schedulers.append(Warmup(0.0, c.lr, c.n_lr_warmup))
        elif c.lr_scheduler == "cosine_warmup":
            lr_schedulers.append(CosineWarmup(
                init=0.0,
                final=c.lr,
                n_warmup_steps=c.n_lr_warmup,
                n_total_steps=len(dataloader),
                final_factor=c.lr_final_factor
            ))

    sparsity_schedulers = [
        Warmup(0.0, c.objective.sparsity_coeff, c.n_sparsity_warmup) for c in cfgs
    ]

    cfg_device_ids = [0] * len(cfgs)
    if num_gpus > 1:
        for i in range(len(cfgs)):
            cfg_device_ids[i] = (i * num_gpus) // len(cfgs)
            saes[i] = saes[i].to(f"{cfg.device}:{cfg_device_ids[i]}")
            objectives[i] = objectives[i].to(f"{cfg.device}:{cfg_device_ids[i]}")
    else:
        saes = saes.to(cfg.device)
        objectives = objectives.to(cfg.device)
    saes.train()
    objectives.train()

    # track active features
    act_freq_scores = [torch.zeros_like(sae.b_enc.data) for sae in saes]
    n_fwd_passes_since_fired = [torch.zeros_like(sae.b_enc.data) for sae in saes]

    global_step, n_patches_seen = 0, 0

    for batch in helpers.progress(dataloader, every=cfg.log_every):
        if num_gpus > 1:
            acts_BD = [batch["act"].to(f"{cfg.device}:{i}", non_blocking=True) for i in range(num_gpus)]
        else:
            acts_BD = [batch["act"].to(cfg.device, non_blocking=True)]
        for sae in saes:
            sae.normalize_w_dec()

        n_patches_seen += len(acts_BD[0])

        # Forward passes and loss calculations.
        losses = []
        for i in range(len(cfgs)):
            x_hat, f_x, h_pre, W_dec = saes[i](acts_BD[cfg_device_ids[i]])
            loss = objectives[i](
                acts_BD[cfg_device_ids[i]],
                f_x,
                x_hat,
                h_pre,
                W_dec,
                n_fwd_passes_since_fired[i],
            )
            losses.append(loss)

            with torch.no_grad():
                act_freq_fired = (f_x > 0).float().sum(0)
                did_fire = act_freq_fired > 0
                n_fwd_passes_since_fired[i] += 1
                n_fwd_passes_since_fired[i][did_fire] = 0

                # Calculate the sparsities, and add it to a list, calculate sparsity metrics
                act_freq_scores[i] += act_freq_fired

        with torch.no_grad():
            if (global_step + 1) % cfg.log_every == 0:
                feature_sparsity = [afs / n_patches_seen for afs in act_freq_scores]
                metrics = [
                    {
                        **loss.metrics(),
                        "progress/n_patches_seen": n_patches_seen,
                        "progress/learning_rate": group["lr"],
                        "progress/sparsity_coeff": objective.sparsity_coeff,
                        "sparsity/mean_passes_since_fired": passes_since_fired.mean().item(),
                        "sparsity/n_passes_since_fired_over_threshold": (
                            loss.dead_neuron_mask.sum().item()
                            if loss.dead_neuron_mask is not None
                            else 0
                        ),
                        "sparsity/below_1e-5": (feat_sparse < 1e-5).float().mean().item(),
                        "sparsity/below_1e-7": (feat_sparse < 1e-7).float().mean().item(),
                    }
                    for loss, objective, group, passes_since_fired, feat_sparse in zip(
                        losses,
                        objectives,
                        optimizer.param_groups,
                        n_fwd_passes_since_fired,
                        feature_sparsity,
                    )
                ]
                run.log(metrics, step=global_step)

                logger.info(
                    ", ".join(
                        f"{key}: {value:.5f}"
                        for key, value in losses[0].metrics().items()
                    )
                )

        for loss in losses:
            loss.loss.backward()

        # Apply gradient clipping if configured
        if cfg.gradient_clip_value > 0.0:
            # Clip gradients by global norm
            torch.nn.utils.clip_grad_norm_(
                parameters=[
                    p for group in optimizer.param_groups for p in group["params"]
                ],
                max_norm=cfg.gradient_clip_value,
            )

        for sae in saes:
            sae.remove_parallel_grads()

        optimizer.step()

        # Update LR and sparsity coefficients.
        for param_group, scheduler in zip(optimizer.param_groups, lr_schedulers):
            param_group["lr"] = scheduler.step()

        for objective, scheduler in zip(objectives, sparsity_schedulers):
            objective.sparsity_coeff = scheduler.step()

        # Don't need these anymore.
        optimizer.zero_grad()

        global_step += 1

    return saes, objectives, run, global_step


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class EvalMetrics:
    """Results of evaluating a trained SAE on a datset."""

    l0: float
    """Mean L0 across all examples."""
    l1: float
    """Mean L1 across all examples."""
    mse: float
    """Mean MSE across all examples."""
    n_dead: int
    """Number of neurons that never fired on any example."""
    n_almost_dead: int
    """Number of neurons that fired on fewer than `almost_dead_threshold` of examples."""
    n_dense: int
    """Number of neurons that fired on more than `dense_threshold` of examples."""

    freqs: Float[Tensor, " d_sae"]
    """How often each feature fired."""
    mean_values: Float[Tensor, " d_sae"]
    """The mean value for each feature when it did fire."""

    almost_dead_threshold: float
    """Threshold for an "almost dead" neuron."""
    dense_threshold: float
    """Threshold for a dense neuron."""

    def for_wandb(self) -> dict[str, int | float]:
        dct = dataclasses.asdict(self)
        # Store arrays as tables.
        dct["freqs"] = wandb.Table(columns=["freq"], data=dct["freqs"][:, None].numpy())
        dct["mean_values"] = wandb.Table(
            columns=["mean_value"], data=dct["mean_values"][:, None].numpy()
        )
        return {f"eval/{key}": value for key, value in dct.items()}


@beartype.beartype
@torch.no_grad()
def evaluate(
    cfgs: list[config.Train], saes: torch.nn.ModuleList, objectives: torch.nn.ModuleList
) -> list[EvalMetrics]:
    """
    Evaluates SAE quality by counting the number of dead features and the number of dense features.
    Also makes histogram plots to help human qualitative comparison.

    .. todo:: Develop automatic methods to use histogram and feature frequencies to evaluate quality with a single number.
    """

    torch.cuda.empty_cache()

    if len(split_cfgs(cfgs)) != 1:
        raise ValueError("Configs are not parallelizeable: {cfgs}.")

    cfg = cfgs[0]

    if torch.cuda.is_available():
        if cfg.span_all_devices:
            num_gpus = torch.cuda.device_count()
        else:
            num_gpus = 1
    else:
        num_gpus = 0

    cfg_device_ids = [0] * len(cfgs)
    if num_gpus > 1:
        for i in range(len(cfgs)):
            cfg_device_ids[i] = (i * num_gpus) // len(cfgs)
            saes[i] = saes[i].to(f"{cfg.device}:{cfg_device_ids[i]}")
    else:
        saes = saes.to(cfg.device)
    saes.eval()

    almost_dead_lim = 1e-7
    dense_lim = 1e-2

    # Create a copy of the data config for evaluation if eval_shard_root is specified
    if cfg.data.eval_shard_root is not None:
        logger.info(f"Using dedicated evaluation set from {cfg.data.eval_shard_root}")
        eval_data_cfg = dataclasses.replace(
            cfg.data,
            shard_root=cfg.data.eval_shard_root,
            shuffled=cfg.data.eval_shuffled,
        )
        dataset = activations.Dataset(eval_data_cfg)
    else:
        logger.info("Using training set for evaluation")
        dataset = activations.Dataset(cfg.data)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.sae_batch_size,
        num_workers=cfg.n_workers,
        pin_memory=cfg.pin_memory,
        prefetch_factor=cfg.prefetch_factor,
        shuffle=False,
    )

    n_fired = torch.zeros((len(cfgs), saes[0].cfg.d_sae))
    values = torch.zeros((len(cfgs), saes[0].cfg.d_sae))
    total_l0 = torch.zeros(len(cfgs))
    total_l1 = torch.zeros(len(cfgs))
    total_mse = torch.zeros(len(cfgs))

    for batch in helpers.progress(dataloader, desc="eval", every=cfg.log_every):
        if num_gpus > 1:
            acts_BD = [batch["act"].to(f"{cfg.device}:{i}", non_blocking=True) for i in range(num_gpus)]
        else:
            acts_BD = [batch["act"].to(cfg.device, non_blocking=True)]
        for i, (sae, objective) in enumerate(zip(saes, objectives)):
            x_hat_BD, f_x_BS, *_ = sae(acts_BD[cfg_device_ids[i]])
            loss = objective(acts_BD[cfg_device_ids[i]], f_x_BS, x_hat_BD)
            n_fired[i] += einops.reduce(f_x_BS > 0, "batch d_sae -> d_sae", "sum").cpu()
            values[i] += einops.reduce(f_x_BS, "batch d_sae -> d_sae", "sum").cpu()
            total_l0[i] += loss.l0.cpu()
            total_l1[i] += loss.l1.cpu()
            total_mse[i] += loss.mse.cpu()

    mean_values = values / n_fired
    freqs = n_fired / len(dataset)

    l0 = (total_l0 / len(dataloader)).tolist()
    l1 = (total_l1 / len(dataloader)).tolist()
    mse = (total_mse / len(dataloader)).tolist()

    n_dead = einops.reduce(freqs == 0, "n_saes d_sae -> n_saes", "sum").tolist()
    n_almost_dead = einops.reduce(
        freqs < almost_dead_lim, "n_saes d_sae -> n_saes", "sum"
    ).tolist()
    n_dense = einops.reduce(freqs > dense_lim, "n_saes d_sae -> n_saes", "sum").tolist()

    metrics = []
    for row in zip(l0, l1, mse, n_dead, n_almost_dead, n_dense, freqs, mean_values):
        metrics.append(EvalMetrics(*row, almost_dead_lim, dense_lim))

    return metrics


class BatchLimiter:
    """
    Limits the number of batches to only return `n_samples` total samples.
    """

    def __init__(self, dataloader: torch.utils.data.DataLoader, n_samples: int):
        self.dataloader = dataloader
        self.n_samples = n_samples
        self.batch_size = dataloader.batch_size

    def __len__(self) -> int:
        return self.n_samples // self.batch_size

    def __iter__(self):
        self.n_seen = 0
        while True:
            for batch in self.dataloader:
                yield batch

                # Sometimes we underestimate because the final batch in the dataloader might not be a full batch.
                self.n_seen += self.batch_size
                if self.n_seen > self.n_samples:
                    return

            # We try to mitigate the above issue by ignoring the last batch if we don't have drop_last.
            if not self.dataloader.drop_last:
                self.n_seen -= self.batch_size


#####################
# Parallel Training #
#####################


CANNOT_PARALLELIZE = set(
    [
        "data",
        "n_workers",
        "n_patches",
        "sae_batch_size",
        "track",
        "wandb_project",
        "tag",
        "log_every",
        "ckpt_path",
        "device",
        "span_all_devices",
        "slurm",
        "slurm_acct",
        "log_to",
        "sae.exp_factor",
        "sae.d_vit",
    ]
)


@beartype.beartype
def split_cfgs(cfgs: list[config.Train]) -> list[list[config.Train]]:
    """
    Splits configs into groups that can be parallelized.

    Arguments:
        A list of configs from a sweep file.

    Returns:
        A list of lists, where the configs in each sublist do not differ in any keys that are in `CANNOT_PARALLELIZE`. This means that each sublist is a valid "parallel" set of configs for `train`.
    """
    # Group configs by their values for CANNOT_PARALLELIZE keys
    groups = {}
    for cfg in cfgs:
        dct = dataclasses.asdict(cfg)

        # Create a key tuple from the values of CANNOT_PARALLELIZE keys
        key_values = []
        for key in sorted(CANNOT_PARALLELIZE):
            key_values.append((key, make_hashable(helpers.get(dct, key))))
        group_key = tuple(key_values)

        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(cfg)

    # Convert groups dict to list of lists
    return list(groups.values())


def make_hashable(obj):
    return json.dumps(obj, sort_keys=True)


##############
# Schedulers #
##############


@beartype.beartype
class Scheduler:
    def step(self) -> float:
        err_msg = f"{self.__class__.__name__} must implement step()."
        raise NotImplementedError(err_msg)

    def __repr__(self) -> str:
        err_msg = f"{self.__class__.__name__} must implement __repr__()."
        raise NotImplementedError(err_msg)


@beartype.beartype
class Warmup(Scheduler):
    """
    Linearly increases from `init` to `final` over `n_warmup_steps` steps.
    """

    def __init__(self, init: float, final: float, n_steps: int):
        self.final = final
        self.init = init
        self.n_steps = n_steps
        self._step = 0

    def step(self) -> float:
        self._step += 1
        if self._step < self.n_steps:
            return self.init + (self.final - self.init) * (self._step / self.n_steps)

        return self.final

    def __repr__(self) -> str:
        return f"Warmup(init={self.init}, final={self.final}, n_steps={self.n_steps})"


@beartype.beartype
class CosineWarmup(Scheduler):
    """
    Implements cosine annealing with warmup.
    First linearly increases from `init` to `final` over `n_warmup_steps` steps,
    then follows a cosine decay from `final` to `final_factor * final` over the remaining steps.
    """

    def __init__(self, init: float, final: float, n_warmup_steps: int, n_total_steps: int, final_factor: float = 0.0):
        self.init = init
        self.final = final
        self.n_warmup_steps = n_warmup_steps
        self.n_total_steps = n_total_steps
        self.final_factor = final_factor
        self._step = 0

    def step(self) -> float:
        self._step += 1
        
        # Warmup phase
        if self._step < self.n_warmup_steps:
            return self.init + (self.final - self.init) * (self._step / self.n_warmup_steps)
        
        # Cosine decay phase
        progress = (self._step - self.n_warmup_steps) / (self.n_total_steps - self.n_warmup_steps)
        progress = min(1.0, progress)  # Ensure progress doesn't exceed 1.0
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        return self.final_factor * self.final + (self.final - self.final_factor * self.final) * cosine_decay

    def __repr__(self) -> str:
        return f"CosineWarmup(init={self.init}, final={self.final}, n_warmup_steps={self.n_warmup_steps}, n_total_steps={self.n_total_steps}, final_factor={self.final_factor})"


def _plot_example_schedules():
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 6))

    n_steps = 1000
    xs = np.arange(n_steps)

    # Regular warmup scheduler
    warmup_schedule = Warmup(0.1, 0.9, 100)
    warmup_ys = []
    for _ in xs:
        warmup_ys.append(warmup_schedule.step())
    
    # Cosine warmup scheduler
    cosine_schedule = CosineWarmup(0.1, 0.9, 100, n_steps, 0.1)
    cosine_ys = []
    for _ in xs:
        cosine_ys.append(cosine_schedule.step())
    
    # Plot both schedulers
    ax.plot(xs, warmup_ys, label=str(warmup_schedule))
    ax.plot(xs, cosine_ys, label=str(cosine_schedule))
    
    ax.set_xlabel('Steps')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedulers Comparison')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    fig.tight_layout()
    fig.savefig("schedules.png")


if __name__ == "__main__":
    _plot_example_schedules()
