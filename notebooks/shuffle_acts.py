import glob
import os
import shutil
import sys

import numpy as np
import torch
import tqdm

from saev import activations, config

source_shard_dir = sys.argv[1]
target_shard_dir = sys.argv[2]
layer = -2 if len(sys.argv) == 3 else sys.argv[3]
try:
    layer = int(layer)
except ValueError:
    pass

source_shard_dirs = sorted(glob.glob(os.path.join(source_shard_dir, "*/*")))
target_shard_dirs = []
for source_shard_dir in source_shard_dirs:
    source_shard_dir_head, source_shard_dir_tail = os.path.split(source_shard_dir)
    target_shard_dir_split = [*os.path.split(source_shard_dir_head), source_shard_dir_tail]
    target_shard_dir_split[0] = target_shard_dir
    target_shard_dir_split[-2] += "_rand"
    target_shard_dir = os.path.join(*target_shard_dir_split)
    target_shard_dirs.append(target_shard_dir)


class PermuteSampler(torch.utils.data.Sampler[int]):
    def __init__(self, dataset: activations.Dataset, seed: int = 42):
        self.perm = np.random.default_rng(seed=seed).permutation(len(dataset)).tolist()

    def __len__(self):
        return len(self.perm)

    def __iter__(self):
        return iter(self.perm)


activations.Dataset.transform = lambda self, x: x

for source_shard_dir, target_shard_dir in zip(
    source_shard_dirs, target_shard_dirs
):
    if not os.path.exists(target_shard_dir):
        os.makedirs(target_shard_dir)
        shutil.copy2(
            os.path.join(source_shard_dir, "metadata.json"), target_shard_dir
        )

    source_dataset_config = config.DataLoad(
        shard_root=source_shard_dir,
        patches="all",
        layer=layer,
        scale_mean=False,
        scale_norm=False,
    )
    source_dataset = activations.Dataset(source_dataset_config)
    permute_sampler = PermuteSampler(source_dataset)
    n_imgs_per_shard = (
        source_dataset.metadata.n_patches_per_shard
        // len(source_dataset.metadata.layers)
        // (source_dataset.metadata.n_patches_per_img + 1)
    )
    n_examples_per_shard = n_imgs_per_shard * (
        source_dataset.metadata.n_patches_per_img + 1
    )
    shard_shape = (
        n_imgs_per_shard,
        len(source_dataset.metadata.layers),
        source_dataset.metadata.n_patches_per_img + 1,
        source_dataset.metadata.d_vit,
    )
    source_dataloader = torch.utils.data.DataLoader(
        source_dataset,
        batch_size=(source_dataset.metadata.n_patches_per_img + 1),
        sampler=permute_sampler,
        num_workers=8,
    )

    shard_id = 0
    num_batches = len(source_dataloader)
    for i, batch in enumerate(tqdm.tqdm(source_dataloader)):
        if i % n_imgs_per_shard == 0:
            acts_fpath = os.path.join(target_shard_dir, f"acts{shard_id:06}.bin")
            acts = np.memmap(acts_fpath, mode="w+", dtype=np.float32, shape=shard_shape)
            acts = acts[:, source_dataset.layer_index, :]

        acts[i % n_imgs_per_shard] = batch["act"].numpy()[:]

        if (i + 1) % 1000 == 0 or i == num_batches - 1:
            acts.flush()

        if (i + 1) % n_imgs_per_shard == 0:
            acts.flush()
            shard_id += 1
