import glob
import os
import sys

import numpy as np
import tqdm

from saev import activations, config

source_shard_root = sys.argv[1]
target_shard_root = sys.argv[2]

source_shards_roots = sorted(glob.glob(os.path.join(source_shard_root, "*/*")))
target_shards_roots = []
for source_shard_root in source_shards_roots:
    source_shard_root_split = os.path.split(source_shard_root)
    target_shard_root_split = list(source_shard_root_split)
    target_shard_root_split[:-2] = [target_shard_root]
    target_shard_root_split[-2] += "_rand"
    target_shard_root = os.path.join(*target_shard_root_split)
    target_shards_roots.append(target_shard_root)

def get_writable_patch(self, i: int) -> activations.Dataset.Example:
    n_imgs_per_shard = (
        self.metadata.n_patches_per_shard
        // len(self.metadata.layers)
        // (self.metadata.n_patches_per_img + 1)
    )
    n_examples_per_shard = n_imgs_per_shard * (self.metadata.n_patches_per_img + 1)

    shard = i // n_examples_per_shard
    pos = i % n_examples_per_shard

    acts_fpath = os.path.join(self.cfg.shard_root, f"acts{shard:06}.bin")
    shape = (
        n_imgs_per_shard,
        len(self.metadata.layers),
        self.metadata.n_patches_per_img + 1,
        self.metadata.d_vit,
    )
    acts = np.memmap(acts_fpath, mode="w+", dtype=np.float32, shape=shape)
    # Choose the layer
    acts = acts[:, self.layer_index, :]

    # Choose an image and token (including CLS)
    img_idx = pos // (self.metadata.n_patches_per_img + 1)
    token_idx = pos % (self.metadata.n_patches_per_img + 1)
    act = acts[img_idx, token_idx]

    # For token_idx=0, this is the CLS token, otherwise it's a patch
    is_cls = token_idx == 0
    patch_i = -1 if is_cls else token_idx - 1

    return self.Example(
        act=self.transform(act),
        image_i=i // (self.metadata.n_patches_per_img + 1),
        patch_i=patch_i,
    )


activations.Dataset.transform = lambda self, x: x
activations.Dataset.get_writable_patch = get_writable_patch

for source_shard_root, target_shard_root in zip(
    source_shards_roots, target_shards_roots
):
    if not os.path.exists(target_shard_root):
        os.makedirs(target_shard_root)

    source_dataset_config = config.DataLoad(
        shard_root=source_shard_root,
        patches="all",
        scale_mean=False,
        scale_norm=False,
    )
    target_dataset_config = config.DataLoad(
        shard_root=target_shard_root,
        patches="all",
        scale_mean=False,
        scale_norm=False,
    )
    source_dataset = activations.Dataset(source_dataset_config)
    target_dataset = activations.Dataset(target_dataset_config)

    perm = np.random.default_rng(seed=42).permutation(len(source_dataset))
    for i, j in tqdm.tqdm(enumerate(perm), total=len(perm)):
        act_i = source_dataset[i]["act"]
        act_j = target_dataset.get_writable_patch(j)["act"]
        act_j[:] = act_i[:]
        act_j.flush()
