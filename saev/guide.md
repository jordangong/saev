# Guide to Training SAEs on Vision Models

1. Record ViT activations and save them to disk.
2. Train SAEs on the activations.
3. Visualize the learned features from the trained SAEs.
4. (your job) Propose trends and patterns in the visualized features.
5. (your job, supported by code) Construct datasets to test your hypothesized trends.
6. Confirm/reject hypotheses using `probing` package.

`saev` helps with steps 1, 2 and 3.

.. note:: `saev` assumes you are running on NVIDIA GPUs. On a multi-GPU system, prefix your commands with `CUDA_VISIBLE_DEVICES=X` to run on GPU X.

## Record ViT Activations to Disk

To save activations to disk, we need to specify:

1. Which model we would like to use
2. Which layers we would like to save.
3. Where on disk and how we would like to save activations.
4. Which images we want to save activations for.

The `saev.activations` module does all of this for us.

Run `uv run python -m saev activations --help` to see all the configuration.

In practice, you might run:

```sh
uv run python -m saev activations \
  --vit-family clip \
  --vit-ckpt ViT-B-32/openai \
  --d-vit 768 \
  --n-patches-per-img 49 \
  --vit-layers -2 \
  --dump-to /local/scratch/$USER/cache/saev \
  --n-patches-per-shard 2_4000_000 \
  data:imagenet-dataset
```

This will save activations for the CLIP-pretrained model ViT-B/32, which has a residual stream dimension of 768, and has 49 patches per image (224 / 32 = 7; 7 x 7 = 49).
It will save the second-to-last layer (`--layer -2`).
It will write 2.4M patches per shard, and save shards to a new directory `/local/scratch$USER/cache/saev`.


.. note:: A note on storage space: A ViT-B/16 will save 1.2M images x 197 patches/layer/image x 1 layer = ~240M activations, each of which take up 768 floats x 4 bytes/float = 3072 bytes, for a **total of 723GB** for the entire dataset. As you scale to larger models (ViT-L has 1024 dimensions, 14x14 patches are 224 patches/layer/image), recorded activations will grow even larger.

This script will also save a `metadata.json` file that will record the relevant metadata for these activations, which will be read by future steps.
The activations will be in `.bin` files, numbered starting from 000000.

To add your own models, see the guide to extending in `saev.activations`.

## Train SAEs on Activations

To train an SAE, we need to specify:

1. Which activations to use as input.
2. SAE architectural stuff.
3. Optimization-related stuff.

`The `saev.training` module handles this.

Run `uv run python -m saev train --help` to see all the configuration.

Continuing on from our example before, you might want to run something like:

```sh
uv run python -m saev train \
  --data.shard-root /local/scratch/$USER/cache/saev/ac89246f1934b45e2f0487298aebe36ad998b6bd252d880c0c9ec5de78d793c8 \
  --data.layer -2 \
  --data.patches patches \
  --data.no-scale-mean \
  --data.no-scale-norm \
  --sae.d-vit 768 \
  --lr 5e-4
```

`--data.*` flags describe which activations to use.

`--data.shard-root` should point to a directory with `*.bin` files and the `metadata.json` file.
`--data.layer` specifies the layer, and `--data.patches` says that want to train on individual patch activations, rather than the [CLS] token activation.
`--data.no-scale-mean` and `--data.no-scale-norm` mean not to scale the activation mean or L2 norm.
Anthropic's and OpenAI's papers suggest normalizing these factors, but `saev` still has a bug with this, so I suggest not scaling these factors.

`--sae.*` flags are about the SAE itself.

`--sae.d-vit` is the only one you need to change; the dimension of our ViT was 768 for a ViT-B, rather than the default of 1024 for a ViT-L.

Finally, choose a slightly larger learning rate than the default with `--lr 5e-4`.

This will train one (1) sparse autoencoder on the data.
See the section on sweeps to learn how to train multiple SAEs in parallel using only a single GPU.

## Visualize the Learned Features

Now that you've trained an SAE, you probably want to look at its learned features.
One way to visualize an individual learned feature \(f\) is by picking out images that maximize the activation of feature \(f\).
Since we train SAEs on patch-level activations, we try to find the top *patches* for each feature \(f\).
Then, we pick out the images those patches correspond to and create a heatmap based on SAE activation values.

.. note:: More advanced forms of visualization are possible (and valuable!), but should not be included in `saev` unless they can be applied to every SAE/dataset combination. If you have specific visualizations, please add them to `contrib/` or another location.

`saev.visuals` records these maximally activating images for us.
You can see all the options with `uv run python -m saev visuals --help`.

So you might run:

```sh
uv run python -m saev visuals \
  --ckpt checkpoints/abcdefg/sae.pt \
  --dump-to /nfs/$USER/saev/webapp/abcdefg \
  --data.shard-root /local/scratch/$USER/cache/saev/ac89246f1934b45e2f0487298aebe36ad998b6bd252d880c0c9ec5de78d793c8 \
  --data.layer -2 \
  --data.patches patches \
  images:imagenet-dataset
```

This will record the top 128 patches, and then save the unique images among those top 128 patches for each feature in the trained SAE.
It will cache these best activations to disk, then start saving images to visualize later on.

`saev.interactive.features` is a small web application based on [marimo](https://marimo.io/) to interactively look at these images.

You can run it with `uv run marimo edit saev/interactive/features.py`.


## Sweeps

.. todo:: Explain how to run grid sweeps.

## Training Metrics and Visualizations

.. todo:: Explain how to use the `saev.interactive.metrics` notebook.
