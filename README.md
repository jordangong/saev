# saev - Sparse Auto-Encoders for Vision

![Coverage](docs/coverage.svg)

Implementation of sparse autoencoders (SAEs) for vision transformers (ViTs) in PyTorch.

This is the codebase used for our preprint "Sparse Autoencoders for Scientifically Rigorous Interpretation of Vision Models"

* [arXiv preprint](https://arxiv.org/abs/2502.06755)
* [Huggingface Models](https://huggingface.co/collections/osunlp/sae-v-67ab8c4fdf179d117db28195)
* [API Docs](https://osu-nlp-group.github.io/SAE-V/saev)
* [Demos](https://osu-nlp-group.github.io/SAE-V/#demos)

## About

saev is a package for training sparse autoencoders (SAEs) on vision transformers (ViTs) in PyTorch.
It also includes an interactive webapp for looking through a trained SAE's features.

Originally forked from [HugoFry](https://github.com/HugoFry/mats_sae_training_for_ViTs) who forked it from [Joseph Bloom](https://github.com/jbloomAus/SAELens).

Read [logbook.md](logbook.md) for a detailed log of my thought process.

See [related-work.md](saev/related-work.md) for a list of works training SAEs on vision models.
Please open an issue or a PR if there is missing work.

## Installation

Installation is supported with [uv](https://docs.astral.sh/uv/).
saev will likely work with pure pip, conda, etc. but I will not formally support it.

To install, clone this repository (maybe fork it first if you want).

In the project root directory, run `uv run python -m saev --help`.
The first invocation should create a virtual environment and show a help message.

## Using `saev`

See the [docs](https://osu-nlp-group.github.io/SAE-V/saev) for an overview.

I recommend using the [llms.txt](https://osu-nlp-group.github.io/SAE-V/llms.txt) file as a way to use any LLM provider to ask questions.
For example, you can run `curl https://osu-nlp-group.github.io/SAE-V/llms.txt | pbcopy` on macOS to copy the text, then paste it into [https://claude.ai](https://claude.ai) and ask any question you have.
