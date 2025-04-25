import functools
import logging

import beartype
import torchvision.datasets
from PIL import Image

from .. import activations, config

logger = logging.getLogger("app.data")


@functools.cache
def get_datasets():
    datasets = {
        "inat21__train_mini": torchvision.datasets.ImageFolder(
            root="/research/nfs_su_809/workspace/stevens.994/datasets/inat21/train_mini/"
        ),
        "imagenet__train": activations.ImageNet(config.ImagenetDataset()),
    }
    logger.info("Loaded datasets.")
    return datasets


@beartype.beartype
def get_img_raw(key: str, i: int) -> tuple[Image.Image, str]:
    """
    Get raw image and processed label from dataset.

    Returns:
        Tuple of Image.Image and classname.
    """
    dataset = get_datasets()[key]
    sample = dataset[i]
    # iNat21 specific: Remove taxonomy prefix
    label = " ".join(sample["label"].split("_")[1:])
    return sample["image"], label


def to_sized(
    img_raw: Image.Image, min_px: int, crop_px: tuple[int, int]
) -> Image.Image:
    """Convert raw vips image to standard model input size (resize + crop)."""
    # Calculate scale factor to make smallest dimension = min_px
    scale = min_px / min(img_raw.width, img_raw.height)

    # Resize maintaining aspect ratio
    img_raw = img_raw.resize(scale)
    assert min(img_raw.width, img_raw.height) == min_px

    # Calculate crop coordinates to center crop
    left = (img_raw.width - crop_px[0]) // 2
    top = (img_raw.height - crop_px[1]) // 2

    # Crop to final size
    return img_raw.crop(left, top, crop_px[0], crop_px[1])


@beartype.beartype
def img_to_b64(img: Image.Image) -> str:
    raise NotImplementedError()
