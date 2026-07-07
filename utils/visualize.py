import matplotlib.pyplot as plt
import numpy as np


def generate_scene_overlay(image, masks, alpha=0.4):
    """
    Generate a colorized segmentation overlay on the RGB image.

    Each mask is painted with a distinct random color blended with the
    original image.  The result is used as the 'overall scene' image that
    the VLM receives alongside each individual crop (Section 3.3).

    Args:
        image: uint8 RGB image (H, W, 3).
        masks: list of SAM mask dicts (must have 'segmentation' key).
        alpha: blending weight for the mask colour (0 = transparent, 1 = solid).

    Returns:
        uint8 RGB image with coloured mask overlays.
    """
    overlay = image.copy().astype(np.float32)
    rng = np.random.default_rng(seed=42)   # reproducible colours

    for mask in masks:
        seg = mask['segmentation']
        colour = rng.integers(60, 220, size=3).astype(np.float32)
        overlay[seg] = overlay[seg] * (1.0 - alpha) + colour * alpha

    return overlay.clip(0, 255).astype(np.uint8)


def show_masks(image, masks):

    plt.figure(figsize=(10,10))

    plt.imshow(image)

    for mask in masks:

        segmentation = mask['segmentation']

        color = np.random.random(3)

        overlay = np.zeros((*segmentation.shape, 3))

        overlay[segmentation] = color

        plt.imshow(overlay, alpha=0.25)

    plt.axis("off")

    plt.show()