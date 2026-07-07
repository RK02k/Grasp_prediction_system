"""
Hierarchical crop splitter — Phase 1 fix.

When SAM produces a bounding-box crop that contains more than one object
(SM2 > 1), this module re-runs SAM *inside* that sub-image to obtain
individual per-object masks, then returns each object as a separate crop
saved to disk.  The caller can then feed each new crop through SM1–SM4
independently, effectively doubling recall on merged-mask scenes.

Algorithm
---------
1. Extract the sub-image using the parent mask's bounding box.
2. Run SAM automatic mask generator on the sub-image with a fine grid
   (points_per_side=32) to resolve tightly packed objects.
3. Apply area / IoU post-processing on the resulting sub-masks.
4. Save each sub-mask crop to ``<crop_basename>_sub<idx>.png``.
5. Return a list of (sub_crop_path, sub_mask_dict) tuples so the caller
   can create scene-context overlays or continue with VLM stages.

Only the RGB sub-image is required; depth cropping is handled optionally.
"""

import os

import cv2
import numpy as np

from utils.mask_postprocess import postprocess_masks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def split_crop_with_sam(
    segmenter,
    full_image: np.ndarray,
    parent_mask: dict,
    crop_save_path: str,
    depth_image: np.ndarray | None = None,
    min_sub_area: int = 2000,
    max_area_ratio: float = 0.85,
    overlap_iou: float = 0.4,
    containment: float = 0.8,
) -> list[tuple[str, dict]]:
    """
    Re-segment a multi-object crop and save individual sub-crops.

    Parameters
    ----------
    segmenter       : SAMSegmenter instance (already loaded).
    full_image      : H×W×3 uint8 RGB image of the full scene.
    parent_mask     : SAM mask dict for the merged crop (must contain
                      ``bbox`` as [x, y, w, h]).
    crop_save_path  : Absolute path of the *existing* parent crop PNG.
                      Sub-crops will be saved alongside it as
                      ``<stem>_sub0.png``, ``<stem>_sub1.png``, …
    depth_image     : Optional H×W depth array (same size as full_image).
    min_sub_area    : Minimum pixel area for a sub-mask to be kept.
    max_area_ratio  : Sub-masks covering > this fraction of the sub-image
                      are treated as background and discarded.
    overlap_iou     : IoU threshold for duplicate removal among sub-masks.
    containment     : Containment threshold for duplicate removal.

    Returns
    -------
    List of (sub_crop_path, sub_mask_dict_in_original_coords) tuples.
    Empty list if re-segmentation produces ≤ 1 valid sub-mask (caller
    should fall back to the original crop in that case).
    """
    x, y, w, h = [int(v) for v in parent_mask["bbox"]]

    # --- guard against degenerate boxes ---
    if w < 20 or h < 20:
        return []

    sub_image = full_image[y: y + h, x: x + w].copy()
    sub_area  = sub_image.shape[0] * sub_image.shape[1]

    # --- run SAM on sub-image ---
    raw_sub_masks = segmenter.generate_masks(sub_image)

    # --- filter by area ---
    max_sub_area = int(sub_area * max_area_ratio)
    sub_masks = [
        m for m in raw_sub_masks
        if min_sub_area <= m["area"] <= max_sub_area
    ]

    if not sub_masks:
        return []

    # --- post-process (IoU dedup + containment) ---
    sub_masks = postprocess_masks(
        sub_masks,
        small_area_threshold=min_sub_area,
        max_area_threshold=max_sub_area,
        overlap_iou_threshold=overlap_iou,
        containment_threshold=containment,
    )

    if len(sub_masks) <= 1:
        # Re-segmentation didn't meaningfully split — not useful
        return []

    # --- save sub-crops and remap masks to original image coordinates ---
    stem, ext = os.path.splitext(crop_save_path)
    depth_dir = os.path.join(os.path.dirname(crop_save_path), "depth")
    results: list[tuple[str, dict]] = []

    for idx, sm in enumerate(sub_masks):
        sx, sy, sw, sh = [int(v) for v in sm["bbox"]]

        # crop within the sub-image
        sub_crop = sub_image[sy: sy + sh, sx: sx + sw].copy()
        sub_seg  = sm["segmentation"][sy: sy + sh, sx: sx + sw]

        isolated = np.zeros_like(sub_crop)
        isolated[sub_seg] = sub_crop[sub_seg]

        save_path = f"{stem}_sub{idx}{ext}"
        cv2.imwrite(save_path, cv2.cvtColor(isolated, cv2.COLOR_RGB2BGR))

        # --- optional depth sub-crop ---
        if depth_image is not None and os.path.isdir(depth_dir):
            depth_sub = depth_image[y: y + h, x: x + w]
            depth_crop = depth_sub[sy: sy + sh, sx: sx + sw].copy()
            depth_crop[~sub_seg] = 0
            depth_save = os.path.join(depth_dir, os.path.basename(save_path))
            cv2.imwrite(depth_save, depth_crop)

        # remap bounding-box to original image coordinates
        orig_bbox = [x + sx, y + sy, sw, sh]

        # remap segmentation mask to original image size
        orig_seg = np.zeros(full_image.shape[:2], dtype=bool)
        orig_seg[y + sy: y + sy + sh, x + sx: x + sx + sw] = sub_seg

        remapped = dict(sm)
        remapped["bbox"]         = orig_bbox
        remapped["segmentation"] = orig_seg
        remapped["area"]         = int(orig_seg.sum())

        results.append((save_path, remapped))

    return results
