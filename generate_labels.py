"""
End-to-end dexterous grasp label generation pipeline.

Implements the full framework from:
  "From-scratch dexterous grasp type annotation with SAM and lightweight
   vision-language models", Wang & Cheng, Pattern Recognition Letters 2026.

Pipeline:
  1. Load RGB-D images and resize to 1280×720.
  2. SAM automatic segmentation.
  3. Post-process masks: small-area removal → IoU overlap deduplication
     → complementary set computation (Section 3.2).
  4. (Optional) depth-based filtering to remove table/background masks.
  5. Save individual RGB + depth crops and a colourised scene overlay.
  6. VLM pipeline — four sequential submodules, each with majority voting
     and a judgment verification step (Section 3.3):
       SM1: object vs. background  → [True/False]
       SM2: object count           → [number]
       SM3: object category        → [label]
       SM4: grasp type             → [grasp type]
  7. Save results to outputs/labels/labels.json.

VLM used: Qwen2.5-VL-3B-Instruct (for both main and judgment roles).
"""

import os
import json
import cv2
import numpy as np

from sam_module.sam_segment import SAMSegmenter

from utils.mask_postprocess import postprocess_masks
from utils.depth_filter import filter_masks_by_depth
from utils.object_crop import save_object_crops
from utils.visualize import generate_scene_overlay

from vlm_module.qwen_grasp import QwenGraspModel
from vlm_module.pipeline import (
    run_submodule,
    GRASP_TYPES,
    SM1_MAIN, SM1_JUDGE,
    SM2_MAIN, SM2_JUDGE,
    SM3_MAIN, SM3_JUDGE,
    SM4_MAIN, SM4_JUDGE,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# VLM inference rounds per submodule (paper best config: 11 decisions, 4 retries)
N_DECISIONS = 11
N_RETRIES   = 4

# SAM mask post-processing thresholds
# small_area: ~0.5% of 1280x720 = 4608 px  (256 is meaningless at this resolution)
# max_area:   ~27% of image     = 245760 px (removes table/wall slabs that
#                                             slip past the depth filter)
# overlap_iou: 0.4 catches more duplicates than the paper's 0.7
# complementary_iou_low: 0.1 handles subtle partial overlaps
# containment: 0.8 catches half-mask duplicates (low IoU but high containment)
SMALL_AREA_THRESHOLD  = 5000
MAX_AREA_THRESHOLD    = 245760
OVERLAP_IOU_THRESHOLD = 0.4
COMPLEMENTARY_IOU_LOW = 0.1
CONTAINMENT_THRESHOLD = 0.8

# Input images
RGB_PATH   = "data/realsense/rgb/0000.png"
DEPTH_PATH = "data/realsense/depth/0000.png"

# Output paths
CROPS_DIR       = "outputs/crops"
DEPTH_CROPS_DIR = "outputs/crops/depth"
LABELS_DIR      = "outputs/labels"
SCENE_OVERLAY   = "outputs/scene_overlay.png"


# =============================================================================
# 1. LOAD & RESIZE IMAGES
# =============================================================================

image = cv2.imread(RGB_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (1280, 720))          # paper: resize to 1280×720

depth = cv2.imread(DEPTH_PATH, cv2.IMREAD_UNCHANGED)
depth = cv2.resize(depth, (1280, 720), interpolation=cv2.INTER_NEAREST)

print(f"RGB  : {image.shape}")
print(f"Depth: {depth.shape}  dtype={depth.dtype}")


# =============================================================================
# 2. SAM SEGMENTATION
# =============================================================================

segmenter = SAMSegmenter(checkpoint_path="sam_vit_l_0b3195.pth")
raw_masks = segmenter.generate_masks(image)
print(f"\nRaw SAM masks: {len(raw_masks)}")


# =============================================================================
# 3. PRELIMINARY AREA FILTER  (remove obvious noise before depth analysis)
# =============================================================================

masks = [m for m in raw_masks if m['area'] >= SMALL_AREA_THRESHOLD]
print(f"After area pre-filter : {len(masks)}")


# =============================================================================
# 4. DEPTH-BASED FILTERING  (run FIRST to remove the table surface mask)
#
# Removing the large table mask before IoU post-processing is critical:
# if the table mask is present during IoU comparisons, object masks that
# partially overlap the table get their complement computed against the table
# region, fragmenting them into inaccurate sub-masks.
# =============================================================================

masks = filter_masks_by_depth(masks, depth)
print(f"After depth filter    : {len(masks)}")


# =============================================================================
# 5. IoU / CONTAINMENT POST-PROCESSING  (Section 3.2)
#
# Now that background / table masks are gone, IoU deduplication operates
# only on genuine object masks, giving much cleaner results.
# =============================================================================

masks = postprocess_masks(
    masks,
    small_area_threshold=SMALL_AREA_THRESHOLD,
    max_area_threshold=MAX_AREA_THRESHOLD,
    overlap_iou_threshold=OVERLAP_IOU_THRESHOLD,
    complementary_iou_low=COMPLEMENTARY_IOU_LOW,
    containment_threshold=CONTAINMENT_THRESHOLD,
)
print(f"After IoU postprocess : {len(masks)}")


# =============================================================================
# 6. SAVE CROPS AND SCENE OVERLAY
# =============================================================================

# Individual RGB + depth crops
save_object_crops(image, masks, depth_image=depth)

# Colourised scene overlay — passed to every VLM call as global scene context
overlay = generate_scene_overlay(image, masks)
os.makedirs("outputs", exist_ok=True)
cv2.imwrite(SCENE_OVERLAY, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
print(f"Scene overlay saved: {SCENE_OVERLAY}")


# =============================================================================
# 7. LOAD VLM
# =============================================================================

model = QwenGraspModel()


# =============================================================================
# 8. VLM LABEL GENERATION  (Section 3.3 — four submodules)
# =============================================================================

crop_files = sorted([f for f in os.listdir(CROPS_DIR) if f.endswith(".png")])

results = []

for crop_file in crop_files:

    crop_path       = os.path.join(CROPS_DIR, crop_file)
    depth_crop_path = os.path.join(DEPTH_CROPS_DIR, crop_file)

    print(f"\n{'='*60}")
    print(f"Processing: {crop_file}")

    # ------------------------------------------------------------------
    # Submodule 1: object vs. background
    # ------------------------------------------------------------------
    is_object = run_submodule(
        model, SCENE_OVERLAY, crop_path,
        SM1_MAIN, SM1_JUDGE,
        n_decisions=N_DECISIONS, n_retries=N_RETRIES,
    )
    print(f"  SM1 (object?): [{is_object}]")

    if is_object is None or is_object.lower() == "false":
        print("  → Skipped (identified as background or no valid response)")
        continue

    # ------------------------------------------------------------------
    # Submodule 2: object count in the region
    # ------------------------------------------------------------------
    obj_count = run_submodule(
        model, SCENE_OVERLAY, crop_path,
        SM2_MAIN, SM2_JUDGE,
        n_decisions=N_DECISIONS, n_retries=N_RETRIES,
    )
    print(f"  SM2 (count):   [{obj_count}]")

    # ------------------------------------------------------------------
    # Submodule 3: object semantic category
    # ------------------------------------------------------------------
    obj_label = run_submodule(
        model, SCENE_OVERLAY, crop_path,
        SM3_MAIN, SM3_JUDGE,
        n_decisions=N_DECISIONS, n_retries=N_RETRIES,
    )
    if obj_label is None:
        obj_label = "unknown"
    print(f"  SM3 (object):  [{obj_label}]")

    # ------------------------------------------------------------------
    # Submodule 4: grasp type (conditioned on object label from SM3)
    # ------------------------------------------------------------------
    sm4_main  = SM4_MAIN.format(object_label=obj_label)
    sm4_judge = SM4_JUDGE.format(object_label=obj_label)

    grasp_label = run_submodule(
        model, SCENE_OVERLAY, crop_path,
        sm4_main, sm4_judge,
        n_decisions=N_DECISIONS, n_retries=N_RETRIES,
    )
    print(f"  SM4 (grasp):   [{grasp_label}]")

    results.append({
        "crop":        crop_file,
        "object_count": obj_count,
        "object":      obj_label,
        "grasp":       grasp_label,
        "depth_crop":  depth_crop_path if os.path.exists(depth_crop_path) else None,
    })


# =============================================================================
# 9. SAVE LABELS
# =============================================================================

os.makedirs(LABELS_DIR, exist_ok=True)
label_path = os.path.join(LABELS_DIR, "labels.json")

with open(label_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"\nLabels saved → {label_path}")
print(f"Total labeled objects: {len(results)}")
