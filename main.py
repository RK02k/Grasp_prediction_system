"""
SINGLE-FRAME DEMO  (data/realsense/rgb/0000.png)
================================================
This script is a quick sanity-check for ONE frame.

For the full dataset pipelines run:
  python generate_labels.py     → processes all 256 realsense frames
  python run_train1_pipeline.py → processes all 30 train_1 scenes
  python test_pipeline.py       → validates both datasets
"""

import cv2

from sam_module.sam_segment import SAMSegmenter

from utils.visualize import show_masks
from utils.mask_filter import filter_small_masks
from utils.depth_filter import filter_masks_by_depth
from utils.object_crop import save_object_crops


# =========================
# LOAD RGB IMAGE
# =========================

rgb_path = "data/realsense/rgb/0000.png"

image = cv2.imread(rgb_path)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# =========================
# LOAD DEPTH IMAGE
# =========================

depth_path = "data/realsense/depth/0000.png"

depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

print("Depth shape:", depth.shape, "dtype:", depth.dtype)


# =========================
# LOAD SAM
# =========================

segmenter = SAMSegmenter(
    checkpoint_path="sam_vit_l_0b3195.pth"
)


# =========================
# GENERATE MASKS
# =========================

masks = segmenter.generate_masks(image)

print("Original masks:", len(masks))


# =========================
# FILTER SMALL MASKS
# =========================

masks = filter_small_masks(
    masks,
    min_area=8000
)

print("After area filter:", len(masks))


# =========================
# FILTER BY DEPTH
# =========================

masks = filter_masks_by_depth(
    masks,
    depth
)

print("After depth filter:", len(masks))


# =========================
# VISUALIZE
# =========================

show_masks(image, masks)


# =========================
# SAVE CROPS
# =========================

save_object_crops(image, masks, depth_image=depth)