import numpy as np


def estimate_table_depth(depth_image):
    """
    Estimate the table surface depth using histogram analysis.
    The table is the largest flat surface — its depth appears most often.
    """

    valid = depth_image[depth_image > 0].flatten()

    if len(valid) == 0:
        return 0

    hist, edges = np.histogram(valid, bins=100)

    peak_idx = np.argmax(hist)

    table_depth = (edges[peak_idx] + edges[peak_idx + 1]) / 2

    return table_depth


def filter_masks_by_depth(
    masks,
    depth_image,
    table_depth_ratio=0.7,
    zero_depth_ratio=0.5,
    max_depth_std=120,
    table_tolerance=15
):
    """
    Filter SAM masks using depth information.

    Removes:
    1. Table/background masks (most pixels at table depth)
    2. Invalid masks (most pixels have zero depth)
    3. Incoherent masks (high depth variance = multiple objects)

    Args:
        masks: list of SAM mask dicts
        depth_image: uint16 depth image (same size as RGB)
        table_depth_ratio: if this fraction of mask pixels match
                           table depth, discard the mask
        zero_depth_ratio: if this fraction of mask pixels have
                          depth=0, discard the mask
        max_depth_std: max allowed std-dev of depth within mask
        table_tolerance: mm tolerance for table depth matching

    Returns:
        filtered list of masks
    """

    table_depth = estimate_table_depth(depth_image)

    print(f"Estimated table depth: {table_depth:.0f} mm")

    filtered = []

    for mask in masks:

        seg = mask['segmentation']

        mask_depths = depth_image[seg]

        total_pixels = len(mask_depths)

        if total_pixels == 0:
            continue

        # --- Check 1: Too many zero-depth pixels ---

        zero_count = np.sum(mask_depths == 0)
        zero_ratio = zero_count / total_pixels

        if zero_ratio > zero_depth_ratio:
            continue

        # --- Check 2: Table/background mask ---

        valid_depths = mask_depths[mask_depths > 0]

        if len(valid_depths) == 0:
            continue

        table_match = np.sum(
            np.abs(valid_depths.astype(float) - table_depth)
            < table_tolerance
        )

        table_ratio = table_match / len(valid_depths)

        if table_ratio > table_depth_ratio:
            continue

        # --- Check 3: Depth coherence ---

        depth_std = np.std(valid_depths)

        if depth_std > max_depth_std:
            continue

        filtered.append(mask)

    return filtered
