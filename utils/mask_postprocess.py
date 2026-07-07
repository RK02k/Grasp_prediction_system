import numpy as np
import cv2


def _recompute_bbox(seg):
    rows = np.any(seg, axis=1)
    cols = np.any(seg, axis=0)
    if not rows.any():
        return [0, 0, 0, 0]
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [int(cmin), int(rmin), int(cmax - cmin + 1), int(rmax - rmin + 1)]


def _keep_largest_component(seg):
    """
    Return a boolean mask containing only the largest connected component.

    When SAM generates a segmentation that spans two spatially disconnected
    objects, this strips away the smaller disconnected fragment so that each
    output mask represents a single contiguous region.
    """
    seg_uint8 = seg.astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(seg_uint8)
    if num_labels <= 2:           # 0=background + at most 1 component
        return seg
    best = max(range(1, num_labels), key=lambda l: int((labels == l).sum()))
    return labels == best


def postprocess_masks(
    masks,
    small_area_threshold=256,
    max_area_threshold=None,
    overlap_iou_threshold=0.4,
    complementary_iou_low=0.1,
    containment_threshold=0.8,
):
    """
    Post-process SAM masks following Section 3.2 of the paper, with additional
    containment-based deduplication.

    Steps:
    1. Remove masks with area < small_area_threshold (fragments).
    2. Remove masks with area > max_area_threshold (oversized background).
    3. Sort largest-first so bigger, more complete masks win deduplication.
    4. For each mask compare against all already-kept masks:

       a. High IoU (> overlap_iou_threshold) OR high containment
          max(intersection/area_A, intersection/area_B) > containment_threshold
          -> discard current mask (same object).

       b. Moderate IoU (complementary_iou_low < IoU <= overlap_iou_threshold)
          -> replace current mask with its non-overlapping complement.
          -> if complement too small -> discard.

    The containment check catches cases where two masks represent the same
    object but have low global IoU because one covers a sub-region of the
    other (e.g. two SAM samples of the same banana with different extents).
    """

    # Step 1 & 2: area bounds
    masks = [m for m in masks if m['area'] >= small_area_threshold]
    if max_area_threshold is not None:
        masks = [m for m in masks if m['area'] <= max_area_threshold]

    # Step 2b: keep only the largest connected component per mask.
    # This strips disconnected fragments so each mask represents one object.
    trimmed = []
    for m in masks:
        new_seg  = _keep_largest_component(m['segmentation'])
        new_area = int(new_seg.sum())
        if new_area >= small_area_threshold:
            nm                = dict(m)
            nm['segmentation'] = new_seg
            nm['area']         = new_area
            nm['bbox']         = _recompute_bbox(new_seg)
            trimmed.append(nm)
    masks = trimmed

    # Step 3: largest-first
    masks = sorted(masks, key=lambda m: m['area'], reverse=True)

    result = []

    for mask in masks:

        seg  = mask['segmentation'].copy()
        area = int(seg.sum())
        skip = False

        for kept in result:

            kept_seg  = kept['segmentation']
            kept_area = kept['area']

            intersection = int(np.logical_and(seg, kept_seg).sum())
            union        = int(np.logical_or(seg, kept_seg).sum())

            iou          = float(intersection) / union      if union      > 0 else 0.0
            contain_curr = float(intersection) / area       if area       > 0 else 0.0
            contain_kept = float(intersection) / kept_area  if kept_area  > 0 else 0.0

            if (iou > overlap_iou_threshold
                    or max(contain_curr, contain_kept) > containment_threshold):
                skip = True
                break

            elif iou > complementary_iou_low:
                complement = seg & ~kept_seg
                comp_area  = int(complement.sum())
                if comp_area >= small_area_threshold:
                    seg  = complement
                    area = comp_area
                else:
                    skip = True
                    break

        if skip:
            continue

        final_area = int(seg.sum())
        if final_area < small_area_threshold:
            continue

        new_mask                 = dict(mask)
        new_mask['segmentation'] = seg
        new_mask['area']         = final_area
        new_mask['bbox']         = _recompute_bbox(seg)
        result.append(new_mask)

    return result
