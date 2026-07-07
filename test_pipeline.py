"""
Comprehensive Dataset Test & Pipeline Validation
================================================

Validates BOTH datasets (data/realsense and data/train_1) and runs
a lightweight end-to-end test on the best recommended scenes without
needing the full VLM stack.

Tests performed:
  1. Dataset integrity   — frame counts, image dimensions, depth validity
  2. Annotation parsing  — all XML files readable, object IDs resolved
  3. Camera calibration  — camK.npy loadable and valid
  4. SAM segmentation    — smoke-test on one recommended frame
  5. Scene ranking       — list best frames for full pipeline runs
  6. Labels summary      — existing outputs/labels/labels.json coverage

Usage:
    cd /home/arc02/Grasp_intent/grasp_pipeline
    python test_pipeline.py [--sam] [--verbose]

    --sam     Run a real SAM segmentation smoke-test (needs checkpoint)
    --verbose Print per-frame details
"""

import os
import sys
import glob
import json
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from collections import Counter

# ─── import project utilities ─────────────────────────────────────────────────
from utils.label_normalize import (
    GT_OBJ_ID_TO_NAME,
    CANONICAL_GRASP_MAP,
    parse_annotation_xml,
    normalize_object_label,
)

# ─── paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RS_DIR      = os.path.join(BASE_DIR, "data", "realsense")
TRAIN1_DIR  = os.path.join(BASE_DIR, "data", "train_1")
LABEL_PATH  = os.path.join(BASE_DIR, "outputs", "labels", "labels.json")
SAM_CKPT    = os.path.join(BASE_DIR, "sam_vit_l_0b3195.pth")

# ─── expected image dimensions ────────────────────────────────────────────────
EXPECTED_H, EXPECTED_W = 720, 1280

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m~\033[0m"


# ══════════════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════════════

def _sep(title=""):
    w = 62
    if title:
        pad = (w - len(title) - 2) // 2
        print("\n" + "─" * pad + f" {title} " + "─" * (w - pad - len(title) - 2))
    else:
        print("\n" + "─" * w)


def _ok(msg):  print(f"  {PASS} {msg}")
def _fail(msg):print(f"  {FAIL} {msg}")
def _warn(msg):print(f"  {WARN} {msg}")


def _count(directory, ext):
    return len(glob.glob(os.path.join(directory, f"*{ext}")))


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1 — data/realsense/ integrity
# ══════════════════════════════════════════════════════════════════════════════

def test_realsense_integrity(verbose=False):
    _sep("TEST 1 · data/realsense/ integrity")
    errors = 0

    subdirs = {
        "rgb":         ".png",
        "depth":       ".png",
        "annotations": ".xml",
        "label":       ".png",
        "meta":        ".mat",
    }

    counts = {}
    for sd, ext in subdirs.items():
        path = os.path.join(RS_DIR, sd)
        n = _count(path, ext) if os.path.isdir(path) else 0
        counts[sd] = n
        if n == 256:
            _ok(f"{sd:>12}/ → {n} files")
        else:
            _fail(f"{sd:>12}/ → {n} files  (expected 256)")
            errors += 1

    # All modalities must have the same count
    unique_counts = set(counts.values())
    if len(unique_counts) == 1:
        _ok("All modalities balanced")
    else:
        _fail(f"Modality counts differ: {counts}")
        errors += 1

    # Camera calibration files
    for cf in ("camK.npy", "cam0_wrt_table.npy", "camera_poses.npy"):
        if os.path.isfile(os.path.join(RS_DIR, cf)):
            _ok(f"{cf}")
        else:
            _fail(f"{cf} MISSING")
            errors += 1

    # Spot-check image shapes on frames 0, 64, 128, 192, 255
    shape_errors = 0
    for idx in (0, 64, 128, 192, 255):
        rgb_path = os.path.join(RS_DIR, "rgb", f"{idx:04d}.png")
        dep_path = os.path.join(RS_DIR, "depth", f"{idx:04d}.png")
        img = cv2.imread(rgb_path)
        dep = cv2.imread(dep_path, cv2.IMREAD_UNCHANGED)
        if img is None or img.shape[:2] != (EXPECTED_H, EXPECTED_W):
            shape_errors += 1
        if dep is None or dep.shape[:2] != (EXPECTED_H, EXPECTED_W):
            shape_errors += 1
        elif verbose:
            _ok(f"  frame {idx:04d}  rgb{img.shape}  depth{dep.shape}  "
                f"d_max={dep.max()}")

    if shape_errors == 0:
        _ok(f"Spot-checked 5 frames — all {EXPECTED_H}×{EXPECTED_W}")
    else:
        _fail(f"{shape_errors} shape mismatches in spot-check")
        errors += shape_errors

    # Depth validity — non-zero ratio
    dep0 = cv2.imread(os.path.join(RS_DIR, "depth", "0000.png"), cv2.IMREAD_UNCHANGED)
    if dep0 is not None:
        valid_ratio = np.count_nonzero(dep0) / dep0.size
        if valid_ratio > 0.70:
            _ok(f"Depth non-zero ratio: {valid_ratio:.1%}")
        else:
            _warn(f"Depth non-zero ratio LOW: {valid_ratio:.1%}")

    return errors


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2 — data/train_1/ integrity
# ══════════════════════════════════════════════════════════════════════════════

def test_train1_integrity(verbose=False):
    _sep("TEST 2 · data/train_1/ integrity")
    errors = 0

    if not os.path.isdir(TRAIN1_DIR):
        _fail("data/train_1/ directory not found!")
        return 1

    scenes = sorted([d for d in os.listdir(TRAIN1_DIR) if d.startswith("scene_")])
    _ok(f"Found {len(scenes)} scenes")

    scene_errors = []
    known_count = unknown_count = 0
    obj_id_freq = Counter()

    for scene in scenes:
        sp   = os.path.join(TRAIN1_DIR, scene)
        rs   = os.path.join(sp, "realsense")
        rgb_n  = _count(os.path.join(rs, "rgb"),         ".png")
        dep_n  = _count(os.path.join(rs, "depth"),       ".png")
        ann_n  = _count(os.path.join(rs, "annotations"), ".xml")
        obj_f  = os.path.join(sp, "object_id_list.txt")
        obj_ids = []
        if os.path.isfile(obj_f):
            obj_ids = [int(x) for x in open(obj_f).read().split() if x.strip()]

        if rgb_n != 256 or dep_n != 256 or ann_n != 256:
            scene_errors.append(f"{scene}: rgb={rgb_n} dep={dep_n} ann={ann_n}")
            errors += 1
        elif verbose:
            _ok(f"{scene}: {rgb_n} frames, {len(obj_ids)} objects")

        for oid in obj_ids:
            obj_id_freq[oid] += 1
            if oid in GT_OBJ_ID_TO_NAME:
                known_count += 1
            else:
                unknown_count += 1

    if scene_errors:
        for e in scene_errors:
            _fail(e)
    else:
        _ok("All 30 scenes balanced (256 rgb = 256 depth = 256 annotations)")

    _ok(f"Object IDs — known: {known_count}, unknown: {unknown_count}")
    if unknown_count == 0:
        _ok("GT_OBJ_ID_TO_NAME covers 100% of train_1 object IDs")
    else:
        _warn(f"{unknown_count} object appearances still unmapped")

    # Top objects by scene frequency
    print("\n  Most common objects in train_1:")
    for oid, cnt in obj_id_freq.most_common(10):
        name = GT_OBJ_ID_TO_NAME.get(oid, f"unknown_{oid}")
        print(f"    id={oid:>3}  {name:<30}  scenes={cnt}")

    return errors


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3 — annotation parsing
# ══════════════════════════════════════════════════════════════════════════════

def test_annotation_parsing():
    _sep("TEST 3 · Annotation XML parsing")
    errors = 0

    ann_dir = os.path.join(RS_DIR, "annotations")
    xml_files = sorted(glob.glob(os.path.join(ann_dir, "*.xml")))

    parse_errors = 0
    obj_counts = []
    for xml_path in xml_files:
        try:
            objs = parse_annotation_xml(xml_path)
            obj_counts.append(len(objs))
        except Exception as e:
            parse_errors += 1
            if parse_errors <= 3:
                _fail(f"{os.path.basename(xml_path)}: {e}")

    if parse_errors == 0:
        _ok(f"All {len(xml_files)} annotation XMLs parsed without error")
    else:
        _fail(f"{parse_errors} parse errors")
        errors += parse_errors

    avg_obj = sum(obj_counts) / max(len(obj_counts), 1)
    _ok(f"Avg objects per view: {avg_obj:.1f}  "
        f"(min={min(obj_counts)}  max={max(obj_counts)})")

    # Unique object IDs across all views
    all_ids = set()
    for xml_path in xml_files[:1]:          # just check one for speed
        for o in parse_annotation_xml(xml_path):
            all_ids.add(o["obj_id"])
    _ok(f"Unique GT object IDs in realsense: "
        f"{sorted(all_ids)} → {[GT_OBJ_ID_TO_NAME.get(i,'?') for i in sorted(all_ids)]}")

    return errors


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4 — camera calibration
# ══════════════════════════════════════════════════════════════════════════════

def test_camera_calibration():
    _sep("TEST 4 · Camera calibration")
    errors = 0

    camK_path = os.path.join(RS_DIR, "camK.npy")
    try:
        camK = np.load(camK_path)
        assert camK.shape == (3, 3), f"Expected (3,3), got {camK.shape}"
        assert camK[0, 0] > 0 and camK[1, 1] > 0, "focal lengths must be positive"
        _ok(f"camK loaded  fx={camK[0,0]:.1f}  fy={camK[1,1]:.1f}  "
            f"cx={camK[0,2]:.1f}  cy={camK[1,2]:.1f}")
    except Exception as e:
        _fail(f"camK.npy: {e}")
        errors += 1

    for fname in ("cam0_wrt_table.npy", "camera_poses.npy"):
        fpath = os.path.join(RS_DIR, fname)
        try:
            arr = np.load(fpath)
            _ok(f"{fname}  shape={arr.shape}  dtype={arr.dtype}")
        except Exception as e:
            _fail(f"{fname}: {e}")
            errors += 1

    return errors


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5 — scene ranking (best frames for pipeline run)
# ══════════════════════════════════════════════════════════════════════════════

def test_scene_ranking():
    _sep("TEST 5 · Scene ranking (best frames for pipeline)")
    errors = 0

    ann_dir = os.path.join(RS_DIR, "annotations")
    dep_dir = os.path.join(RS_DIR, "depth")
    xml_files = sorted(glob.glob(os.path.join(ann_dir, "*.xml")))

    scored = []
    for xml_path in xml_files:
        view_id = os.path.basename(xml_path).replace(".xml", "")
        objs = parse_annotation_xml(xml_path)

        dep_path = os.path.join(dep_dir, f"{view_id}.png")
        dep = cv2.imread(dep_path, cv2.IMREAD_UNCHANGED)
        depth_ratio = np.count_nonzero(dep) / dep.size if dep is not None else 0

        scored.append({
            "id":          view_id,
            "obj_count":   len(objs),
            "depth_ratio": depth_ratio,
            "objects":     [o["canonical_name"] for o in objs],
        })

    # Sort: max objects → best depth
    scored.sort(key=lambda x: (x["obj_count"], x["depth_ratio"]), reverse=True)

    # Bucket-diverse top-5
    selected = []
    buckets_used = set()
    n_buckets = 5
    bsize = len(scored) // n_buckets

    for s in scored:
        bucket = int(s["id"]) // max(bsize, 1)
        if bucket not in buckets_used and s["obj_count"] >= 9:
            selected.append(s)
            buckets_used.add(bucket)
            if len(selected) >= n_buckets:
                break

    # Fallback
    for s in scored:
        if s not in selected and len(selected) < n_buckets:
            selected.append(s)

    print(f"\n  Top {n_buckets} recommended frames for pipeline run:")
    for i, s in enumerate(selected):
        obj_str = ", ".join(s["objects"][:5])
        print(f"    {i+1}. frame {s['id']}  "
              f"({s['obj_count']} objects, depth={s['depth_ratio']:.1%})  "
              f"[{obj_str}]")

    print(f"\n  → To run pipeline on these frames, set in generate_labels.py:")
    ids = [int(s["id"]) for s in selected]
    print(f"      SCENE_START = {min(ids)}  # or use individual frame IDs")

    return errors


# ══════════════════════════════════════════════════════════════════════════════
# TEST 6 — existing labels.json summary
# ══════════════════════════════════════════════════════════════════════════════

def test_labels_summary():
    _sep("TEST 6 · Existing labels.json summary")
    errors = 0

    if not os.path.isfile(LABEL_PATH):
        _warn("No labels.json found — pipeline hasn't run yet")
        return 0

    with open(LABEL_PATH) as f:
        labels = json.load(f)

    n_scenes = len(labels)
    n_objects = sum(len(v) for v in labels.values())
    _ok(f"Labelled scenes: {n_scenes} / 256")
    _ok(f"Total labelled objects: {n_objects}")

    # Grasp distribution
    grasp_dist = Counter()
    obj_dist = Counter()
    for scene_labels in labels.values():
        for entry in scene_labels:
            grasp_dist[entry.get("grasp", "unknown")] += 1
            obj_dist[normalize_object_label(entry.get("object", "unknown"))] += 1

    print("\n  Grasp type distribution:")
    max_grasp = max(grasp_dist.values()) if grasp_dist else 1
    for grasp, cnt in grasp_dist.most_common():
        bar = "█" * (cnt * 30 // max(max_grasp, 1))
        print(f"    {grasp:<28} {cnt:>4}  {bar}")

    print("\n  Top-10 detected objects (normalized):")
    for obj, cnt in obj_dist.most_common(10):
        print(f"    {obj:<30} {cnt:>4}")

    # Coverage: how many GT objects appear at all
    gt_names = set(GT_OBJ_ID_TO_NAME.values())
    detected_gt = gt_names & set(obj_dist.keys())
    _ok(f"GT object coverage: {len(detected_gt)}/{len(gt_names)} "
        f"= {len(detected_gt)/max(len(gt_names),1):.0%}")

    return errors


# ══════════════════════════════════════════════════════════════════════════════
# TEST 7 — SAM smoke-test (optional, skipped by default)
# ══════════════════════════════════════════════════════════════════════════════

def test_sam_smoke(frame_id="0055"):
    _sep("TEST 7 · SAM segmentation smoke-test")

    if not os.path.isfile(SAM_CKPT):
        _warn(f"SAM checkpoint not found at {SAM_CKPT} — skipping")
        return 0

    try:
        from sam_module.sam_segment import SAMSegmenter
        from utils.mask_filter import filter_small_masks
        from utils.depth_filter import filter_masks_by_depth

        _ok("SAM module imported successfully")

        rgb_path   = os.path.join(RS_DIR, "rgb",   f"{frame_id}.png")
        depth_path = os.path.join(RS_DIR, "depth", f"{frame_id}.png")

        image = cv2.imread(rgb_path)
        if image is None:
            _fail(f"Could not load {rgb_path}")
            return 1
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        print(f"  Loading SAM model (this may take ~30 s the first time)…")
        segmenter = SAMSegmenter(checkpoint_path=SAM_CKPT)
        masks = segmenter.generate_masks(image)
        _ok(f"Raw masks: {len(masks)}")

        masks = filter_small_masks(masks, min_area=5000)
        _ok(f"After small-area filter: {len(masks)}")

        masks = filter_masks_by_depth(masks, depth)
        _ok(f"After depth filter: {len(masks)}")

        if len(masks) >= 3:
            _ok(f"SAM smoke-test PASSED on frame {frame_id}")
        else:
            print(f"  ~ Only {len(masks)} masks remained (might be fine for "
                  f"sparse frames)")

    except Exception as e:
        _fail(f"SAM smoke-test error: {e}")
        return 1

    return 0


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Grasp pipeline dataset tester")
    parser.add_argument("--sam",     action="store_true",
                        help="Run SAM smoke-test (slow, needs GPU/checkpoint)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-frame / per-scene details")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║      GRASP PIPELINE — COMPREHENSIVE DATASET TEST SUITE      ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    total_errors = 0
    total_errors += test_realsense_integrity(verbose=args.verbose)
    total_errors += test_train1_integrity(verbose=args.verbose)
    total_errors += test_annotation_parsing()
    total_errors += test_camera_calibration()
    total_errors += test_scene_ranking()
    total_errors += test_labels_summary()

    if args.sam:
        total_errors += test_sam_smoke()

    _sep()
    if total_errors == 0:
        print(f"\n  {PASS}  ALL TESTS PASSED  —  dataset is ready for "
              f"generate_labels.py\n")
    else:
        print(f"\n  {FAIL}  {total_errors} issue(s) found  — review output above\n")

    sys.exit(0 if total_errors == 0 else 1)


if __name__ == "__main__":
    main()
