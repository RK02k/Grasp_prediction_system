"""
Full SAM + VLM pipeline runner for data/train_1/
=================================================

Processes every scene in data/train_1/ (scene_0000 … scene_0029).

For each scene:
  1. Selects the best representative frame (highest depth coverage).
  2. Runs SAM automatic segmentation.
  3. Post-processes masks (area filter → depth filter → IoU dedup).
  4. Saves RGB crops + scene overlay to outputs/train1/crops/<scene>/
  5. Runs the 4-submodule VLM pipeline on every crop.
  6. Compares VLM results against ground-truth object_id_list.txt.
  7. Writes results to outputs/train1/labels/train1_labels.json
     (crash-safe: already-done scenes are skipped on restart).

Usage:
    cd /home/arc02/Grasp_intent/grasp_pipeline
    python run_train1_pipeline.py

    # To process only a subset:
    python run_train1_pipeline.py --start 0 --end 9

    # Dry-run (SAM + crops only, skip VLM):
    python run_train1_pipeline.py --no-vlm
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import cv2

from sam_module.sam_segment import SAMSegmenter

from utils.mask_postprocess import postprocess_masks
from utils.depth_filter import filter_masks_by_depth
from utils.object_crop import save_object_crops
from utils.visualize import generate_scene_overlay
from utils.crop_splitter import split_crop_with_sam
from utils.label_normalize import (
    GT_OBJ_ID_TO_NAME,
    CANONICAL_GRASP_MAP,
    SEMANTIC_GRASP_COMPAT_MAP,
    normalize_object_label,
    semantic_object_match,
)

from vlm_module.qwen_grasp import QwenGraspModel
from vlm_module.pipeline import (
    run_submodule,
    SM1_MAIN, SM1_JUDGE,
    SM2_MAIN, SM2_JUDGE,
    SM3_MAIN, SM3_JUDGE,
    SM4_MAIN, SM4_JUDGE,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

TRAIN1_DIR   = os.path.join("data", "train_1")
SAM_CKPT     = "sam_vit_l_0b3195.pth"

# VLM inference settings  (paper Table 5 best configuration)
N_DECISIONS  = 11    # majority votes per submodule
N_RETRIES    = 4     # retry cycles on bad format / judgment rejection

# Mask thresholds
SMALL_AREA_THRESHOLD  = 5000
MAX_AREA_THRESHOLD    = 245760
OVERLAP_IOU_THRESHOLD = 0.4
COMPLEMENTARY_IOU_LOW = 0.1
CONTAINMENT_THRESHOLD = 0.8

# Output dirs
OUTPUT_ROOT   = os.path.join("outputs", "train1")
CROPS_ROOT    = os.path.join(OUTPUT_ROOT, "crops")
OVERLAYS_DIR  = os.path.join(OUTPUT_ROOT, "scene_overlays")
LABELS_DIR    = os.path.join(OUTPUT_ROOT, "labels")
LABEL_PATH    = os.path.join(LABELS_DIR, "train1_labels.json")


# =============================================================================
# HELPERS
# =============================================================================

def pick_best_frame(scene_path, n_candidates=10):
    """
    Return the frame index with the highest depth non-zero ratio
    from the first n_candidates frames in a scene's realsense/depth/ folder.
    """
    depth_dir = os.path.join(scene_path, "realsense", "depth")
    frames = sorted(glob.glob(os.path.join(depth_dir, "*.png")))[:n_candidates]
    best_idx, best_ratio = 0, 0.0
    for fpath in frames:
        dep = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        if dep is None:
            continue
        ratio = np.count_nonzero(dep) / dep.size
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = int(os.path.basename(fpath).replace(".png", ""))
    return best_idx, best_ratio


def load_gt_objects(scene_path):
    """Return list of (obj_id, canonical_name) for the scene."""
    obj_list_path = os.path.join(scene_path, "object_id_list.txt")
    if not os.path.isfile(obj_list_path):
        return []
    ids = [int(x) for x in open(obj_list_path).read().split() if x.strip()]
    return [(oid, GT_OBJ_ID_TO_NAME.get(oid, f"unknown_{oid}")) for oid in ids]


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",   type=int, default=0,
                        help="First scene index to process (default: 0)")
    parser.add_argument("--end",     type=int, default=29,
                        help="Last scene index to process (default: 29)")
    parser.add_argument("--no-vlm",  action="store_true",
                        help="Skip VLM — only SAM segmentation and crops")
    args = parser.parse_args()

    # ─── collect scenes ────────────────────────────────────────────────────────
    all_scene_dirs = sorted([
        d for d in os.listdir(TRAIN1_DIR)
        if d.startswith("scene_") and os.path.isdir(os.path.join(TRAIN1_DIR, d))
    ])
    scenes_to_run = [
        d for d in all_scene_dirs
        if args.start <= int(d.split("_")[1]) <= args.end
    ]

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║        TRAIN-1 PIPELINE  —  SAM + VLM Label Runner          ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  Scenes to process : {len(scenes_to_run)}  "
          f"(scene_{args.start:04d} → scene_{args.end:04d})")
    print(f"  VLM enabled       : {not args.no_vlm}")
    print(f"  Output labels     : {LABEL_PATH}\n")

    # ─── create output dirs ────────────────────────────────────────────────────
    for d in (CROPS_ROOT, OVERLAYS_DIR, LABELS_DIR):
        os.makedirs(d, exist_ok=True)

    # ─── resume support ────────────────────────────────────────────────────────
    if os.path.isfile(LABEL_PATH):
        with open(LABEL_PATH) as f:
            all_results = json.load(f)
        print(f"Resuming: {len(all_results)} scene(s) already done.\n")
    else:
        all_results = {}

    # ─── load models ──────────────────────────────────────────────────────────
    print("Loading SAM …")
    segmenter = SAMSegmenter(checkpoint_path=SAM_CKPT)

    model = None
    if not args.no_vlm:
        print("Loading VLM …")
        model = QwenGraspModel()

    # ==========================================================================
    # SCENE LOOP
    # ==========================================================================

    for scene_name in scenes_to_run:

        # ── skip if already done ──────────────────────────────────────────────
        if scene_name in all_results:
            print(f"[{scene_name}] Already done — skipping.")
            continue

        scene_path = os.path.join(TRAIN1_DIR, scene_name)
        print(f"\n{'#'*64}")
        print(f"#  {scene_name}  ({scenes_to_run.index(scene_name)+1}/{len(scenes_to_run)})")
        print(f"{'#'*64}")

        # ── ground-truth objects ──────────────────────────────────────────────
        gt_objects = load_gt_objects(scene_path)
        gt_names   = [name for _, name in gt_objects]
        print(f"  GT objects ({len(gt_objects)}): {', '.join(gt_names)}")

        # ── pick best representative frame ────────────────────────────────────
        best_frame_idx, depth_ratio = pick_best_frame(scene_path)
        frame_id   = f"{best_frame_idx:04d}"
        rgb_path   = os.path.join(scene_path, "realsense", "rgb",   f"{frame_id}.png")
        depth_path = os.path.join(scene_path, "realsense", "depth", f"{frame_id}.png")
        print(f"  Best frame: {frame_id}  (depth coverage {depth_ratio:.1%})")

        # ── load images ───────────────────────────────────────────────────────
        image = cv2.imread(rgb_path)
        if image is None:
            print(f"  [WARN] RGB not found: {rgb_path} — skipping scene.")
            all_results[scene_name] = {"gt_objects": gt_names, "vlm_objects": []}
            _save(all_results)
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (1280, 720))

        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            print(f"  [WARN] Depth not found: {depth_path} — skipping scene.")
            all_results[scene_name] = {"gt_objects": gt_names, "vlm_objects": []}
            _save(all_results)
            continue
        depth = cv2.resize(depth, (1280, 720), interpolation=cv2.INTER_NEAREST)

        # ── SAM segmentation ──────────────────────────────────────────────────
        raw_masks = segmenter.generate_masks(image)
        print(f"  Raw SAM masks      : {len(raw_masks)}")

        masks = [m for m in raw_masks if m["area"] >= SMALL_AREA_THRESHOLD]
        print(f"  After area filter  : {len(masks)}")

        masks = filter_masks_by_depth(masks, depth)
        print(f"  After depth filter : {len(masks)}")

        masks = postprocess_masks(
            masks,
            small_area_threshold=SMALL_AREA_THRESHOLD,
            max_area_threshold=MAX_AREA_THRESHOLD,
            overlap_iou_threshold=OVERLAP_IOU_THRESHOLD,
            complementary_iou_low=COMPLEMENTARY_IOU_LOW,
            containment_threshold=CONTAINMENT_THRESHOLD,
        )
        print(f"  After IoU postproc : {len(masks)}")

        if len(masks) == 0:
            print("  No objects found after filtering — skipping VLM.")
            all_results[scene_name] = {"gt_objects": gt_names, "vlm_objects": []}
            _save(all_results)
            continue

        # ── save crops & overlay ──────────────────────────────────────────────
        crops_dir       = os.path.join(CROPS_ROOT, scene_name)
        depth_crops_dir = os.path.join(crops_dir, "depth")
        scene_overlay   = os.path.join(OVERLAYS_DIR, f"{scene_name}.png")

        # ── purge any sub-crops left by a previous interrupted run ────────────
        for _sub_dir in (crops_dir, os.path.join(crops_dir, "depth")):
            if os.path.isdir(_sub_dir):
                for stale in os.listdir(_sub_dir):
                    if "_sub" in stale and stale.endswith(".png"):
                        os.remove(os.path.join(_sub_dir, stale))

        save_object_crops(image, masks, output_dir=crops_dir, depth_image=depth)

        overlay_img = generate_scene_overlay(image, masks)
        cv2.imwrite(scene_overlay, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
        print(f"  Crops saved → {crops_dir}/")
        print(f"  Overlay saved → {scene_overlay}")

        # ── VLM labelling ─────────────────────────────────────────────────────
        if args.no_vlm:
            all_results[scene_name] = {
                "gt_objects":    gt_names,
                "frame_used":    frame_id,
                "masks_found":   len(masks),
                "vlm_objects":   [],
            }
            _save(all_results)
            continue

        # Exclude sub-crops (_sub*.png) — those are created inline during the
        # multi-object split and processed immediately; including them here
        # would process them a second time.
        crop_files = sorted([
            f for f in os.listdir(crops_dir)
            if f.endswith(".png") and "_sub" not in f
        ], key=lambda f: int(os.path.splitext(f)[0].replace("object_", "")))
        scene_results = []

        for crop_file in crop_files:
            crop_path       = os.path.join(crops_dir,       crop_file)
            depth_crop_path = os.path.join(depth_crops_dir, crop_file)

            print(f"\n  ── {crop_file} ──")

            # SM1: is this an object?
            is_obj = run_submodule(
                model, scene_overlay, crop_path,
                SM1_MAIN, SM1_JUDGE,
                n_decisions=N_DECISIONS, n_retries=N_RETRIES,
            )
            print(f"    SM1 (object?): [{is_obj}]")
            if is_obj is None or is_obj.lower() == "false":
                print("    → Skipped (background)")
                continue

            # SM2: object count
            obj_count = run_submodule(
                model, scene_overlay, crop_path,
                SM2_MAIN, SM2_JUDGE,
                n_decisions=N_DECISIONS, n_retries=N_RETRIES,
            )
            print(f"    SM2 (count):   [{obj_count}]")
            try:
                count_val = int(obj_count) if obj_count else 0
            except (ValueError, TypeError):
                count_val = 0

            if count_val > 1:
                # ── Phase 1 fix: hierarchical re-segmentation ─────────────
                # Re-run SAM inside this crop to obtain individual sub-masks,
                # then process each sub-crop through SM3/SM4 independently.
                print(f"    → Multi-object crop (SM2={obj_count}): "
                      f"attempting hierarchical split …")

                # Recover the original mask dict from the ordered mask list
                crop_idx = int(
                    os.path.splitext(crop_file)[0].replace("object_", "")
                )
                parent_mask = masks[crop_idx] if crop_idx < len(masks) else None

                sub_crops: list[tuple[str, dict]] = []
                if parent_mask is not None:
                    sub_crops = split_crop_with_sam(
                        segmenter, image, parent_mask, crop_path,
                        depth_image=depth,
                    )

                if sub_crops:
                    print(f"    → Split into {len(sub_crops)} sub-crop(s)")
                    for sub_path, _sub_mask in sub_crops:
                        sub_file = os.path.basename(sub_path)
                        print(f"\n  ── {sub_file} (sub-crop) ──")

                        # SM3 on sub-crop
                        sub_label_raw = run_submodule(
                            model, scene_overlay, sub_path,
                            SM3_MAIN, SM3_JUDGE,
                            n_decisions=N_DECISIONS, n_retries=N_RETRIES,
                        ) or "unknown"
                        sub_label = normalize_object_label(sub_label_raw)
                        print(f"    SM3 (object):  [{sub_label_raw}] → [{sub_label}]")

                        # SM4 on sub-crop
                        sub_grasp = run_submodule(
                            model, scene_overlay, sub_path,
                            SM4_MAIN.format(object_label=sub_label),
                            SM4_JUDGE.format(object_label=sub_label),
                            n_decisions=N_DECISIONS, n_retries=N_RETRIES,
                        )
                        exp_grasp = CANONICAL_GRASP_MAP.get(sub_label, "—")
                        compat    = SEMANTIC_GRASP_COMPAT_MAP.get(
                            sub_label, {exp_grasp}
                        )
                        exact_m  = (sub_grasp == exp_grasp)
                        sem_m    = bool(sub_grasp and sub_grasp in compat)
                        sym      = "✓" if exact_m else ("~" if sem_m else "✗")
                        print(f"    SM4 (grasp):   [{sub_grasp}]  "
                              f"expected: [{exp_grasp}] {sym}")

                        scene_results.append({
                            "scene":          scene_name,
                            "frame":          frame_id,
                            "crop":           sub_file,
                            "object_count":   "1",
                            "object_raw":     sub_label_raw,
                            "object":         sub_label,
                            "grasp":          sub_grasp,
                            "grasp_correct":  exact_m,
                            "grasp_semantic": sem_m,
                            "depth_crop":     None,
                            "from_split":     True,
                        })
                else:
                    print(f"    → Split failed — skipping crop")
                continue  # done with this parent crop (either split or skipped)

            # SM3: object category
            obj_label_raw = run_submodule(
                model, scene_overlay, crop_path,
                SM3_MAIN, SM3_JUDGE,
                n_decisions=N_DECISIONS, n_retries=N_RETRIES,
            ) or "unknown"
            obj_label = normalize_object_label(obj_label_raw)
            print(f"    SM3 (object):  [{obj_label_raw}] → [{obj_label}]")

            # SM4: grasp type
            grasp = run_submodule(
                model, scene_overlay, crop_path,
                SM4_MAIN.format(object_label=obj_label),
                SM4_JUDGE.format(object_label=obj_label),
                n_decisions=N_DECISIONS, n_retries=N_RETRIES,
            )
            expected_grasp = CANONICAL_GRASP_MAP.get(obj_label, "—")
            compat_set     = SEMANTIC_GRASP_COMPAT_MAP.get(
                obj_label, {expected_grasp}
            )
            exact_match    = (grasp == expected_grasp)
            semantic_match = bool(grasp and grasp in compat_set)
            match_sym      = "✓" if exact_match else ("~" if semantic_match else "✗")
            print(f"    SM4 (grasp):   [{grasp}]  expected: [{expected_grasp}] {match_sym}"
                  f"  compat: {sorted(compat_set)}")

            scene_results.append({
                "scene":            scene_name,
                "frame":            frame_id,
                "crop":             crop_file,
                "object_count":     obj_count,
                "object_raw":       obj_label_raw,
                "object":           obj_label,
                "grasp":            grasp,
                "grasp_correct":    exact_match,
                "grasp_semantic":   semantic_match,
                "depth_crop":       depth_crop_path if os.path.isfile(depth_crop_path) else None,
            })

        # ── per-scene object NMS: deduplicate repeated names ─────────────────
        # If the same canonical name appears more than once, keep only the first
        # occurrence that has a correct grasp, or just the first occurrence.
        # This suppresses "banana×4" style hallucination.
        seen_names: dict[str, int] = {}   # name → count seen so far
        deduped: list[dict] = []
        for r in scene_results:
            name = r["object"]
            seen_names[name] = seen_names.get(name, 0) + 1
            # Allow up to 2 of any name (some scenes genuinely have 2 of the
            # same object); beyond that, discard.
            if seen_names[name] <= 2:
                deduped.append(r)
            else:
                print(f"    [NMS] Dropped duplicate '{name}' "
                      f"(occurrence #{seen_names[name]})")
        scene_results = deduped

        # ── per-scene GT recall ───────────────────────────────────────────────
        vlm_names  = set(r["object"] for r in scene_results)
        gt_set     = set(gt_names)

        # Exact recall: canonical name must match after normalization
        detected_exact = gt_set & vlm_names
        # Semantic recall: any raw VLM output (object_raw) semantically matches GT
        detected_semantic: set[str] = set()
        for gt_obj in gt_set:
            for r in scene_results:
                raw_out = r.get("object_raw", r["object"])
                if semantic_object_match(raw_out, gt_obj):
                    detected_semantic.add(gt_obj)
                    break

        missed     = gt_set - detected_semantic
        fp         = vlm_names - gt_set
        recall_exact    = len(detected_exact)    / max(len(gt_set), 1)
        recall_semantic = len(detected_semantic) / max(len(gt_set), 1)

        grasp_exact    = sum(1 for r in scene_results if r["grasp_correct"])
        grasp_semantic = sum(1 for r in scene_results if r["grasp_semantic"])
        grasp_total    = len(scene_results)

        print(f"\n  ── Scene {scene_name} summary ──")
        print(f"  GT objects        : {sorted(gt_set)}")
        print(f"  VLM detected      : {sorted(vlm_names)}")
        print(f"  Recall (exact)    : {len(detected_exact)}/{len(gt_set)} = {recall_exact:.0%}")
        print(f"  Recall (semantic) : {len(detected_semantic)}/{len(gt_set)} = {recall_semantic:.0%}")
        print(f"  Missed            : {sorted(missed)}")
        print(f"  False positives   : {sorted(fp)}")
        print(f"  Grasp exact acc   : {grasp_exact}/{grasp_total}")
        print(f"  Grasp semantic    : {grasp_semantic}/{grasp_total}")

        all_results[scene_name] = {
            "gt_objects":                gt_names,
            "frame_used":                frame_id,
            "depth_ratio":               round(depth_ratio, 3),
            "masks_found":               len(masks),
            "vlm_objects":               scene_results,
            "recall":                    round(recall_exact,    3),
            "recall_semantic":           round(recall_semantic, 3),
            "grasp_accuracy":            round(grasp_exact    / max(grasp_total, 1), 3),
            "grasp_accuracy_semantic":   round(grasp_semantic / max(grasp_total, 1), 3),
            "detected":                  list(detected_exact),
            "detected_semantic":         list(detected_semantic),
            "missed":                    list(missed),
            "false_positives":           list(fp),
        }
        _save(all_results)
        print(f"\n  [{scene_name}] Done — {len(scene_results)} object(s) labeled. "
              f"Saved → {LABEL_PATH}")

    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================

    _print_summary(all_results)


def _save(results):
    with open(LABEL_PATH, "w") as f:
        json.dump(results, f, indent=4)


def _print_summary(all_results):
    print(f"\n{'='*64}")
    print("TRAIN-1 PIPELINE COMPLETE")
    print(f"{'='*64}")

    total_scenes  = len(all_results)
    total_objects = sum(
        len(v.get("vlm_objects", [])) for v in all_results.values()
    )
    recalls      = [v["recall"]                  for v in all_results.values() if "recall"                  in v]
    recall_sems  = [v.get("recall_semantic", v.get("recall", 0))
                    for v in all_results.values() if "recall" in v]
    grasp_accs   = [v["grasp_accuracy"]           for v in all_results.values() if "grasp_accuracy"           in v]
    grasp_sem    = [v["grasp_accuracy_semantic"]   for v in all_results.values() if "grasp_accuracy_semantic"   in v]

    print(f"  Scenes processed  : {total_scenes}")
    print(f"  Total VLM objects : {total_objects}")

    if recalls:
        print(f"  Avg recall (exact)   : {sum(recalls)/len(recalls):.1%}")
    if recall_sems:
        print(f"  Avg recall (semantic): {sum(recall_sems)/len(recall_sems):.1%}")
    if grasp_accs:
        print(f"  Avg grasp exact   : {sum(grasp_accs)/len(grasp_accs):.1%}")
    if grasp_sem:
        print(f"  Avg grasp semantic: {sum(grasp_sem)/len(grasp_sem):.1%}")

    # Per-scene table
    print(f"\n  {'Scene':<15} {'GT':>4} {'Det':>4} {'SemDet':>7} "
          f"{'Recall':>7} {'SemRec':>7} {'ExactAcc':>9} {'SemAcc':>7}")
    print(f"  {'-'*15} {'-'*4} {'-'*4} {'-'*7} "
          f"{'-'*7} {'-'*7} {'-'*9} {'-'*7}")
    for scene_name in sorted(all_results.keys()):
        v = all_results[scene_name]
        gt_n   = len(v.get("gt_objects",        []))
        det_n  = len(v.get("detected",          []))
        sdet_n = len(v.get("detected_semantic", v.get("detected", [])))
        rec    = v.get("recall",                   0.0)
        rec_s  = v.get("recall_semantic",          rec)
        ga     = v.get("grasp_accuracy",            0.0)
        gs     = v.get("grasp_accuracy_semantic",   0.0)
        print(f"  {scene_name:<15} {gt_n:>4} {det_n:>4} {sdet_n:>7} "
              f"{rec:>7.0%} {rec_s:>7.0%} {ga:>9.0%} {gs:>7.0%}")

    print(f"\n  Labels file → {LABEL_PATH}")


if __name__ == "__main__":
    main()
