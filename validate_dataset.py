#!/usr/bin/env python3
"""
Dataset Validation & Scene Selection Pipeline
=============================================

Audits the GraspNet-style dataset, selects the best test scenes,
compares existing VLM labels against ground-truth annotations, and
generates a comprehensive Markdown report.

Usage:
    cd /home/arc02/Grasp_intent/grasp_pipeline
    python validate_dataset.py
"""

import os
import sys
import json
import glob
import datetime
import numpy as np
import cv2
from collections import Counter, defaultdict

from utils.label_normalize import (
    parse_annotation_xml,
    normalize_object_label,
    CANONICAL_GRASP_MAP,
    GT_OBJ_ID_TO_NAME,
)


# ═════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
REALSENSE_DIR  = os.path.join(BASE_DIR, "data", "realsense")
TRAIN1_DIR     = os.path.join(BASE_DIR, "data", "train_1")
LABEL_PATH     = os.path.join(BASE_DIR, "outputs", "labels", "labels.json")
OUTPUT_DIR     = os.path.join(BASE_DIR, "outputs")
AUDIT_JSON     = os.path.join(OUTPUT_DIR, "dataset_audit.json")
REPORT_MD      = os.path.join(OUTPUT_DIR, "dataset_validation_report.md")

# How many best scenes to recommend
N_RECOMMENDED = 5


# ═════════════════════════════════════════════════════════════════════
# 1. AUDIT: data/realsense/
# ═════════════════════════════════════════════════════════════════════

def audit_realsense():
    """Check file completeness and content validity of data/realsense/."""

    print("\n" + "=" * 60)
    print("  AUDITING: data/realsense/")
    print("=" * 60)

    subdirs = ["rgb", "depth", "annotations", "label", "meta"]
    expected_exts = {
        "rgb":         ".png",
        "depth":       ".png",
        "annotations": ".xml",
        "label":       ".png",
        "meta":        ".mat",
    }

    audit = {"path": REALSENSE_DIR, "subdirs": {}}

    for sd in subdirs:
        sd_path = os.path.join(REALSENSE_DIR, sd)
        if not os.path.isdir(sd_path):
            audit["subdirs"][sd] = {"exists": False, "count": 0}
            print(f"  [MISSING] {sd}/")
            continue

        ext = expected_exts[sd]
        files = sorted(glob.glob(os.path.join(sd_path, f"*{ext}")))
        audit["subdirs"][sd] = {
            "exists": True,
            "count": len(files),
            "first": os.path.basename(files[0]) if files else None,
            "last":  os.path.basename(files[-1]) if files else None,
        }
        print(f"  [{sd:>12}] {len(files):>4} files  "
              f"({os.path.basename(files[0]) if files else '—'}"
              f" → {os.path.basename(files[-1]) if files else '—'})")

    # Camera data
    cam_files = ["camK.npy", "cam0_wrt_table.npy", "camera_poses.npy"]
    audit["camera_files"] = {}
    for cf in cam_files:
        fp = os.path.join(REALSENSE_DIR, cf)
        exists = os.path.isfile(fp)
        audit["camera_files"][cf] = exists
        status = "✓" if exists else "✗ MISSING"
        print(f"  [{cf:>25}] {status}")

    return audit


# ═════════════════════════════════════════════════════════════════════
# 2. AUDIT: data/train_1/
# ═════════════════════════════════════════════════════════════════════

def audit_train1():
    """Check all 30 scenes in data/train_1/ for completeness."""

    print("\n" + "=" * 60)
    print("  AUDITING: data/train_1/")
    print("=" * 60)

    if not os.path.isdir(TRAIN1_DIR):
        print("  [MISSING] data/train_1/ directory not found!")
        return {"exists": False, "scenes": {}}

    scenes = sorted([d for d in os.listdir(TRAIN1_DIR)
                     if os.path.isdir(os.path.join(TRAIN1_DIR, d))
                     and d.startswith("scene_")])

    audit = {"exists": True, "scene_count": len(scenes), "scenes": {}}

    for scene in scenes:
        scene_path = os.path.join(TRAIN1_DIR, scene)
        scene_info = {"path": scene_path}

        # Object list
        obj_list_path = os.path.join(scene_path, "object_id_list.txt")
        if os.path.isfile(obj_list_path):
            with open(obj_list_path) as f:
                obj_ids = [int(x.strip()) for x in f.readlines() if x.strip()]
            scene_info["object_ids"] = obj_ids
            scene_info["object_count"] = len(obj_ids)
            scene_info["object_names"] = [
                GT_OBJ_ID_TO_NAME.get(oid, f"unknown_{oid}")
                for oid in obj_ids
            ]
        else:
            scene_info["object_ids"] = []
            scene_info["object_count"] = 0

        # Check realsense sub-folder
        rs_path = os.path.join(scene_path, "realsense")
        if os.path.isdir(rs_path):
            rgb_dir = os.path.join(rs_path, "rgb")
            depth_dir = os.path.join(rs_path, "depth")
            scene_info["has_realsense"] = True
            scene_info["rgb_count"] = len(glob.glob(os.path.join(rgb_dir, "*.png"))) if os.path.isdir(rgb_dir) else 0
            scene_info["depth_count"] = len(glob.glob(os.path.join(depth_dir, "*.png"))) if os.path.isdir(depth_dir) else 0
        else:
            scene_info["has_realsense"] = False
            scene_info["rgb_count"] = 0
            scene_info["depth_count"] = 0

        audit["scenes"][scene] = scene_info
        obj_str = ", ".join(scene_info.get("object_names", [])[:5])
        print(f"  {scene}: {scene_info['object_count']} objects  "
              f"RGB={scene_info['rgb_count']}  "
              f"Depth={scene_info['depth_count']}  "
              f"[{obj_str}...]")

    return audit


# ═════════════════════════════════════════════════════════════════════
# 3. PARSE ALL ANNOTATIONS & ANALYZE DIVERSITY
# ═════════════════════════════════════════════════════════════════════

def analyze_annotations():
    """Parse all annotation XMLs and compute per-view object stats."""

    print("\n" + "=" * 60)
    print("  ANALYZING ANNOTATIONS")
    print("=" * 60)

    ann_dir = os.path.join(REALSENSE_DIR, "annotations")
    if not os.path.isdir(ann_dir):
        print("  No annotations directory found.")
        return {}

    xml_files = sorted(glob.glob(os.path.join(ann_dir, "*.xml")))
    analysis = {"total_views": len(xml_files), "views": {}}

    all_object_names = Counter()

    for xml_path in xml_files:
        view_id = os.path.basename(xml_path).replace(".xml", "")
        objects = parse_annotation_xml(xml_path)
        canonical_names = [o["canonical_name"] for o in objects]
        all_object_names.update(canonical_names)

        analysis["views"][view_id] = {
            "object_count": len(objects),
            "objects": canonical_names,
            "obj_ids": [o["obj_id"] for o in objects],
        }

    analysis["unique_objects"] = dict(all_object_names)
    analysis["total_unique_objects"] = len(all_object_names)

    print(f"  Total views: {len(xml_files)}")
    print(f"  Unique objects across all views: {len(all_object_names)}")
    for name, count in all_object_names.most_common():
        print(f"    {name:>25}: appears in {count} views")

    return analysis


# ═════════════════════════════════════════════════════════════════════
# 4. SELECT BEST TEST SCENES
# ═════════════════════════════════════════════════════════════════════

def select_test_scenes(annotation_analysis):
    """
    Select N_RECOMMENDED diverse test scenes from data/realsense/.

    Criteria:
    - Spread across camera viewpoints (early / mid / late)
    - All 9 GT objects visible
    - Good depth coverage
    """

    print("\n" + "=" * 60)
    print("  SELECTING BEST TEST SCENES")
    print("=" * 60)

    views = annotation_analysis.get("views", {})
    if not views:
        print("  No views available for selection.")
        return []

    # Score each view
    scored_views = []
    for view_id, info in views.items():
        idx = int(view_id)

        # Check depth quality
        depth_path = os.path.join(REALSENSE_DIR, "depth", f"{view_id}.png")
        depth_quality = 0.5  # default
        if os.path.isfile(depth_path):
            depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_img is not None:
                valid_ratio = np.count_nonzero(depth_img) / max(depth_img.size, 1)
                depth_quality = valid_ratio

        score = {
            "view_id": view_id,
            "index": idx,
            "object_count": info["object_count"],
            "unique_objects": len(set(info["objects"])),
            "depth_quality": round(depth_quality, 3),
            "objects": info["objects"],
        }
        scored_views.append(score)

    # Sort by: max objects, then depth quality
    scored_views.sort(key=lambda x: (x["object_count"], x["depth_quality"]),
                      reverse=True)

    # Select from different viewpoint buckets to ensure angular diversity
    total = len(scored_views)
    bucket_size = total // N_RECOMMENDED
    selected = []
    used_buckets = set()

    for sv in scored_views:
        bucket = sv["index"] // max(bucket_size, 1)
        if bucket not in used_buckets and sv["object_count"] >= 9:
            selected.append(sv)
            used_buckets.add(bucket)
            if len(selected) >= N_RECOMMENDED:
                break

    # Fallback: if we didn't get enough, just take top-scoring
    if len(selected) < N_RECOMMENDED:
        for sv in scored_views:
            if sv not in selected:
                selected.append(sv)
                if len(selected) >= N_RECOMMENDED:
                    break

    print(f"\n  Top {N_RECOMMENDED} recommended test scenes:")
    for i, s in enumerate(selected):
        obj_str = ", ".join(s["objects"][:5])
        print(f"    {i+1}. Scene {s['view_id']}  "
              f"({s['object_count']} objects, "
              f"depth={s['depth_quality']:.1%})  "
              f"[{obj_str}...]")

    return selected


# ═════════════════════════════════════════════════════════════════════
# 5. COMPARE VLM LABELS vs GROUND TRUTH
# ═════════════════════════════════════════════════════════════════════

def compare_labels_vs_gt(annotation_analysis):
    """
    Cross-reference labels.json with GT annotations.
    For each processed scene, compute accuracy metrics.
    """

    print("\n" + "=" * 60)
    print("  COMPARING VLM LABELS vs GROUND TRUTH")
    print("=" * 60)

    if not os.path.isfile(LABEL_PATH):
        print("  No labels.json found — skipping comparison.")
        return {}

    with open(LABEL_PATH) as f:
        labels_data = json.load(f)

    views = annotation_analysis.get("views", {})
    comparison = {
        "scenes_analyzed": 0,
        "total_vlm_objects": 0,
        "total_gt_objects": 0,
        "label_accuracy": {},
        "grasp_accuracy": {},
        "per_scene": {},
        "object_confusion": defaultdict(lambda: Counter()),
        "grasp_distribution_by_object": defaultdict(lambda: Counter()),
        "normalized_label_distribution": Counter(),
    }

    for scene_id, scene_labels in labels_data.items():
        if scene_id not in views:
            continue

        gt_objects = views[scene_id]["objects"]
        gt_set = set(gt_objects)
        comparison["scenes_analyzed"] += 1
        comparison["total_gt_objects"] += len(gt_objects)

        # Normalize VLM labels
        vlm_objects = []
        for entry in scene_labels:
            raw_label = entry.get("object", "unknown")
            normalized = normalize_object_label(raw_label)
            grasp = entry.get("grasp", "unknown")
            vlm_objects.append({
                "raw": raw_label,
                "normalized": normalized,
                "grasp": grasp,
                "crop": entry.get("crop", ""),
            })
            comparison["normalized_label_distribution"][normalized] += 1
            comparison["grasp_distribution_by_object"][normalized][grasp] += 1

        comparison["total_vlm_objects"] += len(vlm_objects)

        # Check which GT objects were detected
        vlm_set = set(o["normalized"] for o in vlm_objects)
        detected_gt = gt_set & vlm_set
        missed_gt = gt_set - vlm_set
        false_positives = vlm_set - gt_set

        # Grasp accuracy (VLM grasp vs expected canonical grasp)
        correct_grasps = 0
        total_checked = 0
        for obj in vlm_objects:
            expected_grasp = CANONICAL_GRASP_MAP.get(obj["normalized"])
            if expected_grasp:
                total_checked += 1
                if obj["grasp"] == expected_grasp:
                    correct_grasps += 1

        comparison["per_scene"][scene_id] = {
            "gt_objects": list(gt_set),
            "vlm_objects": [o["normalized"] for o in vlm_objects],
            "detected": list(detected_gt),
            "missed": list(missed_gt),
            "false_positives": list(false_positives),
            "detection_recall": len(detected_gt) / max(len(gt_set), 1),
            "grasp_correct": correct_grasps,
            "grasp_total": total_checked,
            "grasp_accuracy": correct_grasps / max(total_checked, 1),
        }

    # Aggregate metrics
    if comparison["scenes_analyzed"] > 0:
        recalls = [s["detection_recall"]
                   for s in comparison["per_scene"].values()]
        grasp_accs = [s["grasp_accuracy"]
                      for s in comparison["per_scene"].values()
                      if s["grasp_total"] > 0]

        comparison["avg_detection_recall"] = sum(recalls) / len(recalls)
        comparison["avg_grasp_accuracy"] = (
            sum(grasp_accs) / len(grasp_accs) if grasp_accs else 0.0
        )

        print(f"  Scenes analyzed: {comparison['scenes_analyzed']}")
        print(f"  Avg detection recall: {comparison['avg_detection_recall']:.1%}")
        print(f"  Avg grasp accuracy:   {comparison['avg_grasp_accuracy']:.1%}")
        print(f"  Total VLM objects:    {comparison['total_vlm_objects']}")
        print(f"  Total GT objects:     {comparison['total_gt_objects']}")

    # Convert defaultdicts to regular dicts for JSON serialization
    comparison["object_confusion"] = dict(comparison["object_confusion"])
    comparison["grasp_distribution_by_object"] = {
        k: dict(v)
        for k, v in comparison["grasp_distribution_by_object"].items()
    }
    comparison["normalized_label_distribution"] = dict(
        comparison["normalized_label_distribution"]
    )

    return comparison


# ═════════════════════════════════════════════════════════════════════
# 6. GENERATE MARKDOWN REPORT
# ═════════════════════════════════════════════════════════════════════

def generate_report(realsense_audit, train1_audit, annotation_analysis,
                    selected_scenes, comparison):
    """Generate a comprehensive validation report."""

    print("\n" + "=" * 60)
    print("  GENERATING REPORT")
    print("=" * 60)

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []
    lines.append(f"# Dataset Validation Report")
    lines.append(f"**Generated:** {now}")
    lines.append("")

    # ── Section 1: Dataset Overview ──
    lines.append("## 1. Dataset Overview")
    lines.append("")

    # Realsense
    lines.append("### data/realsense/")
    lines.append("")
    lines.append("| Subdirectory | Files | Status |")
    lines.append("|---|---|---|")
    for sd, info in realsense_audit.get("subdirs", {}).items():
        status = f"✅ {info['count']} files" if info.get("exists") else "❌ Missing"
        lines.append(f"| `{sd}/` | {info.get('count', 0)} | {status} |")
    lines.append("")

    cam_data = realsense_audit.get("camera_files", {})
    lines.append("**Camera calibration files:**")
    for cf, exists in cam_data.items():
        lines.append(f"- `{cf}`: {'✅' if exists else '❌ Missing'}")
    lines.append("")

    # Train_1
    lines.append("### data/train_1/")
    lines.append("")
    if train1_audit.get("exists"):
        lines.append(f"**{train1_audit['scene_count']} scenes** found.")
        lines.append("")
        lines.append("| Scene | Objects | Object Names |")
        lines.append("|---|---|---|")
        for scene, info in sorted(train1_audit.get("scenes", {}).items()):
            names = ", ".join(info.get("object_names", [])[:5])
            lines.append(f"| `{scene}` | {info['object_count']} | {names} |")
        lines.append("")
    else:
        lines.append("> [!WARNING]\n> `data/train_1/` not found!")
        lines.append("")

    # ── Section 2: Annotation Analysis ──
    lines.append("## 2. Ground-Truth Object Analysis")
    lines.append("")

    ann = annotation_analysis
    lines.append(f"- **Total views (annotations):** {ann.get('total_views', 0)}")
    lines.append(f"- **Unique object classes:** {ann.get('total_unique_objects', 0)}")
    lines.append("")

    if ann.get("unique_objects"):
        lines.append("| Object | Appears in N views |")
        lines.append("|---|---|")
        for name, count in sorted(ann["unique_objects"].items(),
                                  key=lambda x: -x[1]):
            lines.append(f"| {name} | {count} |")
        lines.append("")

    # ── Section 3: Recommended Test Scenes ──
    lines.append("## 3. Recommended Test Scenes")
    lines.append("")
    lines.append("> [!TIP]")
    lines.append("> These scenes are selected for **maximum diversity**: different camera angles,")
    lines.append("> all 9 objects visible, and highest depth quality.")
    lines.append("")

    if selected_scenes:
        lines.append("| Rank | Scene ID | Objects | Depth Quality | Object List |")
        lines.append("|---|---|---|---|---|")
        for i, s in enumerate(selected_scenes):
            obj_str = ", ".join(s["objects"][:5])
            lines.append(
                f"| {i+1} | **{s['view_id']}** | "
                f"{s['object_count']} | "
                f"{s['depth_quality']:.1%} | "
                f"{obj_str}... |"
            )
        lines.append("")

        # Show usage example
        lines.append("### How to Test")
        lines.append("```bash")
        lines.append("cd /home/arc02/Grasp_intent/grasp_pipeline")
        for s in selected_scenes[:3]:
            lines.append(f"# Test scene {s['view_id']}")
            lines.append(f"python main.py  # (update rgb_path/depth_path to {s['view_id']})")
        lines.append("```")
        lines.append("")

    # ── Section 4: VLM vs GT Comparison ──
    lines.append("## 4. VLM Label Accuracy vs Ground Truth")
    lines.append("")

    if comparison and comparison.get("scenes_analyzed", 0) > 0:
        lines.append(f"| Metric | Value |")
        lines.append(f"|---|---|")
        lines.append(f"| Scenes analyzed | {comparison['scenes_analyzed']} |")
        lines.append(f"| Total VLM objects | {comparison['total_vlm_objects']} |")
        lines.append(f"| Avg detection recall | "
                     f"**{comparison.get('avg_detection_recall', 0):.1%}** |")
        lines.append(f"| Avg grasp accuracy | "
                     f"**{comparison.get('avg_grasp_accuracy', 0):.1%}** |")
        lines.append("")

        # Per-object grasp distribution
        grasp_dist = comparison.get("grasp_distribution_by_object", {})
        if grasp_dist:
            lines.append("### Grasp Distribution by Object (after normalization)")
            lines.append("")
            lines.append("| Object | Expected Grasp | Actual Distribution |")
            lines.append("|---|---|---|")
            for obj_name, grasps in sorted(grasp_dist.items()):
                expected = CANONICAL_GRASP_MAP.get(obj_name, "—")
                dist_str = ", ".join(f"{g}: {c}" for g, c in
                                    sorted(grasps.items(), key=lambda x: -x[1]))
                lines.append(f"| {obj_name} | {expected} | {dist_str} |")
            lines.append("")

        # Normalized label distribution
        label_dist = comparison.get("normalized_label_distribution", {})
        if label_dist:
            lines.append("### Normalized Label Distribution")
            lines.append("")
            lines.append("| Canonical Label | Count | % |")
            lines.append("|---|---|---|")
            total_labels = sum(label_dist.values())
            for label, count in sorted(label_dist.items(), key=lambda x: -x[1]):
                pct = count / max(total_labels, 1) * 100
                lines.append(f"| {label} | {count} | {pct:.1f}% |")
            lines.append("")

        # Worst scenes
        worst = sorted(comparison.get("per_scene", {}).items(),
                       key=lambda x: x[1]["detection_recall"])[:5]
        if worst:
            lines.append("### Lowest Detection Recall Scenes")
            lines.append("")
            lines.append("| Scene | Recall | Missed Objects |")
            lines.append("|---|---|---|")
            for sid, info in worst:
                missed = ", ".join(info.get("missed", []))
                lines.append(f"| {sid} | {info['detection_recall']:.0%} | "
                             f"{missed or '—'} |")
            lines.append("")
    else:
        lines.append("> [!NOTE]\n> No labels.json found — run `generate_labels.py` first.")
        lines.append("")

    # ── Section 5: Recommendations ──
    lines.append("## 5. Improvement Recommendations")
    lines.append("")
    lines.append("> [!IMPORTANT]")
    lines.append("> Key improvements to make the pipeline more robust and research-grade:")
    lines.append("")
    lines.append("1. **Label Normalization** — Apply `utils/label_normalize.py` "
                 "in the pipeline to map VLM outputs to canonical names "
                 "before grasp type inference.")
    lines.append("2. **Increase VLM Voting** — Use `N_DECISIONS=11, N_RETRIES=4` "
                 "(paper recommendation) instead of current `3/2` to reduce grasp "
                 "type inconsistency.")
    lines.append("3. **Constrained Object Vocabulary** — Feed SM3 a closed list of "
                 "valid object names from the annotation XML to prevent novel/"
                 "hallucinated labels.")
    lines.append("4. **Multi-Object Crop Filtering** — Skip crops with SM2 count > 1 "
                 "during training to avoid ambiguous grasp assignments.")
    lines.append("5. **Scene-Level Consistency Check** — Flag scenes where the same "
                 "object class receives contradictory grasp types.")
    lines.append("6. **Expand Dataset Coverage** — Run the pipeline on "
                 f"`data/train_1/` (30 additional scenes with varied objects) "
                 "to improve grasp type coverage especially for under-represented "
                 "types (Hook, Cylindrical).")
    lines.append("")

    report_text = "\n".join(lines)

    os.makedirs(os.path.dirname(REPORT_MD), exist_ok=True)
    with open(REPORT_MD, "w") as f:
        f.write(report_text)

    print(f"  Report saved → {REPORT_MD}")
    return report_text


# ═════════════════════════════════════════════════════════════════════
# 7. MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       GRASP PIPELINE — DATASET VALIDATION SUITE        ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # 1. Audit realsense data
    realsense_audit = audit_realsense()

    # 2. Audit train_1 data
    train1_audit = audit_train1()

    # 3. Analyze annotations
    annotation_analysis = analyze_annotations()

    # 4. Select best test scenes
    selected_scenes = select_test_scenes(annotation_analysis)

    # 5. Compare VLM labels vs GT
    comparison = compare_labels_vs_gt(annotation_analysis)

    # 6. Save audit JSON
    audit_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "realsense": realsense_audit,
        "train1": train1_audit,
        "annotations": {
            "total_views": annotation_analysis.get("total_views", 0),
            "unique_objects": annotation_analysis.get("unique_objects", {}),
        },
        "recommended_scenes": [
            {"view_id": s["view_id"],
             "objects": s["objects"],
             "depth_quality": s["depth_quality"]}
            for s in selected_scenes
        ],
        "label_comparison": {
            "scenes_analyzed": comparison.get("scenes_analyzed", 0),
            "avg_detection_recall": comparison.get("avg_detection_recall", 0),
            "avg_grasp_accuracy": comparison.get("avg_grasp_accuracy", 0),
            "normalized_label_distribution": comparison.get(
                "normalized_label_distribution", {}),
        },
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(AUDIT_JSON, "w") as f:
        json.dump(audit_data, f, indent=2)
    print(f"\n  Audit data saved → {AUDIT_JSON}")

    # 7. Generate Markdown report
    generate_report(realsense_audit, train1_audit, annotation_analysis,
                    selected_scenes, comparison)

    print("\n" + "=" * 60)
    print("  VALIDATION COMPLETE")
    print("=" * 60)
    print(f"  - Audit JSON:  {AUDIT_JSON}")
    print(f"  - Report:      {REPORT_MD}")
    print(f"  - Recommended test scenes: "
          + ", ".join(s["view_id"] for s in selected_scenes))


if __name__ == "__main__":
    main()
