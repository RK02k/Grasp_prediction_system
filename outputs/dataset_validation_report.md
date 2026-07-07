# Dataset Validation Report
**Generated:** 2026-06-12 10:11

## 1. Dataset Overview

### data/realsense/

| Subdirectory | Files | Status |
|---|---|---|
| `rgb/` | 256 | ✅ 256 files |
| `depth/` | 256 | ✅ 256 files |
| `annotations/` | 256 | ✅ 256 files |
| `label/` | 256 | ✅ 256 files |
| `meta/` | 256 | ✅ 256 files |

**Camera calibration files:**
- `camK.npy`: ✅
- `cam0_wrt_table.npy`: ✅
- `camera_poses.npy`: ✅

### data/train_1/

**30 scenes** found.

| Scene | Objects | Object Names |
|---|---|---|
| `scene_0000` | 9 | peach, cracker_box, banana, pear, flat_screwdriver |
| `scene_0001` | 9 | head_shoulders_care, banana, pear, flat_screwdriver, tape |
| `scene_0002` | 9 | cracker_box, banana, dish, peach, pear |
| `scene_0003` | 9 | glue, mario, shampoo, hanoi_tower, sugar_box |
| `scene_0004` | 9 | thread, glue, hanoi_tower, extra_large_clamp, shampoo |
| `scene_0005` | 9 | hanoi_tower, stapler, shampoo, glue, sugar_box |
| `scene_0006` | 10 | power_drill, scissors, strawberry, toy_airplane_i, sum37_secret_repair |
| `scene_0007` | 10 | power_drill, scissors, strawberry, toy_airplane_i, sum37_secret_repair |
| `scene_0008` | 10 | power_drill, scissors, strawberry, toy_airplane_i, sum37_secret_repair |
| `scene_0009` | 11 | mug, plum, knife, toy_airplane_d, toy_airplane_j |
| `scene_0010` | 11 | mug, plum, knife, toy_airplane_d, toy_airplane_j |
| `scene_0011` | 11 | mug, plum, knife, toy_airplane_d, toy_airplane_j |
| `scene_0012` | 8 | cracker_box, scissors, large_elephant, darlie_box, dabao_facewash |
| `scene_0013` | 8 | cracker_box, darlie_box, large_elephant, scissors, tape |
| `scene_0014` | 8 | cracker_box, scissors, large_elephant, darlie_box, dabao_facewash |
| `scene_0015` | 11 | mug, power_drill, peach, pear, knife |
| `scene_0016` | 11 | mug, power_drill, peach, pear, knife |
| `scene_0017` | 11 | mug, power_drill, peach, pear, knife |
| `scene_0018` | 9 | banana, flat_screwdriver, toy_airplane_f, sum37_secret_repair, dabao_wash_soup |
| `scene_0019` | 9 | banana, flat_screwdriver, dabao_wash_soup, nzskincare_mouth_rinse, dabao_sod |
| `scene_0020` | 9 | banana, flat_screwdriver, dabao_wash_soup, nzskincare_mouth_rinse, dabao_sod |
| `scene_0021` | 10 | strawberry, toy_airplane_d, stapler, dish, camel |
| `scene_0022` | 10 | strawberry, toy_airplane_d, stapler, dish, camel |
| `scene_0023` | 10 | strawberry, toy_airplane_d, stapler, dish, camel |
| `scene_0024` | 10 | peach, pear, flat_screwdriver, dabao_wash_soup, hanoi_tower |
| `scene_0025` | 10 | peach, pear, flat_screwdriver, dabao_wash_soup, hanoi_tower |
| `scene_0026` | 10 | mug, peach, pear, flat_screwdriver, toy_airplane_f |
| `scene_0027` | 10 | strawberry, plum, knife, toy_airplane_i, nzskincare_mouth_rinse |
| `scene_0028` | 10 | strawberry, plum, knife, toy_airplane_i, nzskincare_mouth_rinse |
| `scene_0029` | 8 | cracker_box, power_drill, scissors, shampoo, darlie_box |

## 2. Ground-Truth Object Analysis

- **Total views (annotations):** 256
- **Unique object classes:** 9

| Object | Appears in N views |
|---|---|
| head_shoulders_care | 256 |
| banana | 256 |
| pear | 256 |
| flat_screwdriver | 256 |
| tape | 256 |
| dish | 256 |
| cracker_box | 256 |
| camel | 256 |
| peach | 256 |

## 3. Recommended Test Scenes

> [!TIP]
> These scenes are selected for **maximum diversity**: different camera angles,
> all 9 objects visible, and highest depth quality.

| Rank | Scene ID | Objects | Depth Quality | Object List |
|---|---|---|---|---|
| 1 | **0055** | 9 | 96.7% | head_shoulders_care, banana, pear, flat_screwdriver, tape... |
| 2 | **0035** | 9 | 96.4% | head_shoulders_care, banana, pear, flat_screwdriver, tape... |
| 3 | **0119** | 9 | 95.5% | head_shoulders_care, banana, pear, flat_screwdriver, tape... |
| 4 | **0240** | 9 | 95.4% | head_shoulders_care, banana, pear, flat_screwdriver, tape... |
| 5 | **0158** | 9 | 95.0% | head_shoulders_care, banana, pear, flat_screwdriver, tape... |

### How to Test
```bash
cd /home/arc02/Grasp_intent/grasp_pipeline
# Test scene 0055
python main.py  # (update rgb_path/depth_path to 0055)
# Test scene 0035
python main.py  # (update rgb_path/depth_path to 0035)
# Test scene 0119
python main.py  # (update rgb_path/depth_path to 0119)
```

## 4. VLM Label Accuracy vs Ground Truth

| Metric | Value |
|---|---|
| Scenes analyzed | 40 |
| Total VLM objects | 382 |
| Avg detection recall | **21.7%** |
| Avg grasp accuracy | **19.4%** |

### Grasp Distribution by Object (after normalization)

| Object | Expected Grasp | Actual Distribution |
|---|---|---|
| banana | Lateral pinch | Spherical grasp: 36, Lateral pinch: 14, Pinch grasp: 5 |
| cracker_box | Lateral pinch | Pinch grasp: 210, Lateral pinch: 57, Spherical grasp: 38, Button-press grasp: 2, Cylindrical grasp: 1 |
| flat_screwdriver | Pen grip | Pen-holding grasp: 3 |
| head_shoulders_care | Cylindrical grasp | Cylindrical grasp: 1 |
| knife | Pen grip | Pen-holding grasp: 9 |
| peach | Spherical grasp | Spherical grasp: 1 |
| pepper | — | Spherical grasp: 1 |
| slime | — | Spherical grasp: 1 |
| tape | Cylindrical grasp | Cylindrical grasp: 1 |
| usb drive | — | Cylindrical grasp: 1 |
| window | — | Cylindrical grasp: 1 |

### Normalized Label Distribution

| Canonical Label | Count | % |
|---|---|---|
| cracker_box | 308 | 80.6% |
| banana | 55 | 14.4% |
| knife | 9 | 2.4% |
| flat_screwdriver | 3 | 0.8% |
| tape | 1 | 0.3% |
| head_shoulders_care | 1 | 0.3% |
| peach | 1 | 0.3% |
| slime | 1 | 0.3% |
| window | 1 | 0.3% |
| pepper | 1 | 0.3% |
| usb drive | 1 | 0.3% |

### Lowest Detection Recall Scenes

| Scene | Recall | Missed Objects |
|---|---|---|
| 0007 | 11% | banana, head_shoulders_care, pear, tape, peach, flat_screwdriver, dish, camel |
| 0011 | 11% | banana, head_shoulders_care, pear, tape, peach, flat_screwdriver, dish, camel |
| 0024 | 11% | banana, head_shoulders_care, pear, tape, peach, flat_screwdriver, dish, camel |
| 0025 | 11% | banana, head_shoulders_care, pear, tape, peach, flat_screwdriver, dish, camel |
| 0029 | 11% | banana, head_shoulders_care, pear, tape, peach, flat_screwdriver, dish, camel |

## 5. Improvement Recommendations

> [!IMPORTANT]
> Key improvements to make the pipeline more robust and research-grade:

1. **Label Normalization** — Apply `utils/label_normalize.py` in the pipeline to map VLM outputs to canonical names before grasp type inference.
2. **Increase VLM Voting** — Use `N_DECISIONS=11, N_RETRIES=4` (paper recommendation) instead of current `3/2` to reduce grasp type inconsistency.
3. **Constrained Object Vocabulary** — Feed SM3 a closed list of valid object names from the annotation XML to prevent novel/hallucinated labels.
4. **Multi-Object Crop Filtering** — Skip crops with SM2 count > 1 during training to avoid ambiguous grasp assignments.
5. **Scene-Level Consistency Check** — Flag scenes where the same object class receives contradictory grasp types.
6. **Expand Dataset Coverage** — Run the pipeline on `data/train_1/` (30 additional scenes with varied objects) to improve grasp type coverage especially for under-represented types (Hook, Cylindrical).
