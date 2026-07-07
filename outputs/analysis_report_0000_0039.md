# Dexterous Grasp Label Generation — Analysis Report
**Dataset:** Scenes 0000 – 0039 (40 scenes)  
**Pipeline:** SAM + Qwen2.5-VL-3B-Instruct (N_DECISIONS=3, N_RETRIES=2)  
**Date:** 29 May 2026

---

## 1. Dataset Overview

| Metric | Value |
|---|---|
| Total scenes processed | 40 |
| Total object instances labeled | **382** |
| Average objects per scene | 9.6 |
| Min objects in a single scene | 7 (scenes 0013, 0014) |
| Max objects in a single scene | 10 |

---

## 2. Grasp Type Distribution — CRITICAL IMBALANCE

| Grasp Type | Count | % | Expected Role |
|---|---|---|---|
| Pinch grasp | 215 | **56.3 %** | Flat/small objects |
| Spherical grasp | 77 | 20.2 % | Round/bulky objects |
| Lateral pinch | 71 | 18.6 % | Flat-sided objects |
| Pen-holding grasp | 12 | 3.1 % | Elongated tools |
| Cylindrical grasp | 5 | 1.3 % | Bottles/cylinders |
| Button-press grasp | **2** | **0.5 %** | Push-type interaction |
| **Hook grasp** | **0** | **0 %** | **NEVER USED** |

### Key Problems
- **Hook grasp** is defined in the taxonomy but appears **zero times** across 382 instances. Either the dataset contains no hook-graspable objects *or* the VLM never selects it.
- **Button-press grasp** appears only twice — once wrongly assigned to a `cheez-it` cracker (scene 0013) and once to `cheetos` (scene 0038). Neither is a button-press scenario.
- **Pinch grasp dominates at >56%**, suggesting the VLM defaults to it under uncertainty rather than making a semantically grounded choice.
- The distribution is heavily skewed towards two classes (Pinch + Lateral pinch ≈ 75%), which will harm downstream classifier training.

---

## 3. Object Label Normalization — FRAGMENTED VOCABULARY

The pipeline produces inconsistent label strings for the same physical object class:

| Canonical Object | Label Variants Found | Total Instances | Notes |
|---|---|---|---|
| Cheez-It cracker | `cheez-it` (217), `cheezit` (13), `cheese crackers` (6), `cheez-it box` (1) | **237** | 4 surface forms for 1 object |
| Cheese / Cracker | `cheese` (45) | 45 | Ambiguous — could be cheez-it OR actual cheese block |
| Cheetos | `cheetos` (25) | 25 | Consistent |
| Banana | `banana` (55) | 55 | Consistent |
| Knife | `knife` (9) | 9 | Consistent |
| Pen | `pen` (3) | 3 | Consistent |

### Impact
- **`cheese` (45 instances) is semantically ambiguous.** In scenes 0014–0019 it refers to cheez-it crackers (gets `Pinch grasp`) but in scenes 0017–0025 the same label gets `Spherical grasp`. This breaks any training that uses `object` as a feature.
- **`cheezit` vs `cheez-it`** creates a vocabulary split: 13 instances will mismatch if a lookup or embedding relies on exact string matching.
- **Fix needed:** SM3 prompt should be constrained to a closed vocabulary, or a post-processing normalisation step must canonicalize labels.

---

## 4. Intra-Class Grasp Inconsistency — BANANA CASE STUDY

Banana (55 instances) receives three different grasp types with no systematic pattern:

| Grasp | Count | % of Banana |
|---|---|---|
| Spherical grasp | 36 | 65.5 % |
| Lateral pinch | 14 | 25.5 % |
| Pinch grasp | 5 | **9.1 % — incorrect** |

### Intra-Scene Contradictions (same scene, same object class, different grasp)

| Scene | Banana Grasp Assignments |
|---|---|
| 0001 | `Spherical grasp`, `Pinch grasp` |
| 0006 | `Lateral pinch`, `Spherical grasp` |
| 0008 | `Spherical grasp`, `Lateral pinch` |
| 0013 | `Lateral pinch`, `Spherical grasp` |
| 0015 | `Lateral pinch`, `Spherical grasp` ×3 |

A banana is a banana — it should receive a **consistent** grasp assignment within the same scene. These contradictions indicate the majority-vote + judgment verification mechanism (Section 3.3) is not resolving grasp type reliably for objects with moderate ambiguity.

### Cheez-It Grasp Inconsistency
- `cheez-it` alone: Pinch grasp (171), Lateral pinch (45), Button-press (1)
- The Pinch/Lateral split is legitimately debatable; the Button-press assignment is clearly wrong.

---

## 5. False Positives — SAM Leakage into VLM Pipeline

Four object instances received nonsensical labels, indicating SAM produced a mask that passed the depth filter but is not a graspable object:

| Scene | Crop | Assigned Label | Assigned Grasp | Diagnosis |
|---|---|---|---|---|
| 0010 | object_3.png | `slime` | Spherical grasp | Likely reflection/blob artefact |
| 0021 | object_1.png | `window` | Cylindrical grasp | Background structure leaked through depth filter |
| 0039 | object_5.png | `pepper` | Spherical grasp | Novel object — correct in isolation but not in dataset vocabulary |
| 0039 | object_6.png | `USB drive` | Cylindrical grasp | Novel object — correct in isolation but not in dataset vocabulary |

The `window` case is the most concerning — a structural background element survived depth-based filtering and was passed to the VLM. It suggests the depth threshold or containment threshold needs tightening for scenes with background walls visible.

---

## 6. Low-Yield Scenes — SAM Under-Segmentation

Seven scenes produced fewer than 9 objects, while the dataset typically yields 10:

| Scene | Objects Found | Deficit |
|---|---|---|
| 0013 | 7 | −3 |
| 0014 | 7 | −3 |
| 0011 | 8 | −2 |
| 0016 | 8 | −2 |
| 0029 | 8 | −2 |
| 0030 | 8 | −2 |
| 0034 | 8 | −2 |

These deficits likely reflect over-aggressive mask post-processing (small-area or IoU deduplication) or depth filtering rejecting too many valid masks. Each missing object is a missed training sample.

---

## 7. Object Count Overestimation (Multi-Object Crops)

The SM2 submodule is supposed to flag crops containing more than one object so they can be deprioritized or re-processed. However, high counts appear frequently:

| Reported Count | Instances | % |
|---|---|---|
| 1 | 195 | 51.0 % |
| 2 | 184 | 48.2 % |
| 3 | 1 | 0.3 % |
| 4 | 2 | 0.5 % |

- **48.7% of crops are reported as multi-object.** This is extremely high for a segmentation pipeline — SAM should be producing single-object masks most of the time.
- Count=3 and count=4 crops (3 instances total) are almost certainly over-lapping or poorly separated masks that SAM failed to split.
- Multi-object crops should ideally be excluded from grasp training since the correct reference object is ambiguous.

---

## 8. Monotone Grasp Scenes — VLM Collapse

Two scenes have **all objects assigned the exact same grasp type**, suggesting the VLM converged to a default answer and stopped discriminating:

| Scene | Objects | Unique Grasp | Implication |
|---|---|---|---|
| 0026 | 10 | `Pinch grasp` only | All 10 objects, including a banana, labeled Pinch |
| 0033 | 10 | `Pinch grasp` only | All 10 objects, including a banana, labeled Pinch |

In scene 0026, `banana` (object_5) is assigned `Pinch grasp` — directly contradicting 36 other banana instances. In scene 0033, `banana` (object_3) also gets `Pinch grasp`. This is a clear case of **model collapse** within those scenes, where the judgment VLM accepted a wrong answer without retrying.

---

## 9. Grasp Taxonomy Coverage Gap

Of the 7 grasp types defined in the taxonomy, only 5 appear at all, and 2 appear fewer than 5 times total:

```
Cylindrical grasp   ████░░░░░░░░░░░░░░░░  5 uses    (0000: duct tape, shampoo; 0012: cheetos; 0021: window(!); 0039: USB drive)
Spherical grasp     █████████████░░░░░░░░ 77 uses
Pinch grasp         ████████████████████ 215 uses
Lateral pinch       ██████████████░░░░░░  71 uses
Hook grasp          ░░░░░░░░░░░░░░░░░░░░   0 uses  ← MISSING
Pen-holding grasp   ██░░░░░░░░░░░░░░░░░░  12 uses
Button-press grasp  ░░░░░░░░░░░░░░░░░░░░   2 uses  ← NEAR MISSING
```

For a dataset intended to train a dexterous grasp classifier, this coverage is insufficient. A model trained on these labels will have no ability to predict Hook grasp and very limited ability to predict Button-press, Cylindrical, or Pen-holding grasps.

---

## 10. Summary: Where Improvement is Needed

| Priority | Problem | Affected Instances | Recommended Fix |
|---|---|---|---|
| 🔴 **Critical** | Hook grasp never assigned | 0 / 382 | Add scenes with hook-graspable objects (handles, rings, bags) and force taxonomy coverage check |
| 🔴 **Critical** | Grasp imbalance (Pinch 56%) | 215 / 382 | Diversify dataset with more cylindrical, spherical, and hook objects; verify VLM prompts |
| 🔴 **Critical** | Object label fragmentation (cheez-it/cheezit/cheese) | ~282 / 382 | Add post-processing normalisation map; or constrain SM3 to closed vocabulary list |
| 🟠 **High** | Intra-scene grasp contradictions for banana | 5 scenes | Increase N_DECISIONS (3→11 as per paper) and N_RETRIES (2→4); add object-level consistency check |
| 🟠 **High** | 48.7% multi-object crops | 187 / 382 | Tighten SAM mask post-processing; filter all count>1 crops from training |
| 🟠 **High** | VLM collapse (monotone scenes 0026, 0033) | 20 / 382 | Increase N_RETRIES; add scene-level sanity check flagging scenes with 0 grasp variety |
| 🟡 **Medium** | False positive SAM masks ("window", "slime") | 4 / 382 | Tighten `CONTAINMENT_THRESHOLD` and `DEPTH_FILTER`; add SM1 confidence threshold |
| 🟡 **Medium** | Low-yield scenes (7–8 objects) | 7 scenes | Lower `SMALL_AREA_THRESHOLD` or `OVERLAP_IOU_THRESHOLD` to recover missed masks |
| 🟡 **Medium** | Button-press grasp mis-assigned | 2 / 382 | Audit SM4 prompts; remove these 2 instances from training |
| 🟢 **Low** | "cheese" ambiguity (45 instances) | 45 / 382 | Manual review of scenes 0014–0025 to distinguish cheez-it from cheese blocks |

---

## Appendix: Per-Scene Summary Table

| Scene | Objects | Unique Labels | Unique Grasps | Flag |
|---|---|---|---|---|
| 0000 | 10 | 5 (duct tape, crackers, shampoo, banana, peach) | 3 | Diverse — good baseline |
| 0001 | 10 | 4 (cheez-it, banana, pen) | 4 | Banana grasp contradiction |
| 0002 | 10 | 2 (cheez-it, banana) | 3 | Banana → Spherical grasp (wrong) |
| 0003 | 10 | 2 (cheez-it, banana) | 2 | |
| 0004 | 10 | 2 (cheez-it, banana) | 2 | |
| 0005 | 10 | 2 (cheez-it, banana) | 2 | |
| 0006 | 10 | 3 (cheez-it, knife, banana) | 3 | Banana grasp contradiction |
| 0007 | 9 | 2 (cheez-it, cheese) | 3 | "cheese" label ambiguous |
| 0008 | 10 | 2 (cheez-it, banana) | 3 | Banana grasp contradiction |
| 0009 | 10 | 1 (cheezit) | 2 | Label inconsistency: cheezit |
| 0010 | 10 | 3 (cheez-it, slime, banana) | 3 | **False positive: slime** |
| 0011 | 8 | 2 (cheez-it, knife) | 2 | Under-segmentation |
| 0012 | 9 | 2 (cheetos, banana) | 3 | count=4 crop |
| 0013 | 7 | 4 (cheez-it, cheese, banana, knife) | 4 | Under-seg; **Button-press on cheez-it** |
| 0014 | 7 | 3 (cheetos, cheese, banana) | 3 | Under-seg; count=4 crop |
| 0015 | 10 | 2 (cheez-it, banana, cheese) | 3 | Banana grasp contradiction |
| 0016 | 8 | 3 (cheez-it, banana, cheese) | 2 | Under-segmentation |
| 0017 | 10 | 2 (cheetos, cheese) | 2 | "cheese" ambiguous |
| 0018 | 10 | 3 (cheez-it, banana, cheese) | 3 | |
| 0019 | 10 | 3 (cheez-it, banana, cheese, pen) | 3 | |
| 0020 | 10 | 2 (cheez-it, banana) | 2 | |
| 0021 | 10 | 3 (cheetos, cheese, pen) | 3 | **False positive: window** |
| 0022 | 10 | 3 (cheez-it, banana, cheese) | 3 | |
| 0023 | 10 | 4 (cheez-it, cheese, knife, cheez-it box) | 3 | "cheez-it box" variant |
| 0024 | 10 | 3 (cheez-it, cheese, knife) | 3 | |
| 0025 | 10 | 3 (cheez-it, cheese, cheetos, knife) | 3 | |
| 0026 | 10 | 2 (cheez-it, banana) | **1** | **Monotone collapse: all Pinch** |
| 0027 | 10 | 3 (cheez-it, knife, banana) | 2 | |
| 0028 | 10 | 2 (cheez-it, banana) | 2 | Banana → Pinch grasp (wrong) |
| 0029 | 8 | 1 (cheez-it/cheezit) | 2 | Under-seg; label variant |
| 0030 | 8 | 1 (cheez-it) | 2 | Under-segmentation |
| 0031 | 10 | 2 (cheez-it, banana) | 2 | |
| 0032 | 10 | 3 (cheez-it, knife, banana) | 2 | Banana → Pinch grasp (wrong) |
| 0033 | 10 | 3 (cheez-it, cheetos, banana) | **1** | **Monotone collapse: all Pinch** |
| 0034 | 8 | 2 (cheez-it, knife) | 2 | Under-segmentation |
| 0035 | 10 | 2 (cheez-it, banana) | 3 | |
| 0036 | 10 | 2 (cheez-it, banana) | 2 | |
| 0037 | 10 | 2 (cheez-it, banana) | 2 | |
| 0038 | 10 | 5 (cheez-it, cheese crackers, cheese, banana, cheetos) | 3 | **Button-press on cheetos**; label fragmentation |
| 0039 | 10 | 6 (cheez-it, cheese, cheese crackers, banana, pepper, USB drive) | 4 | **FP: pepper/USB drive**; most diverse scene |
