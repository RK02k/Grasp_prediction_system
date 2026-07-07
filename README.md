# Grasp Prediction System

Automated dexterous grasp type annotation pipeline using **SAM** (Segment Anything Model) and **Qwen2.5-VL** (Vision-Language Model).

Implements the framework from:
> *"From-scratch dexterous grasp type annotation with SAM and lightweight vision-language models"*, Wang & Cheng, Pattern Recognition Letters 2026.

---

## Pipeline Overview

```
RGB-D Image
    │
    ▼
┌──────────────┐
│  SAM Segment │  → Automatic mask generation (sam_vit_l)
└──────┬───────┘
       ▼
┌──────────────┐
│ Post-Process │  → Area filter → Depth filter → IoU dedup
└──────┬───────┘
       ▼
┌──────────────┐
│  Save Crops  │  → Individual RGB + depth crops per object
└──────┬───────┘
       ▼
┌──────────────────────────────────────────────┐
│          VLM Pipeline (4 Submodules)         │
│  SM1: Object vs Background    → True/False   │
│  SM2: Object Count            → Number       │
│  SM3: Object Category         → Label        │
│  SM4: Grasp Type              → Grasp Type   │
│  (each with majority voting + verification)  │
└──────────────────────────────────────────────┘
       ▼
   labels.json
```

---

## Project Structure

```
grasp_pipeline/
├── main.py                    # Single-frame demo (quick sanity check)
├── generate_labels.py         # Full label generation for realsense data
├── run_train1_pipeline.py     # Batch pipeline for train_1 scenes (30 scenes)
├── validate_dataset.py        # Dataset audit & validation report generator
├── test_pipeline.py           # Pipeline testing script
├── requirements.txt           # Python dependencies
│
├── sam_module/
│   └── sam_segment.py         # SAM segmentation wrapper
│
├── vlm_module/
│   ├── qwen_grasp.py          # Qwen2.5-VL-3B model wrapper
│   └── pipeline.py            # VLM 4-submodule pipeline with voting
│
├── utils/
│   ├── mask_filter.py         # Small mask area filter
│   ├── mask_postprocess.py    # IoU dedup + complementary set computation
│   ├── depth_filter.py        # Depth-based background removal
│   ├── object_crop.py         # Save per-object RGB/depth crops
│   ├── crop_splitter.py       # Hierarchical re-segmentation for multi-object crops
│   ├── visualize.py           # Mask overlay visualization
│   └── label_normalize.py     # VLM label → canonical name mapping
│
├── prompts/
│   └── grasp_intent.txt       # Grasp type prompt definitions
│
├── data/                      # Dataset (not tracked in git)
│   ├── realsense/             # Single-scene RGB-D + annotations
│   │   ├── rgb/
│   │   ├── depth/
│   │   ├── annotations/
│   │   └── label/
│   └── train_1/               # 30 GraspNet-style scenes
│       └── scene_XXXX/
│           ├── realsense/rgb/
│           ├── realsense/depth/
│           └── object_id_list.txt
│
└── outputs/                   # Generated outputs (not tracked in git)
    ├── crops/
    ├── labels/
    └── scene_overlays/
```

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/RK02k/Grasp_prediction_system.git
cd Grasp_prediction_system
```

### 2. Create Virtual Environment

```bash
python3 -m venv grasp_env
source grasp_env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download SAM Weights

Download the **SAM ViT-L** checkpoint and place it in the project root:

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
```

### 5. Prepare Dataset

Place your RGB-D data in the `data/` directory following the structure shown above.

- **Realsense data**: `data/realsense/rgb/` and `data/realsense/depth/` (PNG images named `0000.png`, `0001.png`, etc.)
- **Train_1 data**: `data/train_1/scene_XXXX/realsense/rgb/` and `depth/`

---

## Usage

### Quick Demo (Single Frame)

Run a quick sanity check on a single frame:

```bash
python main.py
```

This processes `data/realsense/rgb/0000.png` through SAM segmentation, applies mask filtering, and saves object crops.

### Generate Labels (Realsense Data)

Run the full SAM + VLM pipeline on the realsense dataset:

```bash
python generate_labels.py
```

**Output:** `outputs/labels/labels.json` with object names and grasp types for each detected object.

### Batch Pipeline (Train_1 — 30 Scenes)

Process all 30 scenes in `data/train_1/`:

```bash
python run_train1_pipeline.py
```

**Options:**
```bash
# Process specific scene range
python run_train1_pipeline.py --start 0 --end 9

# Dry run — SAM + crops only, skip VLM
python run_train1_pipeline.py --no-vlm
```

**Output:** `outputs/train1/labels/train1_labels.json`

> **Note:** The pipeline supports crash-safe resume — already-processed scenes are skipped on restart.

### Validate Dataset

Run the dataset audit and validation suite:

```bash
python validate_dataset.py
```

**Output:**
- `outputs/dataset_audit.json` — structured audit data
- `outputs/dataset_validation_report.md` — comprehensive Markdown report

---

## VLM Submodules

The VLM pipeline consists of 4 sequential submodules, each using **majority voting** (11 decisions) with a **judgment verification** step (4 retries):

| Submodule | Task | Output |
|-----------|------|--------|
| **SM1** | Object vs. Background | `True` / `False` |
| **SM2** | Object Count | Integer |
| **SM3** | Object Category | Semantic label (e.g., `banana`, `mug`) |
| **SM4** | Grasp Type | Dexterous grasp type |

### Supported Grasp Types

- Lateral Pinch
- Palmar Pinch
- Tripod Pinch
- Power Sphere
- Power Cylinder
- Hook
- Platform
- Precision Disk

---

## Configuration

Key parameters in `generate_labels.py` and `run_train1_pipeline.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_DECISIONS` | 11 | Majority voting rounds per submodule |
| `N_RETRIES` | 4 | Retry cycles on format/judgment rejection |
| `SMALL_AREA_THRESHOLD` | 5000 px | Minimum mask area |
| `MAX_AREA_THRESHOLD` | 245760 px | Maximum mask area (~27% of 1280×720) |
| `OVERLAP_IOU_THRESHOLD` | 0.4 | IoU threshold for mask deduplication |

---

## References

- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
- Wang & Cheng, *"From-scratch dexterous grasp type annotation with SAM and lightweight vision-language models"*, Pattern Recognition Letters, 2026.
