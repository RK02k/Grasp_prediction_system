"""
Label normalization utilities for the grasp pipeline.

Maps noisy VLM object labels to canonical names derived from the
GraspNet-1Billion annotation vocabulary.  Also provides helpers to
parse the XML annotation files for ground-truth comparison.
"""

import os
import re
import xml.etree.ElementTree as ET


# ─────────────────────────────────────────────────────────────────────
# Ground-truth object name map  (obj_id → clean name)
# Derived from the annotation XMLs in data/realsense/annotations/
# ─────────────────────────────────────────────────────────────────────

GT_OBJ_ID_TO_NAME = {
    # ── realsense flat-dataset objects ──────────────────────────
    0:  "cracker_box",
    5:  "banana",
    14: "peach",
    15: "pear",
    20: "flat_screwdriver",
    46: "dish",
    48: "camel",
    66: "head_shoulders_care",
    70: "tape",
    # ── train_1 known objects ────────────────────────────────────
    2:  "sugar_box",
    21: "large_clamp",
    22: "extra_large_clamp",
    41: "hanoi_tower",
    43: "mario",
    44: "stapler",
    52: "shampoo",
    60: "glue",
    62: "thread",
    # ── train_1 previously-unknown objects (discovered from XMLs) ─
    7:  "mug",
    8:  "power_drill",
    9:  "scissors",
    11: "strawberry",
    17: "plum",
    18: "knife",
    26: "toy_airplane_d",
    27: "toy_airplane_f",
    29: "toy_airplane_i",
    30: "toy_airplane_j",
    34: "sum37_secret_repair",
    36: "dabao_wash_soup",
    37: "nzskincare_mouth_rinse",
    38: "dabao_sod",
    40: "kispa_cleanser",
    51: "large_elephant",
    56: "gorilla",
    57: "weiquan",
    58: "darlie_box",
    61: "dabao_facewash",
    63: "head_shoulders_supreme",
    69: "hippo",
}

# ─────────────────────────────────────────────────────────────────────
# VLM label → canonical name mapping
# Covers common misspellings, synonyms, and partial matches observed
# in the existing labels.json output.
# ─────────────────────────────────────────────────────────────────────

OBJECT_ALIAS_MAP = {
    # cracker_box aliases
    "cheez-it":          "cracker_box",
    "cheezit":           "cracker_box",
    "cheez-it box":      "cracker_box",
    "cheese crackers":   "cracker_box",
    "crackers":          "cracker_box",
    "cracker box":       "cracker_box",
    "cracker":           "cracker_box",
    "cheez it":          "cracker_box",

    # banana
    "banana":            "banana",

    # peach
    "peach":             "peach",

    # pear
    "pear":              "pear",

    # flat_screwdriver aliases
    "screwdriver":              "flat_screwdriver",
    "flat screwdriver":         "flat_screwdriver",
    "flat head screwdriver":    "flat_screwdriver",
    "flathead screwdriver":     "flat_screwdriver",
    # NOTE: "knife" and "pen" intentionally NOT aliased here — both exist as
    # separate GT objects (knife=id18) and the mappings caused false positives.

    # tape aliases
    "tape":              "tape",
    "duct tape":         "tape",
    "roll of tape":      "tape",

    # dish
    "dish":              "dish",
    "plate":             "dish",

    # camel aliases (only specific names, not "toy" which is too broad)
    "camel":             "camel",
    "toy camel":         "camel",

    # head_shoulders_care aliases
    # NOTE: "shampoo" and "bottle" removed — "shampoo" is its own GT object (id=52)
    # and "bottle" is too generic; both caused false-positive normalization.
    "head shoulders":          "head_shoulders_care",
    "head & shoulders":        "head_shoulders_care",
    "head shoulders shampoo":  "head_shoulders_care",
    "h&s shampoo":             "head_shoulders_care",

    # shampoo (GT id=52 — distinct from head_shoulders_care)
    "shampoo":                 "shampoo",
    "shampoo bottle":          "shampoo",
    "hair shampoo":            "shampoo",

    # cheetos — treat as a misidentification of cracker_box in this dataset
    "cheetos":           "cracker_box",
    "cheese":            "cracker_box",

    # mug
    "mug":               "mug",
    "cup":               "mug",
    "coffee mug":        "mug",

    # power_drill
    "drill":             "power_drill",
    "power drill":       "power_drill",

    # scissors
    "scissors":          "scissors",
    "shears":            "scissors",

    # strawberry
    "strawberry":        "strawberry",

    # plum
    "plum":              "plum",

    # knife
    "knife":             "knife",
    "kitchen knife":     "knife",
    "chef knife":        "knife",
    "blade":             "knife",

    # toy airplanes
    "toy airplane":      "toy_airplane_d",
    "airplane":          "toy_airplane_d",
    "plane":             "toy_airplane_d",

    # sugar_box
    "sugar box":         "sugar_box",
    "sugar":             "sugar_box",

    # stapler
    "stapler":           "stapler",

    # large_clamp / extra_large_clamp
    "clamp":             "large_clamp",
    "large clamp":       "large_clamp",
    "extra large clamp": "extra_large_clamp",

    # glue
    "glue":              "glue",
    "glue stick":        "glue",

    # thread
    "thread":            "thread",
    "spool":             "thread",

    # cleanser / skincare bottles
    "cleanser":          "kispa_cleanser",
    "kispa":             "kispa_cleanser",
    "face wash":         "dabao_facewash",
    "facewash":          "dabao_facewash",
    "mouth rinse":       "nzskincare_mouth_rinse",
    "mouthwash":         "nzskincare_mouth_rinse",

    # weiquan / darlie / dabao (brand-specific)
    "weiquan":           "weiquan",
    "darlie":            "darlie_box",
    "toothpaste box":    "darlie_box",
    "dabao":             "dabao_wash_soup",

    # hanoi tower
    "hanoi":             "hanoi_tower",
    "tower":             "hanoi_tower",
    "stacking toy":      "hanoi_tower",

    # sum37
    "sum37":             "sum37_secret_repair",
    "secret repair":     "sum37_secret_repair",
}


# ---------------------------------------------------------------------------
# Semantic object alias map
#
# Maps each GT canonical name → set of raw VLM strings that should be
# considered a *correct* identification for evaluation purposes, even when
# the exact canonical name is not produced.  This decouples evaluation
# recall from the normalization step and lets us credit the model for
# "shampoo bottle" when the GT is "head_shoulders_care".
# ---------------------------------------------------------------------------

SEMANTIC_OBJECT_ALIASES: dict[str, set[str]] = {
    # Branded shampoo / hair-care bottles
    "head_shoulders_care": {
        "shampoo", "shampoo bottle", "head & shoulders", "head shoulders",
        "head and shoulders", "bottle", "hair care", "conditioner",
        "hair shampoo",
    },
    "head_shoulders_supreme": {
        "shampoo", "shampoo bottle", "head & shoulders", "head shoulders",
        "head and shoulders", "head & shoulders supreme",
        "head shoulders supreme", "bottle", "hair care",
    },
    "shampoo": {
        "shampoo", "shampoo bottle", "hair shampoo", "bottle",
    },
    # Chinese branded skincare / beverage
    "dabao_wash_soup": {
        "lotion", "bottle", "soap", "face wash", "cleanser",
        "skincare", "skin care", "moisturizer", "cream", "dabao",
    },
    "dabao_facewash": {
        "face wash", "facewash", "cleanser", "foam cleanser",
        "lotion", "bottle", "dabao",
    },
    "dabao_sod": {
        "lotion", "bottle", "moisturizer", "cream", "skincare", "dabao",
    },
    "nzskincare_mouth_rinse": {
        "mouthwash", "mouth rinse", "bottle", "oral care",
        "rinse", "toothpaste", "nzskincare",
    },
    "kispa_cleanser": {
        "cleanser", "face wash", "facewash", "lotion", "bottle",
        "skincare", "kispa",
    },
    "weiquan": {
        "bottle", "water bottle", "beverage", "drink", "juice",
        "container", "weiquan",
    },
    # Dishes / plates
    "dish": {
        "plate", "dish", "flat plate", "ceramic plate",
        "white plate", "serving plate", "round plate",
    },
    # Toys
    "mario": {
        "toy", "toy figure", "mario", "nintendo", "character",
        "action figure", "figurine", "game character",
    },
    "large_elephant": {
        "elephant", "toy elephant", "stuffed animal", "plush",
        "animal toy", "toy", "stuffed elephant",
    },
    "gorilla": {
        "gorilla", "toy gorilla", "stuffed animal", "plush",
        "animal toy", "toy", "ape",
    },
    "hippo": {
        "hippo", "hippopotamus", "toy hippo", "stuffed animal",
        "animal toy", "toy",
    },
    "camel": {
        "camel", "toy camel", "stuffed animal", "animal toy", "toy",
    },
    "toy_airplane_i": {
        "toy airplane", "airplane", "plane", "toy plane",
        "model airplane", "toy aircraft", "aircraft", "toy jet",
    },
    "toy_airplane_j": {
        "toy airplane", "airplane", "plane", "toy plane",
        "model airplane", "toy aircraft", "aircraft", "toy jet",
    },
    "toy_airplane_d": {
        "toy airplane", "airplane", "plane", "toy plane",
        "model airplane", "aircraft",
    },
    "toy_airplane_f": {
        "toy airplane", "airplane", "plane", "toy plane",
        "model airplane", "aircraft",
    },
    # Small produce (easily confused)
    "pear": {"pear", "green fruit", "fruit", "yellow fruit", "green pear"},
    "peach": {"peach", "orange fruit", "fruit", "nectarine", "yellow peach"},
    "plum": {"plum", "purple fruit", "fruit", "red plum"},
    "strawberry": {"strawberry", "berry", "fruit", "red berry", "red fruit"},
    "banana": {"banana", "yellow fruit", "fruit", "yellow banana"},
    # Tools
    "flat_screwdriver": {
        "screwdriver", "flat screwdriver", "flathead screwdriver",
        "flat head screwdriver", "tool",
    },
    "knife": {
        "knife", "kitchen knife", "chef knife", "blade",
        "cutting tool", "carving knife",
    },
    "power_drill": {
        "drill", "power drill", "electric drill", "tool",
        "cordless drill", "black+decker", "black and decker", "black decker",
    },
    # Common household
    "cracker_box": {
        "cracker box", "cracker", "cheez-it", "cheezit", "crackers",
        "cheese crackers", "snack box", "box", "food box", "cheese",
    },
    "mug": {"mug", "cup", "coffee mug", "tea mug", "ceramic mug", "glass"},
    "scissors": {"scissors", "shears", "cutting tool"},
    "tape": {"tape", "duct tape", "roll of tape", "tape roll"},
    "stapler": {"stapler", "office stapler"},
    "thread": {"thread", "spool", "spool of thread", "yarn", "string"},
    "glue": {"glue", "glue stick", "adhesive"},
    "sugar_box": {"sugar box", "sugar", "box", "food box"},
    "darlie_box": {
        "toothpaste", "toothpaste box", "darlie", "dental care", "box",
    },
    "hanoi_tower": {
        "hanoi", "hanoi tower", "tower", "stacking toy",
        "ring toy", "stacking rings",
    },
}

# Canonical name → expected best grasp type (from paper taxonomy)
# NOTE: "Pen-holding grasp" is the exact string from GRASP_TYPES in pipeline.py
CANONICAL_GRASP_MAP = {
    # ── realsense flat-dataset objects ──────────────────────────
    "cracker_box":            "Lateral pinch",
    "banana":                 "Lateral pinch",
    "peach":                  "Spherical grasp",
    "pear":                   "Spherical grasp",
    "flat_screwdriver":       "Pen-holding grasp",
    "tape":                   "Cylindrical grasp",
    "dish":                   "Lateral pinch",
    "camel":                  "Pinch grasp",
    "head_shoulders_care":    "Cylindrical grasp",
    # ── train_1 known objects ────────────────────────────────────
    "sugar_box":              "Lateral pinch",
    "large_clamp":            "Lateral pinch",
    "extra_large_clamp":      "Lateral pinch",
    "hanoi_tower":            "Pinch grasp",
    "mario":                  "Spherical grasp",
    "stapler":                "Lateral pinch",
    "shampoo":                "Cylindrical grasp",
    "glue":                   "Cylindrical grasp",
    "thread":                 "Cylindrical grasp",
    # ── train_1 newly-resolved objects ───────────────────────────
    "mug":                    "Cylindrical grasp",
    "power_drill":            "Cylindrical grasp",
    "scissors":               "Lateral pinch",
    "strawberry":             "Spherical grasp",
    "plum":                   "Spherical grasp",
    "knife":                  "Pen-holding grasp",
    "toy_airplane_d":         "Pinch grasp",
    "toy_airplane_f":         "Pinch grasp",
    "toy_airplane_i":         "Pinch grasp",
    "toy_airplane_j":         "Pinch grasp",
    "sum37_secret_repair":    "Cylindrical grasp",
    "dabao_wash_soup":        "Cylindrical grasp",
    "nzskincare_mouth_rinse": "Cylindrical grasp",
    "dabao_sod":              "Cylindrical grasp",
    "kispa_cleanser":         "Cylindrical grasp",
    "large_elephant":         "Spherical grasp",
    "gorilla":                "Spherical grasp",
    "weiquan":                "Cylindrical grasp",
    "darlie_box":             "Lateral pinch",
    "dabao_facewash":         "Cylindrical grasp",
    "head_shoulders_supreme": "Cylindrical grasp",
    "hippo":                  "Spherical grasp",
}

# ---------------------------------------------------------------------------
# Semantic grasp compatibility map
#
# Per the paper (Section 4.2): "even when the object classification was
# inaccurate, the resulting label of shape was plausible, prompting manual
# acceptance as a valid classification."
#
# Each object maps to the *set* of grasps a human rater would consider
# reasonable, not just the single best answer.  This is used for the
# semantic accuracy metric (in addition to the exact-match metric).
# ---------------------------------------------------------------------------

SEMANTIC_GRASP_COMPAT_MAP = {
    # Boxes / flat cartons — lateral pinch is canonical; cylindrical also works
    # for taller boxes gripped from the sides
    "cracker_box":            {"Lateral pinch", "Cylindrical grasp"},
    "sugar_box":              {"Lateral pinch", "Cylindrical grasp"},
    "darlie_box":             {"Lateral pinch", "Cylindrical grasp"},

    # Fruits — rounded; spherical is canonical but cylindrical/pinch reasonable
    "banana":                 {"Lateral pinch", "Cylindrical grasp"},
    "peach":                  {"Spherical grasp", "Pinch grasp"},
    "pear":                   {"Spherical grasp", "Pinch grasp"},
    "strawberry":             {"Spherical grasp", "Pinch grasp"},
    "plum":                   {"Spherical grasp", "Pinch grasp"},

    # Tool-type objects with long handles
    "flat_screwdriver":       {"Pen-holding grasp", "Lateral pinch"},
    "knife":                  {"Pen-holding grasp", "Lateral pinch"},
    "power_drill":            {"Cylindrical grasp", "Lateral pinch"},

    # Cylindrical bottles / containers
    "tape":                   {"Cylindrical grasp", "Hook grasp"},
    "head_shoulders_care":    {"Cylindrical grasp"},
    "shampoo":                {"Cylindrical grasp"},
    "glue":                   {"Cylindrical grasp", "Pen-holding grasp"},
    "sum37_secret_repair":    {"Cylindrical grasp"},
    "dabao_wash_soup":        {"Cylindrical grasp"},
    "nzskincare_mouth_rinse": {"Cylindrical grasp"},
    "dabao_sod":              {"Cylindrical grasp"},
    "kispa_cleanser":         {"Cylindrical grasp"},
    "weiquan":                {"Cylindrical grasp"},
    "dabao_facewash":         {"Cylindrical grasp"},
    "head_shoulders_supreme": {"Cylindrical grasp"},

    # Mug — cylindrical body or hook-finger-in-handle
    "mug":                    {"Cylindrical grasp", "Hook grasp"},

    # Scissors — lateral pinch to close them; hook also possible
    "scissors":               {"Lateral pinch", "Hook grasp"},

    # Thread spool
    "thread":                 {"Cylindrical grasp", "Pinch grasp"},

    # Flat / wide objects
    "dish":                   {"Lateral pinch", "Pinch grasp"},
    "stapler":                {"Lateral pinch"},
    "large_clamp":            {"Lateral pinch"},
    "extra_large_clamp":      {"Lateral pinch"},

    # Toy figures / animals — rounded; spherical or pinch
    "camel":                  {"Pinch grasp", "Spherical grasp"},
    "mario":                  {"Spherical grasp", "Pinch grasp"},
    "large_elephant":         {"Spherical grasp", "Pinch grasp"},
    "gorilla":                {"Spherical grasp", "Pinch grasp"},
    "hippo":                  {"Spherical grasp", "Pinch grasp"},

    # Toy airplanes — pinch the fuselage or wings
    "toy_airplane_d":         {"Pinch grasp", "Lateral pinch"},
    "toy_airplane_f":         {"Pinch grasp", "Lateral pinch"},
    "toy_airplane_i":         {"Pinch grasp", "Lateral pinch"},
    "toy_airplane_j":         {"Pinch grasp", "Lateral pinch"},

    # Misc
    "hanoi_tower":            {"Pinch grasp", "Lateral pinch"},
}


def normalize_object_label(raw_label: str) -> str:
    """
    Normalize a raw VLM object label to its canonical dataset name.

    Strategy
    --------
    1. Exact match against OBJECT_ALIAS_MAP → return canonical immediately.
    2. Fuzzy substring match — only for aliases that are at least 5 characters
       long to prevent single-word false-positive mappings (e.g. "pen" matching
       inside "open" or "pencil").
    3. If no match found → return the cleaned lower-case label as-is so the
       caller gets the raw VLM output rather than a forced wrong canonical name.

    Args:
        raw_label: str — the label produced by SM3 of the VLM pipeline.

    Returns:
        str — canonical label, or the lowercased original if no alias matches.
    """
    if raw_label is None:
        return "unknown"

    key = raw_label.strip().lower()

    # Step 1: exact match
    if key in OBJECT_ALIAS_MAP:
        return OBJECT_ALIAS_MAP[key]

    # Step 2: fuzzy substring match — require alias to be at least 5 chars to
    # avoid over-eager single-word matches that cause false positive labels.
    # Only check ``alias in key`` (the alias phrase appears inside the VLM label)
    # NOT ``key in alias`` which would fire when the VLM outputs a short word
    # that happens to be a substring of a longer alias (e.g. "shampoo" matches
    # inside "head shoulders shampoo" even though shampoo is its own GT object).
    for alias, canonical in OBJECT_ALIAS_MAP.items():
        if len(alias) >= 5 and alias in key:
            return canonical

    # Step 3: no confident match — return the raw cleaned label so the pipeline
    # records the actual VLM output rather than a fabricated canonical name.
    return key


def semantic_object_match(raw_vlm_label: str, gt_canonical: str) -> bool:
    """
    Return True if *raw_vlm_label* is a semantically acceptable description
    of *gt_canonical*, even when exact normalization fails.

    Matching order
    --------------
    1. Exact canonical match after normalization.
    2. GT canonical appears as a substring of the VLM label.
    3. VLM label (or any word in it) appears in SEMANTIC_OBJECT_ALIASES[gt_canonical].
    4. VLM label normalizes to a name that matches gt_canonical via the alias map
       (reverse lookup).

    Args:
        raw_vlm_label : raw string from the VLM (SM3 output, un-normalized).
        gt_canonical  : canonical GT object name (from GT_OBJ_ID_TO_NAME).

    Returns:
        bool
    """
    if not raw_vlm_label or not gt_canonical:
        return False

    vlm_clean = raw_vlm_label.strip().lower()
    gt_clean  = gt_canonical.strip().lower()

    # 1. Direct canonical match after normalization
    if normalize_object_label(vlm_clean) == gt_clean:
        return True

    # 2. GT canonical (with underscores replaced) appears in VLM label
    gt_readable = gt_clean.replace("_", " ")
    if gt_readable in vlm_clean or gt_clean in vlm_clean:
        return True

    # 3. Check against SEMANTIC_OBJECT_ALIASES for this GT object
    aliases = SEMANTIC_OBJECT_ALIASES.get(gt_canonical, set())
    if vlm_clean in aliases:
        return True
    # also check each word/phrase token in vlm_clean against the alias set
    for alias in aliases:
        if alias in vlm_clean:
            return True

    return False


def parse_annotation_xml(xml_path):
    """
    Parse a GraspNet annotation XML file.

    Args:
        xml_path: str — path to the .xml annotation file.

    Returns:
        list of dicts, each with keys:
            obj_id (int), obj_name (str), canonical_name (str),
            position (list[float]), orientation (list[float])
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    objects = []
    for obj_elem in root.findall("obj"):
        obj_id = int(obj_elem.find("obj_id").text.strip())
        obj_name_raw = obj_elem.find("obj_name").text.strip()

        # Clean the .ply extension and number prefix
        clean_name = obj_name_raw.replace(".ply", "")
        # Remove leading YCB number prefix like "011_" or "003_"
        clean_name = re.sub(r"^\d+_", "", clean_name)

        pos_text = obj_elem.find("pos_in_world").text.strip()
        ori_text = obj_elem.find("ori_in_world").text.strip()

        objects.append({
            "obj_id":         obj_id,
            "obj_name_raw":   obj_name_raw,
            "canonical_name": GT_OBJ_ID_TO_NAME.get(obj_id, clean_name),
            "position":       [float(x) for x in pos_text.split()],
            "orientation":    [float(x) for x in ori_text.split()],
        })

    return objects


def build_gt_object_map(annotation_xml):
    """
    Build a mapping from obj_id to canonical object name from an annotation file.

    Args:
        annotation_xml: str — path to the XML file.

    Returns:
        dict {int: str} — obj_id → canonical_name
    """
    objects = parse_annotation_xml(annotation_xml)
    return {o["obj_id"]: o["canonical_name"] for o in objects}
