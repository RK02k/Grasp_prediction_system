"""
VLM-based label generation pipeline — Section 3.3 of the paper.

Four sequential submodules, each using:
  - A main VLM   : runs N_DECISIONS times → majority vote for the answer.
  - A judgment VLM: runs N_DECISIONS times → majority vote to verify the answer.
  - Up to N_RETRIES retry cycles when format is wrong or judgment rejects.

Both the overall scene overlay and the individual object crop are passed as
images to every VLM call so the model has full scene context.
"""

import re
from collections import Counter


# ---------------------------------------------------------------------------
# Grasp taxonomy (Fugl-Meyer + 2 additional types)
# ---------------------------------------------------------------------------

GRASP_TYPES = [
    "Cylindrical grasp",
    "Spherical grasp",
    "Pinch grasp",
    "Lateral pinch",
    "Hook grasp",
    "Pen-holding grasp",
    "Button-press grasp",
]

# ---------------------------------------------------------------------------
# Prompt templates  (verbatim from Appendix A of the paper)
# ---------------------------------------------------------------------------

# Submodule 1 — object vs. background
SM1_MAIN = (
    "The image shows an overall scene of a desktop with objects, "
    "as well as a specific region within the image.\n"
    "Please determine whether this region is an object within the overall scene or not.\n"
    "If it is an object, answer [True]; "
    "if it is background (e.g., desktop, wall), answer [False].\n"
    "If the region is too small, it should also be [False].\n"
    "The answer should be wrapped in square brackets, for example: [True] or [False].\n"
    "Do not provide any additional response."
)

SM1_JUDGE = (
    "The image shows an overall scene of a desktop with objects, "
    "as well as a specific region within the image.\n"
    "Here are some text descriptions about whether this region is an object "
    "or background (e.g., wall or desktop): <result>\n"
    "Please judge: Whether the description is reasonable — "
    "if it describes an object, it should be True, otherwise False.\n"
    "If the region is too small, it should also be False.\n"
    "The answer should be wrapped in square brackets, for example: [True] or [False].\n"
    "Do not provide any additional response."
)

# Submodule 2 — object count
SM2_MAIN = (
    "The image shows an overall scene of a desktop with objects, "
    "as well as a specific region within the image.\n"
    "Please confirm how many objects are shown in the segmented region.\n"
    "Answer only with the number in the format: [number].\n"
    "For example, if there is 1 object, answer [1].\n"
    "The answer should be wrapped in square brackets.\n"
    "Do not provide any additional response."
)

SM2_JUDGE = (
    "The image shows an overall scene of a desktop with objects, "
    "as well as a specific region within the image.\n"
    "Here is a description of how many objects are in the "
    "segmented region (e.g., 1, 2): <result>\n"
    "Please judge: Whether the description is reasonable.\n"
    "The answer should be wrapped in square brackets, for example: [True] or [False].\n"
    "Do not provide any additional response."
)

# Submodule 3 — object type / semantic label
SM3_MAIN = (
    "The image shows an overall scene of a desktop with objects, "
    "as well as a specific region within the image.\n"
    "Identify the single object in the segmented region.\n"
    "If you are certain of the specific product or object name, use it.\n"
    "If you are not certain, describe the object category instead.\n"
    "Use short, precise descriptions such as:\n"
    "  shampoo bottle, toy airplane, banana, power drill, "
    "kitchen knife, mug, scissors, cracker box, fruit, stapler\n"
    "Return only one answer wrapped in square brackets, for example: [banana].\n"
    "Do not describe background, color, or position.\n"
    "Do not provide any additional response."
)

SM3_JUDGE = (
    "The image shows an overall scene of a desktop with objects, "
    "as well as a specific region within the image.\n"
    "Here is a description of what the object in that region is: <result>\n"
    "Please judge: Whether the description is reasonable.\n"
    "The answer should be wrapped in square brackets, for example: [True] or [False].\n"
    "Do not provide any additional response."
)

# Submodule 4 — grasp type  (object label injected at runtime via .format())
SM4_MAIN = (
    "The image shows an overall scene of a desktop with objects, "
    "as well as a specific region within the image.\n"
    "The object in this region is: {object_label}.\n"
    "Identify the best human grasp type for picking up this object.\n"
    "Guidance:\n"
    "  - Use [Cylindrical grasp] for bottles, mugs, jars, cylinders.\n"
    "  - Use [Lateral pinch] for flat boxes, dishes, thin books, cards.\n"
    "  - Use [Spherical grasp] for round fruits, balls, globe-shaped objects.\n"
    "  - Use [Pen-holding grasp] for pens, screwdrivers, thin rods.\n"
    "  - Use [Hook grasp] for handles, loops, bags.\n"
    "  - Use [Pinch grasp] only for very small objects < 3 cm (coins, small clips).\n"
    "  - Use [Button-press grasp] only for objects that require pressing a button.\n"
    "Choose ONLY from the following options:\n"
    + "\n".join(f"[{g}]" for g in GRASP_TYPES) + "\n"
    "Return ONLY one answer wrapped in square brackets.\n"
    "Do not provide any additional response."
)

SM4_JUDGE = (
    "The image shows an overall scene of a desktop with objects, "
    "as well as a specific region within the image.\n"
    "The object in this region is: {object_label}.\n"
    "Here is a suggested grasp type: <result>\n"
    "Please judge: Whether this grasp type is reasonable for this object.\n"
    "The answer should be wrapped in square brackets, for example: [True] or [False].\n"
    "Do not provide any additional response."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_bracketed(text):
    """Return the content of the first [...] found in text, or None."""
    match = re.search(r'\[([^\[\]]+)\]', text)
    if match:
        return match.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Core submodule runner
# ---------------------------------------------------------------------------

def run_submodule(
    model,
    scene_path,
    crop_path,
    main_prompt,
    judgment_prompt,
    n_decisions=11,
    n_retries=4,
):
    """
    Execute one VLM submodule with majority voting and judgment verification.

    Algorithm (Section 3.3):
      For each retry attempt:
        1. Run the main VLM `n_decisions` times.
        2. Extract bracketed results; collect only valid (parseable) responses.
        3. If no valid responses → continue to next retry.
        4. Take the majority vote as `majority`.
        5. Run the judgment VLM `n_decisions` times with `majority` embedded.
        6. Take majority vote of judgment responses.
        7. If judgment is [True]  → accept and return `majority`.
           If judgment is [False] → retry from step 1.
           If judgment fails to parse → accept `majority` (best-effort).
      After all retries exhausted → return the last valid `majority` (best-effort).

    Args:
        model        : QwenGraspModel instance.
        scene_path   : path to the overall scene segmentation overlay.
        crop_path    : path to the individual object crop image.
        main_prompt  : prompt string for the main VLM.
        judgment_prompt: prompt string for the judgment VLM;
                         the placeholder <result> is replaced with the majority answer.
        n_decisions  : number of inference rounds per majority vote.
        n_retries    : maximum number of full retry cycles.

    Returns:
        str or None — the extracted (un-bracketed) result, or None if all retries fail.
    """
    last_majority = None

    for attempt in range(n_retries):

        # --- Step 1 & 2: main VLM majority vote ---
        raw_responses = []
        for _ in range(n_decisions):
            raw = model.ask([scene_path, crop_path], main_prompt)
            parsed = extract_bracketed(raw)
            if parsed:
                raw_responses.append(parsed)

        # --- Step 3: format failure → retry ---
        if not raw_responses:
            continue

        # --- Step 4: majority vote ---
        majority = Counter(raw_responses).most_common(1)[0][0]
        last_majority = majority

        # --- Step 5 & 6: judgment VLM majority vote ---
        j_prompt = judgment_prompt.replace("<result>", majority)
        j_responses = []
        for _ in range(n_decisions):
            raw = model.ask([scene_path, crop_path], j_prompt)
            parsed = extract_bracketed(raw)
            if parsed:
                j_responses.append(parsed.lower().strip())

        # --- Step 7: evaluate judgment ---
        if not j_responses:
            # Judgment produced no parseable output → accept majority (best-effort)
            return majority

        j_majority = Counter(j_responses).most_common(1)[0][0]

        if j_majority == "true":
            return majority

        # j_majority == "false" → fall through to next retry cycle

    # All retries exhausted — return best-effort result
    return last_majority
