from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
import matplotlib.pyplot as plt

sam = sam_model_registry["vit_l"](
    checkpoint="sam_vit_l_0b3195.pth"
)

mask_generator = SamAutomaticMaskGenerator(sam)

image = cv2.imread("test.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

masks = mask_generator.generate(image)

print(f"Number of masks: {len(masks)}")

import numpy as np

# Draw masks as colored overlays
overlay = image.copy().astype(np.float32) / 255.0
np.random.seed(42)
for mask_data in masks:
    color = np.random.rand(3)
    m = mask_data["segmentation"]
    overlay[m] = overlay[m] * 0.4 + color * 0.6

overlay = (overlay * 255).astype(np.uint8)

# Save result
out_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
cv2.imwrite("output_masks.png", out_bgr)
print("Saved: output_masks.png")