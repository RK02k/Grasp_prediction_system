import os
import cv2
import numpy as np

def save_object_crops(image, masks, output_dir="outputs/crops",
                      depth_image=None):

    os.makedirs(output_dir, exist_ok=True)

    if depth_image is not None:
        depth_dir = os.path.join(output_dir, "depth")
        os.makedirs(depth_dir, exist_ok=True)

    for idx, mask in enumerate(masks):

        segmentation = mask['segmentation']

        bbox = mask['bbox']

        x, y, w, h = map(int, bbox)

        crop = image[y:y+h, x:x+w]

        mask_crop = segmentation[y:y+h, x:x+w]

        isolated = np.zeros_like(crop)

        isolated[mask_crop] = crop[mask_crop]

        save_path = f"{output_dir}/object_{idx}.png"
        print(f"Processing object {idx}")
        cv2.imwrite(
            save_path,
            cv2.cvtColor(isolated, cv2.COLOR_RGB2BGR)
        )

        print("Saved:", save_path)

        # Save depth crop if depth image provided
        if depth_image is not None:

            depth_crop = depth_image[y:y+h, x:x+w].copy()

            depth_crop[~mask_crop] = 0

            depth_save = os.path.join(
                depth_dir, f"object_{idx}.png"
            )

            cv2.imwrite(depth_save, depth_crop)

            print("Saved depth:", depth_save)