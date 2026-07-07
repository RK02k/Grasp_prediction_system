from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator

class SAMSegmenter:

    def __init__(self, checkpoint_path):

        sam = sam_model_registry["vit_l"](
            checkpoint=checkpoint_path
        )

        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=16,          # fewer but more confident grid points
            pred_iou_thresh=0.90,        # require higher mask quality
            stability_score_thresh=0.95, # require more stable masks
            crop_n_layers=0,             # CRITICAL: disable sub-crop re-runs
                                         # (crop_n_layers=1 is the main source of
                                         #  fragment / duplicate masks)
            min_mask_region_area=1000    # discard tiny fragments early
        )

    def generate_masks(self, image):

        masks = self.mask_generator.generate(image)

        return masks