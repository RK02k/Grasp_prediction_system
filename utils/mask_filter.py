def filter_small_masks(masks, min_area=1000):

    filtered_masks = []

    for mask in masks:

        if mask['area'] > min_area:

            filtered_masks.append(mask)

    return filtered_masks