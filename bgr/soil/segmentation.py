import numpy as np
import cv2


def remove_ruler(image):
    """

    :param image:
    :return:
    """

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 200, apertureSize=3)

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=150, maxLineGap=10)

    # Filter lines to identify ruler lines
    ruler_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x1 - x2) < 10:  # Filter for nearly vertical lines
            ruler_lines.append(line)

    # Find bounding box of ruler region
    x_min = min(line[0][0] for line in ruler_lines)
    x_max = max(line[0][2] for line in ruler_lines)

    # Crop the image, excluding the ruler region
    cropped_image = image[:, :x_min] if x_min > image.shape[1] // 2 else image[:, x_max:]

    return cropped_image


def remove_sky(img_path, thresh=250, closing_kernel_size=30):
    """thresh should be chosen fairly high (close to white)
    """

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur the gray image (to smoothen out further isolated white regions not belonging to the sky)
    img_blur = cv2.GaussianBlur(gray, (15, 15), 20)

    # Separate sky background (marked black) from soil (marked white)
    _, thresh_img = cv2.threshold(img_blur, thresh, 255, cv2.THRESH_BINARY_INV)

    # Apply morphological closing to remove small noise
    kernel = np.ones((closing_kernel_size, closing_kernel_size), np.uint8)
    closed_mask = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)

    # Find the indices of the sky pixels (black)
    sky_indices = np.where(closed_mask == 0)

    # If there are at least 10k black pixels left after closing, then accept them as sky background and crop
    if len(sky_indices[0] > 10 ** 4):

        # Find the maximum y-coordinate (row index) of the sky pixels
        lower_bound = np.max(sky_indices[0])
        return img[lower_bound:, :]
    else:
        return img


def create_rect_mask(height, width, lowbound_list):
    """

    :param height:
    :param width:
    :param lowbound_list:
    :return:
    """

    mask = np.zeros((height, width), dtype=int)
    current_region = 0

    # Some horizons have the lowest cutoff below -1 meter in the images
    # Add an extra border at 100 to avoid errors when drawing the rectangles
    if np.max(lowbound_list) < 100.:
        lowbound_list.append(100.)

    for i in range(height):
        if i > lowbound_list[current_region] * height/100.:
            current_region += 1

        mask[i, :] = current_region+1

    return mask


def group_patches(lowbound_list, seg_mask):
    """

    :param lowbound_list:
    :param seg_mask:
    :return:
    """

    height = seg_mask.shape[0] # same as original image
    lowbound_pix_y = np.asarray(lowbound_list) * height/100

    combi_mask = np.zeros_like(seg_mask)
    for patch_id in np.unique(seg_mask):

        # Get indexes of patch pixels
        patch_inds = np.where(seg_mask == patch_id) # tuple of y and x coords

        # Check whether the patch is intersecting any of the rectangle borders
        # It would mean that a boundary lies between the minimal and maximal row index of that patch
        intersecting_border = [ u for u in lowbound_pix_y if np.min(patch_inds[0]) <= u <= np.max(patch_inds[0]) ]
        if not intersecting_border:
            # Get the index of the first rectangle border that is below the lowest row index of the patch
            next_lower_border = np.min([u for u in lowbound_pix_y if u > np.max(patch_inds[0])])
            combi_mask[patch_inds] = np.where(lowbound_pix_y == next_lower_border)[0]

        # Check whether the patch has more pixels in the rectangle above or below the border
        # Count how many row indexes are above the y coord of the intersecting border and below
        # Assign the patch the label of the rectangle intersecting the patch the most
        else:
            num_pix_above = np.count_nonzero(patch_inds[0] <= intersecting_border[0])
            num_pix_below = np.count_nonzero(patch_inds[0] > intersecting_border[0])

            if num_pix_above > num_pix_below:
                combi_mask[patch_inds] = np.where(lowbound_pix_y == intersecting_border[0])[0]
            else:
                combi_mask[patch_inds] = np.where(lowbound_pix_y == intersecting_border[0])[0] + 1

    return combi_mask


def save_horizon_overlays(mask, original_image, base_dir, orig_file):
    """
    Splits a segmented soil mask into separate horizon overlays and saves them.

    Parameters:
        mask (np.ndarray): The segmented mask where each integer represents a horizon.
        original_image (np.ndarray): The original image (same dimensions as the mask).

    """

    # Get the unique horizon values in the mask
    unique_horizon_ids = np.unique(mask)

    for horizon_id in unique_horizon_ids:
        # Create a binary mask for the current horizon
        binary_mask = np.where(mask == horizon_id, 1, 0).astype(np.uint8)

        # Expand the binary mask to 3 channels for filtering
        binary_mask_3c = cv2.merge([binary_mask] * 3)

        # Filter the original image using the binary mask
        filtered_image = original_image * binary_mask_3c
        filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_RGB2BGR)

        # Construct new file names for each horizon patch
        jpg_orig = orig_file.split('/')[-1]
        jpg_base, jpg_ext = jpg_orig.split('.')
        horizon_file = jpg_base + '_hor' + str(horizon_id+1) + '.' + jpg_ext # id+1 to match ids in horizons table

        # Save the overlay as an image
        cv2.imwrite(base_dir + horizon_file, filtered_image)