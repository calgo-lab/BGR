import numpy as np
import cv2

def center_crop(img_file, x_len, y_len):
    """Crops the image to a bounding box y_len x x_len centered around center of the image.

    :param img_file:
    :param x_len:
    :param y_len:
    :return:
    """

    img = cv2.imread(img_file) #Image.open(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # OpenCV uses BGR format

    center_x, center_y = int(img.shape[0] / 2), int(img.shape[1] / 2)

    left  = center_x - x_len
    right = center_x + x_len
    upper = center_y - y_len
    lower = center_y + y_len

    return img[upper:lower, left:right]


# Definiere die Custom-Transformation für das Zuschneiden des Bildes
class CenterCropTransform:
    def __init__(self, crop_width=240, crop_height=450):
        self.crop_width = crop_width
        self.crop_height = crop_height

    def __call__(self, img):
        # Bildgröße ermitteln
        width, height = img.size

        # Berechne die Koordinaten für den zentrierten Ausschnitt
        left = (width - self.crop_width) // 2
        upper = (height - self.crop_height) // 2
        right = left + self.crop_width
        lower = upper + self.crop_height

        # Zuschnitt des Bildes
        return img.crop((left, upper, right, lower))


def remove_ruler(image):

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


def create_rect_mask(height, width, untergrenze_list):

    mask = np.zeros((height, width), dtype=int)
    current_region = 0

    # Some horizons have the lowest cutoff below -1 meter in the images
    # Add an extra border at 100 to avoid errors when drawing the rectangles
    if np.max(untergrenze_list) < 100.:
        untergrenze_list.append(100.)

    for i in range(height):
        if i > untergrenze_list[current_region] * height/100.:
            current_region += 1

        mask[i, :] = current_region+1

    return mask


def group_patches(untergrenze_list, seg_mask):

    height = seg_mask.shape[0] # same as original image
    untergrenze_pix_y = np.asarray(untergrenze_list) * height/100

    combi_mask = np.zeros_like(seg_mask)
    for patch_id in np.unique(seg_mask):

        # Get indexes of patch pixels
        patch_inds = np.where(seg_mask == patch_id) # tuple of y and x coords

        # Check whether the patch is intersecting any of the rectangle borders
        # It would mean that a boundary lies between the minimal and maximal row index of that patch
        intersecting_border = [ u for u in untergrenze_pix_y if np.min(patch_inds[0]) <= u <= np.max(patch_inds[0]) ]
        if not intersecting_border:
            # Get the index of the first rectangle border that is below the lowest row index of the patch
            next_lower_border = np.min([u for u in untergrenze_pix_y if u > np.max(patch_inds[0])])
            combi_mask[patch_inds] = np.where(untergrenze_pix_y == next_lower_border)[0]

        # Check whether the patch has more pixels in the rectangle above or below the border
        # Count how many row indexes are above the y coord of the intersecting border and below
        # Assign the patch the label of the rectangle intersecting the patch the most
        else:
            num_pix_above = np.count_nonzero(patch_inds[0] <= intersecting_border[0])
            num_pix_below = np.count_nonzero(patch_inds[0] > intersecting_border[0])

            if num_pix_above > num_pix_below:
                combi_mask[patch_inds] = np.where(untergrenze_pix_y == intersecting_border[0])[0]
            else:
                combi_mask[patch_inds] = np.where(untergrenze_pix_y == intersecting_border[0])[0] + 1

    return combi_mask