import random
from PIL import Image
import cv2


class VerticalStripeCrop(object):
    """

    """
    def __init__(self, stripe_width_range=(10, 50)):  # Adjust width range as needed
        self.stripe_width_range = stripe_width_range

    def __call__(self, img):
        img_width, img_height = img.size  # Get image dimensions

        stripe_width = random.randint(*self.stripe_width_range)
        if stripe_width > img_width: # Avoid errors when the stripe width is larger than the image width
            stripe_width = img_width

        left = random.randint(0, img_width - stripe_width)  # Random left edge

        # Crop the vertical stripe and resize
        cropped_img = img.crop((left, 0, left + stripe_width, img_height))
        cropped_img = cropped_img.resize((img_width, img_height), Image.LANCZOS)

        return cropped_img


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
    """

    """
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