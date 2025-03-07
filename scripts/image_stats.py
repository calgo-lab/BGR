import os
from PIL import Image
import numpy as np

path_pictures = '../data/BGR/Profilbilder_no_ruler_no_sky/'

# Lists to store widths and heights
widths = []
heights = []

# Iterate over images in the folder
for filename in os.listdir(path_pictures):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(path_pictures, filename)
        with Image.open(img_path) as img:
            width, height = img.size
            widths.append(width)
            heights.append(height)

# Compute median dimensions
median_width = np.median(widths)
median_height = np.median(heights)

print(f"Median Width: {np.median(widths)}")
print(f"Median Height: {np.median(heights)}")
print(f"Mean Width: {np.mean(widths)}")
print(f"Mean Height: {np.mean(heights)}")