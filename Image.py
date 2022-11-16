import numpy as np


class Image:
    filename        = ""
    rotation_factor = 0

    original_image  = np.array(0)
    output_image    = np.array(0)

    finger_lengths  = []
    countours       = []

    def __init__(self, image_original, filename):
        image_original = image_original



