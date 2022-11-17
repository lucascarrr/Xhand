import os

import cv2 as cv
import numpy as np

from ImageContainer import ImageContainer
from Image import Image


#loading images into objects -> image_container
def helper(path, holder):
    for filename in os.listdir(path):
        if (filename.endswith(".png")):
            source_img  = np.array(cv.imread(os.path.join(path, filename), cv.IMREAD_GRAYSCALE))
            holder.addImage(Image(source_img, filename))

def main():
    input_path  = "/Users/stranger/Desktop/dataset"
    output_path = "/Users/stranger/Desktop/output/"
    holder = ImageContainer()

    print("Running Program")
    helper(input_path, holder)
    holder.writeImages(output_path)


if __name__ == "__main__":
    main()